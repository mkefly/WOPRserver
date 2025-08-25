from __future__ import annotations

import sys
import os
import time
import tarfile
import subprocess
import asyncio
import multiprocessing as mp
from pathlib import Path
from contextlib import contextmanager
from queue import Empty
from multiprocessing import spawn as mp_spawn  # has get_executable() on 3.10+

import pytest

from mlserver.env import Environment
from mlserver.settings import ModelSettings, ModelParameters, Settings
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver.codecs import StringCodec

from woprserver.parallel.worker import Worker
from woprserver.parallel.messages import (
    ModelUpdateMessage,
    ModelUpdateType,
    ModelRequestMessage,
    ModelResponseMessage,
)
from woprserver.parallel.model import ModelMethods

from .fixtures import EnvModel
import contextlib

import pytest
import asyncio



# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
async def _cleanup_pending_tasks():
    yield
    # After each test, cancel leftover tasks like Dispatcher._observe_ack
    tasks = [t for t in asyncio.all_tasks() if not t.done()]
    for t in tasks:
        t.cancel()
    for t in tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await t

def _extract_tarball(src: str, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src, "r:gz") as tf:
        tf.extractall(dst)


def _run_in_env(env_dir: Path, args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a command using the env's own binaries (e.g. bin/python)."""
    bin_dir = env_dir / "bin"
    python_exe = bin_dir / "python"
    if not python_exe.exists():
        # Sometimes it's named python3
        python_exe = bin_dir / "python3"
    if not python_exe.exists():
        raise AssertionError(f"Env python not found under {bin_dir}")

    # Best-effort relocation fix (conda-pack)
    conda_unpack = bin_dir / "conda-unpack"
    if conda_unpack.exists():
        subprocess.run([str(conda_unpack)], check=True, cwd=str(env_dir), timeout=timeout)

    env = os.environ.copy()
    # Keep it clean; avoid host sitecustomize/usersite
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    return subprocess.run(
        [str(python_exe), *args],
        cwd=str(env_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,  # we assert manually to show nice stderr on failure
    )


@contextmanager
def mp_exec(python_exe: str):
    """
    Temporarily override the interpreter used by multiprocessing "spawn".
    Ensures the worker is started with the env's Python.
    """
    try:
        prev = mp_spawn.get_executable()
    except Exception:
        prev = sys.executable
    mp.set_executable(python_exe)
    try:
        yield
    finally:
        mp.set_executable(prev)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_environment_tarball_runs_python_and_imports_sklearn(env_tarball: str, tmp_path):
    """
    Pure sanity: the packed environment should be runnable in isolation and be
    able to import sklearn and print its version.
    """
    env_dir = tmp_path / "env_unpack"
    _extract_tarball(env_tarball, env_dir)

    proc = _run_in_env(env_dir, ["-c", "import sklearn, sys; print(sklearn.__version__)"])
    if proc.returncode != 0:
        raise AssertionError(
            "Env python failed to import sklearn.\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n"
            f"Env dir: {env_dir}"
        )

    version = proc.stdout.strip()
    assert version == "1.6.1", f"Unexpected sklearn version: {version!r}"


@pytest.mark.asyncio
async def test_mlserver_environment_from_tarball_extracts(env_tarball: str, tmp_path):
    """
    If you suspect the wrapper extractor, confirm Environment.from_tarball works.
    This doesn't run Python inside the env; it validates the MLServer extractor path.
    """
    from mlserver.env import compute_hash_of_file

    env_hash = await compute_hash_of_file(env_tarball)
    env_path = str(tmp_path / "env_from_tarball")

    env = await Environment.from_tarball(env_tarball, env_path)
    assert getattr(env, "_env_path") == env_path
    assert env.env_hash == env_hash

    # Bonus: try executing the env python once to ensure basic run works
    proc = _run_in_env(Path(env_path), ["-V"])
    assert proc.returncode == 0, f"Env python -V failed:\nSTDERR:\n{proc.stderr}"
    assert proc.stdout or proc.stderr  # one of them prints version on some builds


@pytest.mark.asyncio
async def test_env_model_in_worker_without_pool(
    _woprserver_settings: Settings,
    env_tarball: str,
    inference_request: InferenceRequest,
    tmp_path: Path,
):
    # 1) Extract env (optionally run conda-unpack once)
    env_path = str(tmp_path / "env_worker")
    env = await Environment.from_tarball(env_tarball, env_path)

    unpack = Path(env_path) / "bin" / "conda-unpack"
    if unpack.exists():
        subprocess.run([str(unpack)], check=True, cwd=env_path)

    # 2) Use the ENV'S PYTHON to spawn the worker (fixes NumPy ABI mismatch)
    ctx = mp.get_context("spawn")
    responses = ctx.Queue()

    with mp_exec(env._exec_path):
        worker = Worker(_woprserver_settings, responses, env=env)
        worker.start()

    try:
        # 3) Load the model
        ms = ModelSettings(
            name="env-model-worker",
            implementation=EnvModel,
            parameters=ModelParameters(),
        )
        load_msg = ModelUpdateMessage.from_model_settings(
            update_type=ModelUpdateType.Load,
            model_settings=ms,
        )
        worker.send_update(load_msg)

        # Wait for load ACK
        def _wait_for(q, msg_id: str, worker: Worker, timeout: int = 30):
            deadline = time.time() + timeout
            while time.time() < deadline:
                if not worker.is_alive():
                    raise AssertionError(f"Worker died early (exitcode={worker.exitcode})")
                try:
                    msg = q.get(timeout=0.25)
                except Empty:
                    continue
                if isinstance(msg, ModelResponseMessage) and msg.id == msg_id:
                    if msg.exception:
                        raise AssertionError(f"Worker error on load: {msg.exception}")
                    return
            raise TimeoutError("Timed out waiting for worker response")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _wait_for, responses, load_msg.id, worker)

        # 4) Predict
        req_msg = ModelRequestMessage(
            model_name=ms.name,
            model_version=ms.parameters.version,
            method_name=ModelMethods.Predict.value,
            method_args=[inference_request],
        )
        worker.send_request(req_msg)

        def _wait_predict(q, msg_id: str, worker: Worker, timeout: int = 30):
            deadline = time.time() + timeout
            while time.time() < deadline:
                if not worker.is_alive():
                    raise AssertionError(f"Worker died early (exitcode={worker.exitcode})")
                try:
                    msg = q.get(timeout=0.25)
                except Empty:
                    continue
                if isinstance(msg, ModelResponseMessage) and msg.id == msg_id:
                    if msg.exception:
                        raise AssertionError(f"Predict failed: {msg.exception}")
                    return msg.return_value
            raise TimeoutError("Timed out waiting for prediction response")

        resp: InferenceResponse = await loop.run_in_executor(None, _wait_predict, responses, req_msg.id, worker)

        # 5) Check sklearn version
        assert len(resp.outputs) == 1
        [sk_version] = StringCodec.decode_output(resp.outputs[0])
        assert sk_version == "1.6.1"

    finally:
        try:
            await worker.stop()
        except Exception:
            pass

        if worker.is_alive():
            worker.terminate()
        worker.join(timeout=15)

        try:
            responses.close()
            responses.join_thread()
        except Exception:
            pass