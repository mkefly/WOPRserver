# tests/parallel/utils.py
from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import socket
import time
import contextlib
from asyncio import subprocess
from itertools import filterfalse
from string import Template
from typing import Optional

import aiohttp
import yaml
from aiohttp.client_exceptions import (
    ClientConnectorError,
    ClientOSError,
    ServerDisconnectedError,
)
from aiohttp_retry import ExponentialRetry, RetryClient

from mlserver.types import InferenceRequest, InferenceResponse, RepositoryIndexResponse
from mlserver.utils import generate_uuid
from woprserver.logging import get_logger

logger = get_logger()

# --------------------------------------------------------------------------------------
# Networking / ports
# --------------------------------------------------------------------------------------
def get_available_ports(n: int = 1) -> list[int]:
    """
    Reserve `n` free TCP ports and return them. Uses ephemeral sockets.
    """
    ports: set[int] = set()
    while len(ports) < n:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("", 0))
            ports.add(s.getsockname()[1])
        finally:
            s.close()
    return list(ports)

# --------------------------------------------------------------------------------------
# Subprocess helpers
# --------------------------------------------------------------------------------------
async def _run(cmd: str) -> None:
    """
    Backwards-compatible runner that executes a *shell* command and streams its
    combined stdout+stderr to the logger in real time. Raises on non-zero exit.

    Prefer `_run_streaming` for new code.
    """
    logger.info("$ %s", cmd)
    process = await asyncio.create_subprocess_shell(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    assert process.stdout is not None
    async for raw in process.stdout:
        line = raw.decode(errors="replace").rstrip()
        if line:
            logger.info(line)

    return_code = await process.wait()
    if return_code != 0:
        logger.debug("Failed to run command %r", cmd)
        raise RuntimeError(f"Command '{cmd}' failed with code '{return_code}'")


async def _run_streaming(args: list[str], env: Optional[dict] = None, cwd: Optional[str] = None) -> None:
    """
    Execute a command (argv list, no shell), streaming combined stdout+stderr
    to the logger line-by-line. Raises RuntimeError on non-zero exit.
    """
    pretty = " ".join(shlex.quote(x) for x in args)
    logger.info("$ %s", pretty)

    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env, cwd=cwd
    )

    assert proc.stdout is not None
    async for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        if line:
            logger.info(line)

    rc = await proc.wait()
    if rc != 0:
        logger.debug("Failed to run command %s", pretty)
        raise RuntimeError(f"Command failed ({rc}): {pretty}")

# --------------------------------------------------------------------------------------
# Env YAML helpers
# --------------------------------------------------------------------------------------
def _render_env_yml(src_env_yml: str, dst_env_yml: str) -> None:
    """Render a template env.yml with environment variables."""
    with open(src_env_yml) as src_env_file:
        rendered = Template(src_env_file.read()).substitute(os.environ)
    with open(dst_env_yml, "w") as dst_env_file:
        dst_env_file.write(rendered)


def _read_env(env_yml: str) -> dict:
    with open(env_yml) as env_file:
        return yaml.safe_load(env_file.read())


def _write_env(env: dict, env_yml: str) -> None:
    with open(env_yml, "w") as env_file:
        env_file.write(yaml.safe_dump(env, sort_keys=False))


def _is_python(dep: str) -> bool:
    return isinstance(dep, str) and "python" in dep


def _inject_python_version(
    version: tuple[int, int],
    env_yml: str,
    tarball_path: str,
) -> str:
    """
    Inject a Python requirement into an environment YAML, pinning to the
    requested major.minor version while letting conda/mamba resolve to
    the latest patch. We also force `*_cpython` to avoid buggy
    "packaged by conda-forge" builds that break platform._sys_version.
    """
    import yaml

    with open(env_yml) as f:
        env = yaml.safe_load(f)

    major, minor = version
    deps_wo_python = [d for d in env["dependencies"] if not str(d).startswith("python")]

    # Force CPython build, allow latest patch
    python_spec = f"python={major}.{minor}.*"
    env["dependencies"] = [python_spec, *deps_wo_python]

    patched_yml = f"{tarball_path}.tmp.yml"
    with open(patched_yml, "w") as f:
        yaml.safe_dump(env, f)

    return patched_yml



def _pick_env_tool() -> str:
    """
    Choose an environment tool, preferring conda, then mamba, then micromamba.
    """
    for exe in ("conda", "mamba", "micromamba"):
        if shutil.which(exe):
            return exe
    raise FileNotFoundError("No conda/mamba/micromamba found on PATH.")

# --------------------------------------------------------------------------------------
# Pack environment with logs
# --------------------------------------------------------------------------------------
async def _pack(version: tuple[int, int], env_yml: str, tarball_path: str) -> None:
    """
    Create a temporary env for the given Python version from `env_yml` and pack it
    to `tarball_path`, streaming logs. Uses micromamba only.
    """
    fixed_env_yml = _inject_python_version(version, env_yml, tarball_path)
    env_name = f"mlserver-{generate_uuid()}"
    py_major, py_minor = version

    logger.info(
        "Environment pack start | tool=micromamba  python=%s.%s  env_yml=%s  out=%s",
        py_major, py_minor, fixed_env_yml, tarball_path,
    )
    t0 = time.time()

    env_vars = dict(os.environ)
    default_root = os.path.join(os.path.dirname(tarball_path), "mamba-root")
    env_vars.setdefault("MAMBA_ROOT_PREFIX", default_root)

    try:
        # Show tool version (best-effort)
        with contextlib.suppress(Exception):
            await _run_streaming(["micromamba", "--version"], env=env_vars)

        # Create env
        await _run_streaming(
            ["micromamba", "create", "-y", "-n", env_name, "-f", fixed_env_yml],
            env=env_vars,
        )

        # Compute the absolute prefix
        env_prefix = os.path.join(env_vars["MAMBA_ROOT_PREFIX"], "envs", env_name)

        # Pack environment
        conda_pack_exe = shutil.which("conda-pack") or shutil.which("conda-pack.exe")
        pack_common = ["--ignore-missing-files", "--exclude", "lib/python3.1", "-o", tarball_path]
        if conda_pack_exe:
            pack_cmd = [conda_pack_exe, "-p", env_prefix, *pack_common]
            await _run_streaming(pack_cmd, env=env_vars)
        else:
            run_prefix = ["micromamba", "run", "-n", env_name]
            await _run_streaming(run_prefix + ["python", "-m", "conda_pack", *pack_common], env=env_vars)

        elapsed = time.time() - t0
        logger.info("Packed environment to %s (%.1fs)", tarball_path, elapsed)

    finally:
        # Cleanup
        try:
            await _run_streaming(
                ["micromamba", "remove", "-y", "-n", env_name, "--all"],
                env=env_vars,
            )
        except Exception as e:
            logger.warning("Env cleanup failed (non-fatal): %s", e)

def _get_tarball_name(version: tuple[int, int]) -> str:
    major, minor = version
    return f"environment-py{major}{minor}.tar.gz"

# --------------------------------------------------------------------------------------
# Simple REST client used by tests
# --------------------------------------------------------------------------------------
class RESTClient:
    """
    Tiny HTTP client for MLServer integration tests with retry helpers.
    """

    def __init__(self, http_server: str):
        self._session = aiohttp.ClientSession(raise_for_status=True)
        self._http_server = http_server

    async def close(self) -> None:
        await self._session.close()

    async def _retry_get(self, endpoint: str) -> None:
        retry_options = ExponentialRetry(
            attempts=10,
            start_timeout=0.5,
            statuses=[400],
            exceptions={ClientConnectorError, ClientOSError, ServerDisconnectedError, ConnectionRefusedError},
        )
        async with RetryClient(raise_for_status=True, retry_options=retry_options) as retry_client:
            await retry_client.get(endpoint)

    async def wait_until_ready(self) -> None:
        await self._retry_get(f"http://{self._http_server}/v2/health/ready")

    async def wait_until_model_ready(self, model_name: str) -> None:
        await self._retry_get(f"http://{self._http_server}/v2/models/{model_name}/ready")

    async def wait_until_live(self) -> None:
        await self._retry_get(f"http://{self._http_server}/v2/health/live")

    async def ready(self) -> bool:
        res = await self._session.get(f"http://{self._http_server}/v2/health/ready")
        return res.status == 200

    async def live(self) -> bool:
        res = await self._session.get(f"http://{self._http_server}/v2/health/live")
        return res.status == 200

    async def list_models(self) -> RepositoryIndexResponse:
        response = await self._session.post(f"http://{self._http_server}/v2/repository/index", json={"ready": True})
        return RepositoryIndexResponse.parse_raw(await response.text())

    async def infer(self, model_name: str, inference_request: InferenceRequest) -> InferenceResponse:
        response = await self._session.post(
            f"http://{self._http_server}/v2/models/{model_name}/infer", json=inference_request.model_dump()
        )
        return InferenceResponse.parse_raw(await response.text())
