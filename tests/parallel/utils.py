import asyncio
import os
import shlex
import shutil
import socket
import time
import uuid
from asyncio import subprocess
from itertools import filterfalse
from string import Template

import aiohttp
import yaml
from aiohttp.client_exceptions import (
    ClientConnectorError,
    ClientOSError,
    ServerDisconnectedError,
)
from aiohttp_retry import ExponentialRetry, RetryClient
from woprserver.logging import get_logger
from mlserver.types import InferenceRequest, InferenceResponse, RepositoryIndexResponse
from mlserver.utils import generate_uuid

logger = get_logger()

# --------------------------
# Networking / ports
# --------------------------
def get_available_ports(n: int = 1) -> list[int]:
    ports = set()
    while len(ports) < n:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        ports.add(port)
    return list(ports)


# --------------------------
# Subprocess helpers
# --------------------------
async def _run(cmd: str) -> None:
    """
    Backwards-compatible runner that executes a *shell* command and streams its
    combined stdout+stderr to the logger in real time. Raises on non-zero exit.

    Kept for compatibility with any external usages; new code should prefer
    the list-arg runner `_run_streaming`.
    """
    logger.info("$ %s", cmd)
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert process.stdout is not None
    async for raw in process.stdout:
        line = raw.decode(errors="replace").rstrip()
        if line:
            logger.info(line)

    return_code = await process.wait()
    if return_code != 0:
        logger.debug(f"Failed to run command '{cmd}'")
        raise Exception(f"Command '{cmd}' failed with code '{return_code}'")


async def _run_streaming(args: list[str], env: dict | None = None, cwd: str | None = None) -> None:
    """
    Execute a command (argv list, no shell), streaming combined stdout+stderr
    to the logger line-by-line. Raises RuntimeError on non-zero exit.
    """
    pretty = " ".join(shlex.quote(x) for x in args)
    logger.info("$ %s", pretty)

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
        cwd=cwd,
    )

    assert proc.stdout is not None
    async for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        if line:
            logger.info(line)

    rc = await proc.wait()
    if rc != 0:
        logger.debug(f"Failed to run command '{pretty}'")
        raise RuntimeError(f"Command failed ({rc}): {pretty}")


# --------------------------
# Env YAML helpers
# --------------------------
def _render_env_yml(src_env_yml: str, dst_env_yml: str) -> None:
    with open(src_env_yml) as src_env_file:
        env = Template(src_env_file.read()).substitute(os.environ)
    with open(dst_env_yml, "w") as dst_env_file:
        dst_env_file.write(env)


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
    To test the same environment.yml fixture with different Python versions,
    inject the requested version into dependencies.
    """
    env = _read_env(env_yml)
    major, minor = version
    deps = env.get("dependencies", [])
    without_python = list(filterfalse(_is_python, deps))
    with_env_python = [f"python == {major}.{minor}", *without_python]
    env["dependencies"] = with_env_python

    dst_folder = os.path.dirname(tarball_path)
    os.makedirs(dst_folder, exist_ok=True)
    new_env_yml = os.path.join(dst_folder, f"environment-py{major}{minor}.yml")
    _write_env(env, new_env_yml)
    return new_env_yml


def _pick_env_tool() -> str:
    """
    Choose an environment tool, preferring conda, then mamba, then micromamba.
    """
    for exe in ("conda", "mamba", "micromamba"):
        if shutil.which(exe):
            return exe
    raise FileNotFoundError("No conda/mamba/micromamba found on PATH.")


# --------------------------
# Pack environment with logs
# --------------------------
async def _pack(version: tuple[int, int], env_yml: str, tarball_path: str) -> None:
    """
    Create a temporary env for the given Python version from `env_yml` and pack it
    to `tarball_path`, streaming tool logs. Compatible with conda/mamba/micromamba.
    """
    fixed_env_yml = _inject_python_version(version, env_yml, tarball_path)
    env_name = f"mlserver-{generate_uuid()}"
    tool = _pick_env_tool()
    py_major, py_minor = version

    logger.info(
        "Environment pack start | tool=%s  python=%s.%s  env_yml=%s  out=%s",
        tool, py_major, py_minor, fixed_env_yml, tarball_path,
    )
    t0 = time.time()

    # ensure all subprocesses share the same per-test root so we avoid global locks
    env_vars = dict(os.environ)
    if tool in ("mamba", "micromamba"):
        default_root = os.path.join(os.path.dirname(tarball_path), "mamba-root")
        env_vars.setdefault("MAMBA_ROOT_PREFIX", default_root)

    try:
        # Show tool version (best-effort)
        try:
            await _run_streaming([tool, "--version"], env=env_vars)
        except Exception as e:
            logger.debug("Unable to get %s version (non-fatal): %s", tool, e)

        # Create environment
        if tool == "micromamba":
            await _run_streaming([tool, "create", "-y", "-n", env_name, "-f", fixed_env_yml], env=env_vars)
        else:
            await _run_streaming([tool, "env", "create", "-y", "-n", env_name, "-f", fixed_env_yml], env=env_vars)

        # Compute the absolute prefix for this env (so conda-pack doesn't need conda)
        env_prefix = None
        if tool in ("mamba", "micromamba"):
            root = env_vars.get("MAMBA_ROOT_PREFIX")
            if root:
                env_prefix = os.path.join(root, "envs", env_name)

        # Pack environment
        conda_pack_exe = shutil.which("conda-pack") or shutil.which("conda-pack.exe")
        pack_common = ["--ignore-missing-files", "--exclude", "lib/python3.1", "-o", tarball_path]
        if conda_pack_exe:
            pack_cmd = [conda_pack_exe]
            # Prefer -p <prefix> so we don't rely on conda being on PATH
            if env_prefix and os.path.isdir(env_prefix):
                pack_cmd += ["-p", env_prefix]
            else:
                pack_cmd += ["-n", env_name]
            pack_cmd += pack_common
            await _run_streaming(pack_cmd, env=env_vars)
        else:
            # Fallback: run the module inside the env (no CLI needed)
            run_prefix = [tool, "run", "-n", env_name]
            await _run_streaming(run_prefix + ["python", "-m", "conda_pack", *pack_common], env=env_vars)

        elapsed = time.time() - t0
        logger.info("Packed environment to %s (%.1fs)", tarball_path, elapsed)

    finally:
        # Best-effort cleanup
        try:
            if tool == "micromamba":
                await _run_streaming([tool, "remove", "-y", "-n", env_name, "--all"], env=env_vars)
            else:
                await _run_streaming([tool, "env", "remove", "-y", "-n", env_name], env=env_vars)
        except Exception as e:
            logger.warning("Env cleanup failed (non-fatal): %s", e)



def _get_tarball_name(version: tuple[int, int]) -> str:
    major, minor = version
    return f"environment-py{major}{minor}.tar.gz"


# --------------------------
# Simple REST client used by tests
# --------------------------
class RESTClient:
    def __init__(self, http_server: str):
        self._session = aiohttp.ClientSession(raise_for_status=True)
        self._http_server = http_server

    async def close(self) -> None:
        await self._session.close()

    async def _retry_get(self, endpoint: str):
        retry_options = ExponentialRetry(
            attempts=10,
            start_timeout=0.5,
            statuses=[400],
            exceptions={
                ClientConnectorError,
                ClientOSError,
                ServerDisconnectedError,
                ConnectionRefusedError,
            },
        )
        retry_client = RetryClient(raise_for_status=True, retry_options=retry_options)
        async with retry_client:
            await retry_client.get(endpoint)

    async def wait_until_ready(self) -> None:
        endpoint = f"http://{self._http_server}/v2/health/ready"
        await self._retry_get(endpoint)

    async def wait_until_model_ready(self, model_name: str) -> None:
        endpoint = f"http://{self._http_server}/v2/models/{model_name}/ready"
        await self._retry_get(endpoint)

    async def wait_until_live(self) -> None:
        endpoint = f"http://{self._http_server}/v2/health/live"
        await self._retry_get(endpoint)

    async def ready(self) -> bool:
        endpoint = f"http://{self._http_server}/v2/health/ready"
        res = await self._session.get(endpoint)
        return res.status == 200

    async def live(self) -> bool:
        endpoint = f"http://{self._http_server}/v2/health/live"
        res = await self._session.get(endpoint)
        return res.status == 200

    async def list_models(self) -> RepositoryIndexResponse:
        endpoint = f"http://{self._http_server}/v2/repository/index"
        response = await self._session.post(endpoint, json={"ready": True})
        raw_payload = await response.text()
        return RepositoryIndexResponse.parse_raw(raw_payload)

    async def infer(
        self, model_name: str, inference_request: InferenceRequest
    ) -> InferenceResponse:
        endpoint = f"http://{self._http_server}/v2/models/{model_name}/infer"
        response = await self._session.post(
            endpoint, json=inference_request.model_dump()
        )
        raw_payload = await response.text()
        return InferenceResponse.parse_raw(raw_payload)
