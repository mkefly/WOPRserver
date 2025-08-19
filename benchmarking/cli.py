#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path

import click

ROOT = Path(__file__).parent.resolve()

# -------- Defaults (overridable via env or CLI flags) --------
DEFAULTS = {
    "MLSERVER_HOST": os.getenv("MLSERVER_HOST", "0.0.0.0"),
    "MLSERVER_HTTP_PORT": os.getenv("MLSERVER_HTTP_PORT", "8080"),
    "MLSERVER_GRPC_PORT": os.getenv("MLSERVER_GRPC_PORT", "8081"),
    # MLServer metrics service defaults
    "MLSERVER_METRICS_PORT": os.getenv("MLSERVER_METRICS_PORT", "8082"),
    "MLSERVER_METRICS_ENDPOINT": os.getenv("MLSERVER_METRICS_ENDPOINT", "/metrics"),
    "K6": os.getenv("K6", "./k6"),
    "MLSERVER": os.getenv("MLSERVER", "mlserver"),
    # Optional explicit proto path from env (highest precedence)
    "MLSERVER_PROTO": os.getenv("MLSERVER_PROTO", ""),
}

SCENARIOS = {
    "inference_rest": ROOT / "scenarios" / "inference-rest.js",
    "inference_grpc": ROOT / "scenarios" / "inference-grpc.js",
    "streaming_rest": ROOT / "scenarios" / "streaming-rest.js",
    "streaming_grpc": ROOT / "scenarios" / "streaming-grpc.js",
    "mms": ROOT / "scenarios" / "mms.js",
}


# ------------------------- helpers -------------------------
def _env(overrides: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "MLSERVER_HOST": overrides.get("mlserver_host", DEFAULTS["MLSERVER_HOST"]),
            "MLSERVER_HTTP_PORT": overrides.get("http_port", DEFAULTS["MLSERVER_HTTP_PORT"]),
            "MLSERVER_GRPC_PORT": overrides.get("grpc_port", DEFAULTS["MLSERVER_GRPC_PORT"]),
            "MLSERVER_METRICS_PORT": overrides.get("metrics_port", DEFAULTS["MLSERVER_METRICS_PORT"]),
            "MLSERVER_METRICS_ENDPOINT": overrides.get("metrics_endpoint", DEFAULTS["MLSERVER_METRICS_ENDPOINT"]),
        }
    )
    # If caller provided an explicit proto path via overrides, set it (empty is ignored)
    proto = overrides.get("proto_path")
    if proto:
        env["MLSERVER_PROTO"] = proto
    elif DEFAULTS["MLSERVER_PROTO"]:
        env["MLSERVER_PROTO"] = DEFAULTS["MLSERVER_PROTO"]
    return env


def _run(cmd: list[str], env: dict[str, str] | None = None, check: bool = True) -> int:
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    click.echo(click.style(f"$ {cmd_str}", fg="cyan"))
    proc = subprocess.run(cmd, env=env)
    if check and proc.returncode != 0:
        raise click.ClickException(f"Command failed with exit code {proc.returncode}: {cmd_str}")
    return proc.returncode


def _which(binary: str) -> str | None:
    from shutil import which
    return which(binary)


def _ensure_exists(binary: str, how_to_install: str) -> None:
    if not _which(binary):
        raise click.ClickException(
            f"Required tool '{binary}' not found in PATH.\n{how_to_install}"
        )


def _is_windows() -> bool:
    return platform.system() == "Windows"


def _file_exists(p: Path | None) -> bool:
    return bool(p and p.exists() and p.is_file())


def _find_proto_in_repo() -> Path | None:
    """
    Try several common locations in this repo:
      - benchmarking/proto/dataplane.proto
      - proto/dataplane.proto (one level up)
    """
    candidates = [
        ROOT / "benchmarking" / "proto" / "dataplane.proto",
        ROOT / "proto" / "dataplane.proto",
    ]
    for c in candidates:
        if _file_exists(c):
            return c
    return None


def _find_proto_in_site_packages() -> Path | None:
    """
    Try to locate MLServer's installed dataplane.proto in site-packages.
    Typically under: mlserver/grpc/protos/dataplane.proto
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec("mlserver")
        if not spec or not spec.origin:
            return None
        base = Path(spec.origin).resolve().parent  # .../site-packages/mlserver
        cand = base / "grpc" / "protos" / "dataplane.proto"
        if _file_exists(cand):
            return cand
    except Exception:
        return None
    return None


def _resolve_proto_path(explicit: str | None) -> Path | None:
    """
    Resolution order:
      1) Explicit CLI flag --proto or env MLSERVER_PROTO (absolute or relative)
      2) Repository copy (benchmarking/proto/dataplane.proto, proto/dataplane.proto)
      3) Installed mlserver package (site-packages/mlserver/grpc/protos/dataplane.proto)
      4) None (let k6 error out with a helpful message)
    """
    # 1) explicit
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if _file_exists(p):
            return p
        # If explicit was relative and not found resolved, try relative to repo root
        p2 = (ROOT / explicit).resolve()
        if _file_exists(p2):
            return p2

    env_proto = DEFAULTS["MLSERVER_PROTO"]
    if env_proto:
        p = Path(env_proto).expanduser().resolve()
        if _file_exists(p):
            return p

    # 2) repo
    repo_p = _find_proto_in_repo()
    if repo_p:
        return repo_p.resolve()

    # 3) site-packages
    site_p = _find_proto_in_site_packages()
    if site_p:
        return site_p.resolve()

    # 4) not found
    return None


# ------------------------- CLI root -------------------------
@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--mlserver-host", default=DEFAULTS["MLSERVER_HOST"], show_default=True, help="MLServer host")
@click.option("--http-port", default=DEFAULTS["MLSERVER_HTTP_PORT"], show_default=True, help="MLServer HTTP port")
@click.option("--grpc-port", default=DEFAULTS["MLSERVER_GRPC_PORT"], show_default=True, help="MLServer gRPC port")
@click.option("--metrics-port", default=DEFAULTS["MLSERVER_METRICS_PORT"], show_default=True, help="MLServer metrics port")
@click.option(
    "--metrics-endpoint",
    default=DEFAULTS["MLSERVER_METRICS_ENDPOINT"],
    show_default=True,
    help="MLServer metrics endpoint path",
)
@click.option("--mlserver-bin", default=DEFAULTS["MLSERVER"], show_default=True, help="mlserver binary")
@click.option("--k6-bin", default=DEFAULTS["K6"], show_default=True, help="k6 binary")
@click.option(
    "--proto",
    default=DEFAULTS["MLSERVER_PROTO"],
    show_default=False,
    help="Absolute (or relative to repo root) path to dataplane.proto; overrides auto-detection",
)
@click.pass_context
def cli(ctx, mlserver_host, http_port, grpc_port, metrics_port, metrics_endpoint, mlserver_bin, k6_bin, proto):
    """Repo task runner (Click edition): start server, generate data, and run benchmarks."""
    ctx.ensure_object(dict)
    ctx.obj.update(
        mlserver_host=mlserver_host,
        http_port=http_port,
        grpc_port=grpc_port,
        metrics_port=metrics_port,
        metrics_endpoint=metrics_endpoint,
        mlserver_bin=mlserver_bin,
        k6_bin=k6_bin,
        proto=proto,
    )


# ------------------------- server lifecycle -------------------------
@cli.command("start")
@click.option("--models-dir", default="testserver", show_default=True, help="Path to models dir")
@click.pass_context
def start_cmd(ctx, models_dir):
    """Start MLServer with local test models."""
    mlserver = ctx.obj["mlserver_bin"]
    _ensure_exists(mlserver, "Install mlserver (pip install mlserver) or adjust --mlserver-bin.")
    models_path = (ROOT / models_dir).as_posix()
    _run([mlserver, "start", models_path])


@cli.command("stop")
@click.pass_context
def stop_cmd(ctx):
    """Stop a locally-started MLServer (best-effort)."""
    click.echo(click.style("Stopping MLServer (best-effort)…", fg="yellow"))
    if _is_windows():
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-Process mlserver,python,uvicorn -ErrorAction SilentlyContinue | "
            "Where-Object {$_.Path -match 'mlserver|uvicorn'} | Stop-Process -Force",
        ]
        _run(cmd, check=False)
    else:
        _run(["bash", "-lc", r"pkill -f 'mlserver start .*?/testserver' || true"], check=False)
        _run(["bash", "-lc", r"pkill -f 'uvicorn.*mlserver' || true"], check=False)


@cli.command("restart")
@click.option("--models-dir", default="testserver", show_default=True)
@click.pass_context
def restart_cmd(ctx, models_dir):
    """Restart the local MLServer."""
    ctx.invoke(stop_cmd)
    ctx.invoke(start_cmd, models_dir=models_dir)


# ------------------------- dev setup / data -------------------------
@cli.command("install-dev")
@click.pass_context
def install_dev_cmd(ctx):
    """Check k6 is installed; print install help if missing."""
    k6 = ctx.obj["k6_bin"]
    if not _which(k6):
        click.echo("k6 command not found!\nTo install k6, see:")
        click.echo("  https://k6.io/docs/getting-started/installation/")
    else:
        click.echo(click.style(f"k6 found at '{_which(k6)}'", fg="green"))


@cli.command("generate")
def generate_cmd():
    """Re-generate benchmark payloads into ./data."""
    _run([sys.executable, "generator.py"])


# ------------------------- proto helper command -------------------------
@cli.command("proto-path")
@click.pass_context
def proto_path_cmd(ctx):
    """Print the dataplane.proto path that would be used (or an error)."""
    proto = _resolve_proto_path(ctx.obj.get("proto") or DEFAULTS["MLSERVER_PROTO"])
    if proto:
        click.echo(proto.as_posix())
    else:
        raise click.ClickException(
            "Could not locate dataplane.proto.\n"
            "Provide it with --proto PATH or env MLSERVER_PROTO, or place it under benchmarking/proto/dataplane.proto.\n"
            "Tip: it is often installed at site-packages/mlserver/grpc/protos/dataplane.proto"
        )


# ------------------------- k6 helpers -------------------------
def _k6_env(ctx) -> dict[str, str]:
    # Always include MLSERVER_* base env
    env = _env(ctx.obj)

    return env


def _k6_run(ctx, scenario_path: Path):
    k6 = ctx.obj["k6_bin"]
    _ensure_exists(k6, "Install k6 and ensure it's on PATH, or pass --k6-bin.")

    # Resolve proto and pass to k6 via -e MLSERVER_PROTO=...
    proto_path = _resolve_proto_path(ctx.obj.get("proto") or DEFAULTS["MLSERVER_PROTO"])
    k6_cmd = [
        k6,
        "run",
        "-e", f"MLSERVER_HOST={ctx.obj['mlserver_host']}",
        "-e", f"MLSERVER_HTTP_PORT={ctx.obj['http_port']}",
        "-e", f"MLSERVER_GRPC_PORT={ctx.obj['grpc_port']}",
        "-e", f"MLSERVER_METRICS_PORT={ctx.obj['metrics_port']}",
        "-e", f"MLSERVER_METRICS_ENDPOINT={ctx.obj['metrics_endpoint']}",
    ]
    if proto_path:
        k6_cmd += ["-e", f"MLSERVER_PROTO={proto_path.as_posix()}"]

    k6_cmd.append(scenario_path.as_posix())

    _run(k6_cmd, env=_k6_env(ctx))


# ------------------------- benchmarks (unary) -------------------------
@cli.command("bench-rest")
@click.pass_context
def bench_rest_cmd(ctx):
    """Run REST unary inference benchmark (iris)."""
    _k6_run(ctx, SCENARIOS["inference_rest"])


@cli.command("bench-grpc")
@click.pass_context
def bench_grpc_cmd(ctx):
    """Run gRPC unary inference benchmark (iris)."""
    _k6_run(ctx, SCENARIOS["inference_grpc"])


# ------------------------- benchmarks (streaming) -------------------------
@cli.command("stream-rest")
@click.pass_context
def stream_rest_cmd(ctx):
    """Run REST SSE streaming benchmarks (summodel + llm)."""
    _k6_run(ctx, SCENARIOS["streaming_rest"])


@cli.command("stream-grpc")
@click.pass_context
def stream_grpc_cmd(ctx):
    """Run gRPC streaming benchmarks (summodel + llm)."""
    _k6_run(ctx, SCENARIOS["streaming_grpc"])


@cli.command("stream-all")
@click.pass_context
def stream_all_cmd(ctx):
    """Run both REST and gRPC streaming benchmarks."""
    ctx.invoke(stream_rest_cmd)
    ctx.invoke(stream_grpc_cmd)


# ------------------------- multi-model scenario -------------------------
@cli.command("mms")
@click.pass_context
def mms_cmd(ctx):
    """Run multi-model load/infer scenario."""
    _k6_run(ctx, SCENARIOS["mms"])


if __name__ == "__main__":
    cli()
