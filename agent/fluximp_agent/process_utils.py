#!/usr/bin/env python3
"""
Shared process-management utilities for agent subprocesses.

Provides unified CLI command building, subprocess launching, and graceful
termination.  Lives in the fluximp-agent package because it concerns launching
and managing agent processes rather than the file-IO user-agent protocol.

Agent subprocesses are launched via ``python -m fluximp_agent`` in all cases.
When running from a source checkout (packages not installed) the launcher
injects a ``PYTHONPATH`` covering the source package dirs so the child
process can resolve the ``fluximp_agent`` / ``fluximp_protocol`` /
``fluximp_agent_sdk`` packages.  When the packages are pip-installed they are
already importable, so no ``PYTHONPATH`` injection is needed.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional


def _source_package_root() -> Optional[Path]:
    """Return the source-tree package root (the ``pkg`` dir) when
    ``fluximp_agent`` is being imported from a source checkout, else ``None``.

    In a source checkout ``fluximp_agent/__init__.py`` lives at
    ``<pkg>/agent/fluximp_agent/__init__.py``; three parents up is ``<pkg>``.
    When pip-installed the module lives under site-packages and the sibling
    ``agent`` directory does not exist, so we return ``None``.
    """
    here = Path(__file__).resolve().parent.parent.parent  # <pkg>
    if (here / "agent" / "fluximp_agent" / "__init__.py").exists():
        return here
    return None


def _source_pythonpath() -> Optional[list]:
    """Return the source package dirs to prepend to ``PYTHONPATH`` when
    launching from a source checkout, else ``None``."""
    root = _source_package_root()
    if root is None:
        return None
    return [str(root / d) for d in ("agent", "protocol", "sdk", "wx")]


def build_launch_env(env: Optional[dict] = None) -> dict:
    """Environment for a launched agent subprocess.

    Merges the caller-supplied *env* (or the current process environment) with
    the source ``PYTHONPATH`` when running from a source checkout, so an
    uninstalled subprocess launched via ``python -m fluximp_agent`` can resolve
    the ``fluximp_agent`` / ``fluximp_protocol`` / ``fluximp_agent_sdk``
    packages.  When the packages are pip-installed the caller env (or the
    current process env) is returned unchanged.

    Args:
        env: Optional base environment to extend (defaults to current env).

    Returns:
        The environment dict to pass to the subprocess.
    """
    base = dict(env) if env is not None else os.environ.copy()
    source_pythonpath = _source_pythonpath()
    if source_pythonpath:
        existing = base.get("PYTHONPATH")
        base["PYTHONPATH"] = os.pathsep.join(
            source_pythonpath + ([existing] if existing else [])
        )
    return base


# ---------------------------------------------------------------------------
# CLI command builder
# ---------------------------------------------------------------------------

def build_agent_command(
    *,
    agent_type: str,
    io_dir: Path,
    working_dir: Optional[Path] = None,
    read_root_dir: Optional[Path] = None,
    write_file: Optional[Path] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_requests: Optional[int] = None,
    max_transient_api_error_retries: Optional[int] = None,
    input_timeout: Optional[float] = None,
    permission_timeout: Optional[float] = None,
    plugin_dir: Optional[Path] = None,
    persist_checkpoints: bool = False,
    agent_configs: Optional[list] = None,
    extra_args: Optional[list] = None,
) -> list[str]:
    """Build the CLI command for launching an agent subprocess.

    All parameters except *agent_type* and *io_dir* are optional.

    The agent is launched via ``python -m fluximp_agent``.  When the packages
    are not installed (source checkout), :func:`launch_agent_process` /
    :func:`build_launch_env` inject a ``PYTHONPATH`` covering the source
    package dirs so the subprocess can resolve ``fluximp_agent`` and its
    dependencies.

    Args:
        agent_type: Type of agent (e.g. "developer", "coder", "scrutinator").
        io_dir: Directory for I/O files.
        working_dir: Working directory for file operations.
        read_root_dir: Root directory for read/query operations (must be ancestor of working_dir).
        write_file: Path to the write file.
        provider: Model provider name.
        model: Model name.
        max_requests: Maximum API requests allowed.
        max_transient_api_error_retries: Maximum retries for transient errors.
        input_timeout: Timeout for waiting on input (seconds).
        permission_timeout: Timeout for permission responses (seconds).
        plugin_dir: Path to test-runner plugin directory.
        persist_checkpoints: Enable checkpoint persistence.
        agent_configs: List of paths to JSON agent config files (passed as --agent-config args).
        extra_args: Additional CLI arguments appended verbatim.

    Returns:
        Command as a list of strings.
    """
    cmd = [
        sys.executable, "-u", "-m", "fluximp_agent",
        "--agent-type", agent_type,
        "--io-dir", str(io_dir),
    ]

    if working_dir is not None:
        cmd.extend(["--working-dir", str(working_dir)])
    if read_root_dir is not None:
        cmd.extend(["--read-root-dir", str(read_root_dir)])
    if write_file is not None:
        cmd.extend(["--write-file", str(write_file)])
    if provider is not None:
        cmd.extend(["--provider", provider])
    if model is not None:
        cmd.extend(["--model", model])
    if max_requests is not None:
        cmd.extend(["--max-requests", str(max_requests)])
    if max_transient_api_error_retries is not None:
        cmd.extend(["--max-transient-api-error-retries", str(max_transient_api_error_retries)])
    if input_timeout is not None:
        cmd.extend(["--input-timeout", str(input_timeout)])
    if permission_timeout is not None:
        cmd.extend(["--permission-timeout", str(permission_timeout)])
    if plugin_dir is not None:
        cmd.extend(["--test-runner-plugin-dir", str(plugin_dir)])

    if persist_checkpoints:
        cmd.append("--persist-checkpoints")
    if agent_configs:
        for config_path in agent_configs:
            cmd.extend(["--agent-config", str(config_path)])
    if extra_args:
        cmd.extend(extra_args)

    return cmd


# ---------------------------------------------------------------------------
# Subprocess launch helper
# ---------------------------------------------------------------------------

def launch_agent_process(
    cmd: list[str],
    *,
    stdout=None,
    stderr=None,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
) -> subprocess.Popen:
    """Launch an agent subprocess.

    Wraps ``subprocess.Popen`` with sensible defaults (text mode, UTF-8) and
    injects the source ``PYTHONPATH`` when running from a source checkout.

    Args:
        cmd: Command list (from :func:`build_agent_command`).
        stdout: stdout destination (file, PIPE, STDOUT, etc.).
        stderr: stderr destination.
        cwd: Working directory for the subprocess.
        env: Environment variables (merged with source PYTHONPATH if present).

    Returns:
        The ``subprocess.Popen`` instance.
    """
    return subprocess.Popen(
        cmd,
        stdout=stdout,
        stderr=stderr,
        cwd=cwd,
        env=build_launch_env(env),
        text=True,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Graceful termination
# ---------------------------------------------------------------------------

def terminate_process(
    process: subprocess.Popen,
    *,
    timeout_graceful: float = 5.0,
    timeout_terminate: float = 2.0,
    timeout_kill: float = 2.0,
    pre_terminate_hook: Optional[Callable[[], None]] = None,
    cleanup_hook: Optional[Callable[[], None]] = None,
) -> int | None:
    """Terminate a subprocess with escalation: graceful wait -> SIGTERM -> SIGKILL.

    1. Optionally calls *pre_terminate_hook* (e.g. to create an end-file).
    2. Waits up to *timeout_graceful* seconds for the process to exit on its own.
    3. Sends SIGTERM and waits up to *timeout_terminate* seconds.
    4. Sends SIGKILL and waits up to *timeout_kill* seconds.
    5. Always calls *cleanup_hook* in a ``finally`` block.

    Args:
        process: The subprocess to terminate.
        timeout_graceful: Seconds to wait for graceful exit.
        timeout_terminate: Seconds to wait after SIGTERM.
        timeout_kill: Seconds to wait after SIGKILL.
        pre_terminate_hook: Called before the first wait (e.g. create end-file).
        cleanup_hook: Called in ``finally`` (e.g. close file handles).

    Returns:
        Process return code, or ``None`` if resolution fails.
    """
    try:
        if pre_terminate_hook is not None:
            pre_terminate_hook()
        process.wait(timeout=timeout_graceful)
    except subprocess.TimeoutExpired:
        try:
            process.terminate()
            process.wait(timeout=timeout_terminate)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout_kill)
    finally:
        if cleanup_hook is not None:
            cleanup_hook()
    return process.returncode
