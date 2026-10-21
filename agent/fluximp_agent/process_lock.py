#!/usr/bin/env python3
"""Scope-directory liveness locks for FluxImp processes.

Exactly one FluxImp process of a given kind may own a scope directory at a
time, and "does anything own it right now?" is answerable exactly, without
PID bookkeeping:

Held-ness — not file presence — is the signal. The lock file (POSIX) or
named kernel mutex (Windows) is created once and NEVER deleted by anyone.
The OS keeps the truth:

- A lock held by a process is released automatically when the process dies
  (clean exit, crash, OOM kill, TerminateProcess/SIGKILL — anything), so a
  probe's can-I-take-it verdict is an exact is-the-holder-alive oracle that
  PID reuse cannot fool (the lock never consults a PID).
- probe() never reports 'free' on an open/lookup failure: transient
  filesystem/OS errors (AV scanners, indexers, permissions) return
  'unknown' so callers degrade to their previous behavior instead of
  misreporting a live process as dead.

Platform notes:

- POSIX uses flock (an open-file-description lock) on a content-free lock
  file inside the scope dir — created once with the scope, never deleted
  (a stale file is inert: presence never implies liveness). flock, not
  fcntl record locks (lockf/F_SETLK): record locks are per-process and are
  silently dropped when *any* fd to the file is closed anywhere in the
  process. The held fd stays open for the owner's process lifetime (Python
  opens fds non-inheritable by default, so spawned children cannot inherit
  and prematurely release it).
- Windows uses a named kernel mutex (Local\\fluximp-<kind>-<hash>) instead
  of a byte-range file lock (msvcrt.locking/LockFile): mandatory file
  locks make the locked range unreadable to every other process, and
  AV/indexer/backup tools routinely break opens on locked files. The mutex
  is also released on process death, and WaitForSingleObject's
  WAIT_ABANDONED verdict additionally distinguishes "previous holder
  terminated without releasing" (crash) from a clean release — recorded in
  the module's last-probe/acquire detail for future clean-vs-crash
  classification. That classification is observable only on a wait that
  was pending across the death (the waiter's own handle keeps the object
  alive through the collapse): probes open and close their own handles,
  so a probe consulted after a holder died finds the object either
  cleanly released (nothing to distinguish) or destroyed when the last
  handle closed (ERROR_FILE_NOT_FOUND — indistinguishable from
  never-created). The deferred clean-vs-crash work must therefore read
  the acquire path's recorded abandoned=True (DirLock.acquire taking over
  a dead holder), never after-the-death probe verdicts.

Kinds: 'agent' (io/run-directory single-owner for agent processes, primary
and conditional alike — each has its own io directory) and 'gui'
(base-directory single-instance for the GUI process).
"""

import errno
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Optional

_VALID_KINDS = ('agent', 'gui')

# POSIX lock-file names per kind. Content-free, created once, never deleted.
_POSIX_LOCK_NAMES = {
    'agent': 'agent.lock',
    'gui': '.fluximp-gui.lock',
}


# -----------------------------------------------------------------------------
# Scope identity + probe detail (clean-vs-crash forensics for future work)
# -----------------------------------------------------------------------------

def _validate_kind(kind: str) -> None:
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"unknown lock kind {kind!r} (expected one of {_VALID_KINDS})")


def _scope_key(scope_dir) -> str:
    """Stable cross-platform identity string of a scope directory.

    Resolved + case-normalized so two spellings of the same directory hash
    to the same id (both GUIs/agents must agree on the single "owner of
    this scope" question).
    """
    path = Path(scope_dir)
    try:
        norm = os.path.normcase(str(path.resolve()))
    except OSError:
        norm = os.path.normcase(str(path.absolute()))
    return hashlib.sha256(norm.encode('utf-8', 'surrogatepass')).hexdigest()[:16]


# (kind, scope_key) -> {'abandoned': bool, 'ts': float}: the detail of
# the most recent probe OR acquire for a scope. Windows WAIT_ABANDONED
# (the prior holder terminated without releasing) records abandoned=True;
# retained so future clean-vs-crash classification can consult it without
# re-architecting the platform layers. Windows caveat: that classification
# survives a holder's death only on a wait another process had pending
# across the death (that waiter's handle keeps the object alive) — the
# acquire path is where a crash is reliably recorded; probes consulted
# after the death usually find the object destroyed (ERROR_FILE_NOT_FOUND
# reads exactly like never-created).
_probe_details = {}


def _record_detail(scope_key: str, kind: str, abandoned: bool) -> None:
    try:
        _probe_details[(kind, scope_key)] = {
            'abandoned': abandoned, 'ts': time.time()}
    except Exception:
        pass


def probe_detail(scope_dir, kind: str) -> dict:
    """Return the most recent probe's detail dict for a scope (possibly {})."""
    _validate_kind(kind)
    return dict(_probe_details.get((kind, _scope_key(scope_dir)), {}))


# -----------------------------------------------------------------------------
# Platform layer
# -----------------------------------------------------------------------------

if sys.platform == 'win32':
    import ctypes
    from ctypes import wintypes

    _WAIT_OBJECT_0 = 0x00000000
    _WAIT_TIMEOUT = 0x00000102
    _WAIT_ABANDONED = 0x00000080
    _WAIT_FAILED = 0xFFFFFFFF
    _ERROR_FILE_NOT_FOUND = 2
    _SYNCHRONIZE = 0x00100000

    _kernel32 = None

    def _kernel32_dll():
        """Lazy-load kernel32 with typed signatures (single load per proc)."""
        global _kernel32
        if _kernel32 is None:
            dll = ctypes.WinDLL('kernel32', use_last_error=True)
            dll.CreateMutexW.restype = wintypes.HANDLE
            dll.CreateMutexW.argtypes = (
                wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
            dll.OpenMutexW.restype = wintypes.HANDLE
            dll.OpenMutexW.argtypes = (
                wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
            dll.WaitForSingleObject.restype = wintypes.DWORD
            dll.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            dll.ReleaseMutex.restype = wintypes.BOOL
            dll.ReleaseMutex.argtypes = (wintypes.HANDLE,)
            dll.CloseHandle.restype = wintypes.BOOL
            dll.CloseHandle.argtypes = (wintypes.HANDLE,)
            _kernel32 = dll
        return _kernel32

    def _mutex_name(scope_key: str, kind: str) -> str:
        # Local\ (session-local): no cross-session scenario exists, and
        # Global\ requires privileges. The hash keeps the name short and
        # free of path characters regardless of the scope path's length.
        return f"Local\\fluximp-{kind}-{scope_key}"

    def _probe_impl(scope_dir, scope_key: str, kind: str) -> str:
        k32 = _kernel32_dll()
        handle = k32.OpenMutexW(_SYNCHRONIZE, False, _mutex_name(scope_key, kind))
        if not handle:
            err = ctypes.get_last_error()
            if err == _ERROR_FILE_NOT_FOUND:
                # No process ever created the mutex for this scope: nothing
                # holds it.
                _record_detail(scope_key, kind, abandoned=False)
                return 'free'
            # Access denied or another lookup failure: unknown, never a
            # false-free.
            return 'unknown'
        rc = k32.WaitForSingleObject(handle, 0)
        if rc == _WAIT_TIMEOUT:
            k32.CloseHandle(handle)
            return 'held'
        if rc == _WAIT_ABANDONED:
            abandoned = True
        elif rc == _WAIT_OBJECT_0:
            abandoned = False
        else:
            # WAIT_FAILED: unknown, never a false verdict.
            k32.CloseHandle(handle)
            return 'unknown'
        # We acquired the mutex (freshly or via abandonment transfer):
        # release it immediately — a probe takes no ownership — and report
        # the scope free, with the abandonment classification recorded.
        k32.ReleaseMutex(handle)
        k32.CloseHandle(handle)
        _record_detail(scope_key, kind, abandoned)
        return 'free'

    def _acquire_impl(scope_dir, scope_key: str, kind: str,
                      wait_timeout: float, poll_interval: float):
        k32 = _kernel32_dll()
        name = _mutex_name(scope_key, kind)
        deadline = time.monotonic() + max(0.0, wait_timeout)
        while True:
            # CreateMutexW returns the existing mutex when the name is
            # taken (ownership is still decided by the wait below).
            handle = k32.CreateMutexW(None, False, name)
            if not handle:
                return None
            remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
            rc = k32.WaitForSingleObject(handle, remaining_ms)
            if rc == _WAIT_OBJECT_0:
                _record_detail(scope_key, kind, abandoned=False)
                return handle
            if rc == _WAIT_ABANDONED:
                # The prior holder terminated without releasing (crash):
                # ownership transferred to this wait — acquired.
                _record_detail(scope_key, kind, abandoned=True)
                return handle
            k32.CloseHandle(handle)
            if rc != _WAIT_TIMEOUT:
                # WAIT_FAILED: unretryable.
                return None
            if time.monotonic() >= deadline:
                return None
            time.sleep(min(poll_interval,
                           max(0.0, deadline - time.monotonic())))

    def _release_impl(handle) -> None:
        if handle is None:
            return
        k32 = _kernel32_dll()
        try:
            k32.ReleaseMutex(handle)
        except Exception:
            pass
        try:
            k32.CloseHandle(handle)
        except Exception:
            pass

else:
    import fcntl

    def _lock_file_path(scope_dir, kind: str) -> Path:
        return Path(scope_dir) / _POSIX_LOCK_NAMES[kind]

    def _try_flock(scope_dir, kind: str):
        """Open the (created-once) lock file and attempt a non-blocking
        exclusive flock.

        Returns (verdict, fd): verdict is 'acquired' (fd held open, the
        caller owns the lock), 'held' (a live process owns it; fd closed),
        or 'unknown' (open/flock failure that is not a held signal; never a
        false-free).
        """
        path = _lock_file_path(scope_dir, kind)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError:
            return 'unknown', None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            try:
                os.close(fd)
            except OSError:
                pass
            if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES,
                           errno.EBUSY):
                # Held by a live process.
                return 'held', None
            return 'unknown', None
        return 'acquired', fd

    def _probe_impl(scope_dir, scope_key: str, kind: str) -> str:
        verdict, fd = _try_flock(scope_dir, kind)
        if verdict != 'acquired':
            return verdict
        # We immediately held it: nothing owns the scope right now. Undo
        # the probe's acquisition so it takes no ownership (posix locks are
        # advisory per open file description; unlock + close).
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass
        _record_detail(scope_key, kind, abandoned=False)
        return 'free'

    def _acquire_impl(scope_dir, scope_key: str, kind: str,
                      wait_timeout: float, poll_interval: float):
        deadline = time.monotonic() + max(0.0, wait_timeout)
        while True:
            verdict, fd = _try_flock(scope_dir, kind)
            if verdict == 'acquired':
                _record_detail(scope_key, kind, abandoned=False)
                return fd
            if verdict != 'held':
                # 'unknown' (open/lock failure) is not retryable.
                return None
            if time.monotonic() >= deadline:
                return None
            time.sleep(min(poll_interval,
                           max(0.0, deadline - time.monotonic())))

    def _release_impl(fd) -> None:
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def probe(scope_dir, kind: str) -> str:
    """Probe held-ness of a scope directory's lock (no ownership taken).

    Args:
        scope_dir: The scope directory (io/run dir for 'agent'; the
            base dir for 'gui').
        kind: Lock kind, 'agent' or 'gui'.

    Returns:
        'held'   — a live process owns the scope (the recorded holder is
                   alive: kernel-held state cannot be fooled by PID reuse),
        'free'   — no process owns the scope, or a prior holder crashed
                   (Windows abandonment detail available via probe_detail),
        'unknown'— the probe itself failed; never treat as alive or dead.
    """
    _validate_kind(kind)
    scope_key = _scope_key(scope_dir)
    try:
        verdict = _probe_impl(scope_dir, scope_key, kind)
    except Exception:
        return 'unknown'
    return verdict if verdict in ('held', 'free', 'unknown') else 'unknown'


class DirLock:
    """An acquired scope-directory lock.

    The owning process holds it for its process lifetime; the OS releases
    it when the process dies, so even a crash frees the scope instantly
    and no stale file can ever block a restart. release() is hygiene:
    explicit, idempotent, and never required for correctness — but it lets
    a same-process caller observe the flip promptly.
    """

    def __init__(self, handle, scope_dir, kind: str, scope_key: str):
        self._handle = handle
        self._released = False
        self.scope_dir = scope_dir
        self.kind = kind
        self.scope_key = scope_key

    @classmethod
    def acquire(cls, scope_dir, kind: str, wait_timeout: float = 15.0,
                poll_interval: float = 0.25) -> Optional['DirLock']:
        """Acquire the scope lock, bounded by *wait_timeout* seconds.

        The bounded retry absorbs a legitimate ownership handoff window —
        e.g. a terminate-then-restart whose prior holder is still draining
        its graceful/SIGTERM/SIGKILL ladder — so ordinary restart sequences
        self-synchronize instead of racing. On timeout or unretryable
        failure None is returned and the caller must not run.

        Args:
            scope_dir: The scope directory to own.
            kind: Lock kind, 'agent' or 'gui'.
            wait_timeout: Max seconds to wait for ownership.
            poll_interval: Sleep between bounded retries.

        Returns:
            The held DirLock, or None.

        Raises:
            ValueError: For an unknown lock kind.
        """
        _validate_kind(kind)
        scope_key = _scope_key(scope_dir)
        try:
            handle = _acquire_impl(scope_dir, scope_key, kind,
                                   wait_timeout, poll_interval)
        except Exception:
            return None
        if handle is None:
            return None
        return cls(handle, scope_dir, kind, scope_key)

    def release(self) -> None:
        """Release the lock (idempotent; never required for correctness)."""
        if self._released:
            return
        self._released = True
        handle, self._handle = self._handle, None
        try:
            _release_impl(handle)
        except Exception:
            pass
