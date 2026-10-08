"""Checkpoint marker files for mid-turn recovery.

Writes and reads {fork}.{turn}.checkpoint.txt files in the .checkpoints
directory.  These files store the LangGraph checkpoint ID associated with
a completed turn so that mid-turn recovery can detect and resume from a
crash.

Marker files are only written when using the SQLite saver
(persist_checkpoints=True) because mid-turn recovery is only possible
when checkpoints are persisted to disk.
"""

from pathlib import Path
from typing import Optional, Tuple


def write_checkpoint_marker(
    checkpoints_dir: Path, fork: int, turn: int, checkpoint_id: str
) -> None:
    """Write a checkpoint marker file.

    Args:
        checkpoints_dir: Path to the .checkpoints directory
        fork: Fork number
        turn: Turn number
        checkpoint_id: The LangGraph checkpoint ID
    """
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    marker_path = checkpoints_dir / f"{fork}.{turn}.checkpoint.txt"
    marker_path.write_text(checkpoint_id, encoding="utf-8")


def recover_checkpoint_markers(
    checkpoints_dir: Path,
) -> Optional[Tuple[int, int, str]]:
    """Recover the latest checkpoint marker from the .checkpoints directory.

    Scans all {fork}.{turn}.checkpoint.txt files and returns the marker
    with the highest (fork, turn) ordering.

    Args:
        checkpoints_dir: Path to the .checkpoints directory

    Returns:
        Tuple of (fork, turn, checkpoint_id) for the latest marker,
        or None if no markers found or directory does not exist.
    """
    if not checkpoints_dir.exists():
        return None

    marker_files = list(checkpoints_dir.glob("*.checkpoint.txt"))
    if not marker_files:
        return None

    latest: Optional[Tuple[int, int, str]] = None
    latest_key: Tuple[int, int] = (-1, -1)

    for marker_path in marker_files:
        # Parse fork.turn from filename like "0.2.checkpoint.txt"
        stem = marker_path.name  # e.g. "0.2.checkpoint.txt"
        parts = stem.rsplit(".checkpoint.txt", 1)[0]  # "0.2"
        try:
            fork_str, turn_str = parts.split(".", 1)
            fork = int(fork_str)
            turn = int(turn_str)
        except (ValueError, IndexError):
            print(f"  Warning: skipping malformed checkpoint marker file: {marker_path.name}")
            continue

        key = (fork, turn)
        if key > latest_key:
            checkpoint_id = marker_path.read_text(encoding="utf-8").strip()
            latest = (fork, turn, checkpoint_id)
            latest_key = key

    return latest
