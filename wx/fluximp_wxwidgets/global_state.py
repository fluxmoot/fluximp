"""Global state management for deferred deletions and other app-wide state."""

import json
import threading
from pathlib import Path
from typing import Optional, Set


class GlobalState:
    """Manages global application state stored in base_dir/state.json.

    This includes the deferred_deletions set which tracks conversation
    directories that couldn't be deleted (e.g., due to file locks) and
    need to be retried on next startup.
    """

    _instance: Optional["GlobalState"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, base_dir: Optional[Path] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, base_dir: Optional[Path] = None):
        if self._initialized:
            return

        if base_dir is None:
            base_dir = Path.home() / ".fluximp"
        self.base_dir = base_dir
        self.state_file = base_dir / "state.json"

        # In-memory cache of deferred deletions (absolute paths as strings)
        self._deferred_deletions: Set[str] = set()

        # Thread safety for state modifications
        self._file_lock = threading.Lock()

        # Load existing state
        self._load()

        self._initialized = True

    def _load(self):
        """Load state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._deferred_deletions = set(data.get('deferred_deletions', []))
            except (json.JSONDecodeError, IOError):
                self._deferred_deletions = set()
        else:
            self._deferred_deletions = set()

    def _save(self):
        """Save state to disk."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        data = {
            'deferred_deletions': list(self._deferred_deletions)
        }
        try:
            with self._file_lock:
                with open(self.state_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
        except IOError:
            pass  # Silently fail if we can't write

    def add_deferred_deletion(self, path: Path) -> None:
        """Add a path to the deferred deletion set.

        Args:
            path: The directory path to delete later
        """
        abs_path = str(path.absolute())
        self._deferred_deletions.add(abs_path)
        self._save()

    def remove_deferred_deletion(self, path: Path) -> None:
        """Remove a path from the deferred deletion set.

        Args:
            path: The directory path to remove from deferred set
        """
        abs_path = str(path.absolute())
        if abs_path in self._deferred_deletions:
            self._deferred_deletions.discard(abs_path)
            self._save()

    def get_deferred_deletions(self) -> Set[str]:
        """Get a copy of the deferred deletion set.

        Returns:
            Set of absolute paths (as strings) marked for deletion
        """
        return self._deferred_deletions.copy()

    def is_deferred(self, path: Path) -> bool:
        """Check if a path is in the deferred deletion set.

        Args:
            path: The directory path to check

        Returns:
            True if the path is marked for deferred deletion
        """
        return str(path.absolute()) in self._deferred_deletions

    def process_deferred_deletions(self) -> None:
        """Attempt to delete all deferred directories.

        Removes successfully deleted paths from the set.
        Failed deletions remain for next startup attempt.
        """
        import shutil

        deleted = []
        for path_str in self._deferred_deletions:
            path = Path(path_str)
            if not path.exists():
                # Already gone, mark for removal
                deleted.append(path_str)
                continue

            try:
                shutil.rmtree(path)
                # Success
                deleted.append(path_str)
            except (PermissionError, OSError):
                # Still locked, will retry next startup
                pass

        # Remove successfully deleted paths
        if deleted:
            for path_str in deleted:
                self._deferred_deletions.discard(path_str)
            self._save()


# Convenience function to get the singleton instance
def get_global_state(base_dir: Optional[Path] = None) -> GlobalState:
    """Get the GlobalState singleton instance."""
    return GlobalState(base_dir)
