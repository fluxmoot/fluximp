"""Application-level (global) settings persisted in base_dir/settings.json.

These settings apply to the whole application rather than to a single
workspace (see WorkspaceSettingsTab for per-workspace settings).  The file
is deliberately separate from state.json (GlobalState's deferred-deletion
state) and window_state.json (window geometry) so each concern stays
independent.
"""

import json
from pathlib import Path
from typing import Optional

# Setting key: when True (the default) the Provider: and Model: dropdowns
# are limited to the curated set defined in curated_models.py.
LIMIT_TO_CURATED_KEY = "limit_providers_and_models_to_curated"


class AppSettings:
    """Application-level settings backed by base_dir/settings.json.

    A missing or unreadable file falls back to the defaults so a corrupt
    file can never prevent startup.  Writes are best-effort (IOError is
    swallowed), matching GlobalState and MainFrame._save_window_state.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            base_dir = Path.home() / ".fluximp"
        self.base_dir = base_dir
        self.settings_file = base_dir / "settings.json"

        # Default: limit providers and models to the curated set.
        self._limit_providers_and_models_to_curated: bool = True

        self._load()

    def _load(self) -> None:
        """Load settings from disk, keeping defaults on any failure."""
        if not self.settings_file.exists():
            return
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return
        if not isinstance(data, dict):
            # Valid JSON that isn't an object (list/string/number/null) can't
            # hold the setting; keep the defaults rather than raising.
            return
        value = data.get(LIMIT_TO_CURATED_KEY)
        if isinstance(value, bool):
            self._limit_providers_and_models_to_curated = value

    def _save(self) -> None:
        """Persist settings to disk (best-effort)."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        data = {
            LIMIT_TO_CURATED_KEY: self._limit_providers_and_models_to_curated,
        }
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except IOError:
            pass  # Silently fail if we can't write the file

    @property
    def limit_providers_and_models_to_curated(self) -> bool:
        """True when the Provider/Model dropdowns are limited to the curated set."""
        return self._limit_providers_and_models_to_curated

    @limit_providers_and_models_to_curated.setter
    def limit_providers_and_models_to_curated(self, value: bool) -> None:
        value = bool(value)
        if value == self._limit_providers_and_models_to_curated:
            return
        self._limit_providers_and_models_to_curated = value
        self._save()
