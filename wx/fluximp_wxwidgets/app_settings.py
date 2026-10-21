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

from fluximp_wxwidgets.api_metadata_display import (
    API_METADATA_DEFAULTS,
    API_METADATA_KEYS,
)
from fluximp_wxwidgets.api_chart import (
    CHART_SERIES_DEFAULTS,
    CHART_SERIES_KEYS,
)
from fluximp_wxwidgets.grid_metadata_sums import (
    GRID_SUM_DEFAULTS,
    GRID_SUM_KEYS,
)
from fluximp_wxwidgets import themes

# Setting key: when True (the default) the Provider: and Model: dropdowns
# are limited to the curated set defined in curated_models.py.
LIMIT_TO_CURATED_KEY = "limit_providers_and_models_to_curated"

# Setting key: dict of API Request metadata item key -> bool controlling
# which items render on cards (the View > API Request Card Metadata submenu).
API_METADATA_VISIBILITY_KEY = "api_metadata_visible"

# Setting key: dict of API chart series key -> bool controlling which
# per-request token lines the right panel's chart draws (the View > API
# Chart submenu).
CHART_SERIES_VISIBILITY_KEY = "api_chart_series_visible"

# Setting key: dict of metadata-sum column key -> bool controlling which
# per-turn sum columns the right panel's grid shows (the View > API Grid
# Sums submenu).  Every column is off by default, so the grid is hidden
# until at least one is enabled.
GRID_SUM_VISIBILITY_KEY = "grid_sum_columns_visible"

# Setting key: when True the workflow/conversation history renders every
# API Request card; when False (the default) only the last completed card
# after the last You/Assistant panel and the still-pending card(s) are
# rendered (the View > API Request Card toggle).
API_REQUEST_CARDS_VISIBLE_KEY = "api_request_cards_visible"

# Setting key: active GUI theme (a key from fluximp_wxwidgets.themes.THEMES).
THEME_KEY = "theme"


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

        # Per-item visibility for API Request metadata (the View > API
        # Request Card Metadata submenu).  Seeded from the registry defaults
        # (only Prompt Tokens visible) and persisted in settings.json.
        self._api_metadata_visible: dict = dict(API_METADATA_DEFAULTS)

        # Per-series visibility for the right panel's API chart (View > API
        # Chart submenu).  Seeded from the registry defaults (only Prompt
        # visible) and persisted in settings.json.
        self._chart_series_visible: dict = dict(CHART_SERIES_DEFAULTS)

        # Per-column visibility for the right panel's metadata-sum grid
        # columns (View > API Grid Sums submenu).  Seeded from the
        # registry defaults (every column off, so the grid starts hidden) and
        # persisted in settings.json.
        self._grid_sum_visible: dict = dict(GRID_SUM_DEFAULTS)

        # Whether every API Request card is rendered in the workflow history
        # (View > API Request Card toggle).  Off by default: only the last
        # completed card after the last You/Assistant panel and the pending
        # card(s) are shown, keeping the history compact.
        self._api_request_cards_visible: bool = False

        # Active GUI theme key (fluximp_wxwidgets.themes.THEMES); read at
        # startup by main.py before the frame is built.
        self._theme: str = themes.DEFAULT_THEME_KEY

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

        stored = data.get(API_METADATA_VISIBILITY_KEY)
        if isinstance(stored, dict):
            # Merge per-item booleans over the registry defaults; ignore
            # unknown keys and non-bool values so a hand-edited file
            # cannot crash startup.
            for key in API_METADATA_KEYS:
                item = stored.get(key)
                if isinstance(item, bool):
                    self._api_metadata_visible[key] = item

        chart_stored = data.get(CHART_SERIES_VISIBILITY_KEY)
        if isinstance(chart_stored, dict):
            # Same merge-over-defaults treatment as the metadata toggles: a
            # hand-edited file cannot crash startup.
            for key in CHART_SERIES_KEYS:
                item = chart_stored.get(key)
                if isinstance(item, bool):
                    self._chart_series_visible[key] = item

        grid_stored = data.get(GRID_SUM_VISIBILITY_KEY)
        if isinstance(grid_stored, dict):
            # Same merge-over-defaults treatment as the other toggles: a
            # hand-edited file cannot crash startup.
            for key in GRID_SUM_KEYS:
                item = grid_stored.get(key)
                if isinstance(item, bool):
                    self._grid_sum_visible[key] = item

        cards_visible = data.get(API_REQUEST_CARDS_VISIBLE_KEY)
        if isinstance(cards_visible, bool):
            self._api_request_cards_visible = cards_visible

        theme = data.get(THEME_KEY)
        if isinstance(theme, str) and theme.strip().lower() in themes.THEMES:
            self._theme = theme.strip().lower()

    def _save(self) -> None:
        """Persist settings to disk (best-effort)."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        data = {
            LIMIT_TO_CURATED_KEY: self._limit_providers_and_models_to_curated,
            API_METADATA_VISIBILITY_KEY: self._api_metadata_visible,
            CHART_SERIES_VISIBILITY_KEY: self._chart_series_visible,
            GRID_SUM_VISIBILITY_KEY: self._grid_sum_visible,
            API_REQUEST_CARDS_VISIBLE_KEY: self._api_request_cards_visible,
            THEME_KEY: self._theme,
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

    @property
    def api_request_cards_visible(self) -> bool:
        """True when every API Request card is rendered in the history.

        Backs the View > API Request Card toggle; False (the default) keeps
        the history compact, showing only the last completed card after the
        last You/Assistant panel and the pending card(s).
        """
        return self._api_request_cards_visible

    @api_request_cards_visible.setter
    def api_request_cards_visible(self, value: bool) -> None:
        value = bool(value)
        if value == self._api_request_cards_visible:
            return
        self._api_request_cards_visible = value
        self._save()

    @property
    def theme(self) -> str:
        """Active theme key (a key from fluximp_wxwidgets.themes.THEMES)."""
        return self._theme

    @theme.setter
    def theme(self, value: str) -> None:
        # Unknown keys fall back to the default, mirroring themes.set_active,
        # so the persisted value always matches what gets painted.
        resolved = (
            value.strip().lower()
            if isinstance(value, str) and value.strip().lower() in themes.THEMES
            else themes.DEFAULT_THEME_KEY)
        if resolved == self._theme:
            return
        self._theme = resolved
        self._save()

    def is_api_metadata_visible(self, key: str) -> bool:
        """Whether the given API metadata item is shown on cards.

        Missing keys fall back to the item's registry default so a
        settings.json written before an item existed still resolves.
        """
        default = API_METADATA_DEFAULTS.get(key, True)
        return self._api_metadata_visible.get(key, default)

    def set_api_metadata_visible(self, key: str, value: bool) -> None:
        """Persist the visibility of one API metadata item."""
        if key not in API_METADATA_DEFAULTS:
            return
        value = bool(value)
        if self._api_metadata_visible.get(key) == value:
            return
        self._api_metadata_visible[key] = value
        self._save()

    def set_all_api_metadata_visible(self, value: bool) -> None:
        """Set every API metadata item's visibility at once (single save).

        Backs the View > API Request Card Metadata submenu's Enable All /
        Disable All bulk items so sweeping the whole set persists in one
        write.
        """
        value = bool(value)
        if all(self._api_metadata_visible.get(key) == value
               for key in API_METADATA_KEYS):
            return
        for key in API_METADATA_KEYS:
            self._api_metadata_visible[key] = value
        self._save()

    def is_chart_series_visible(self, key: str) -> bool:
        """Whether the given per-request token series is drawn on the chart.

        Missing keys fall back to the series' registry default (only Prompt
        is on) so a settings.json written before a series existed still
        resolves.
        """
        default = CHART_SERIES_DEFAULTS.get(key, False)
        return self._chart_series_visible.get(key, default)

    def set_chart_series_visible(self, key: str, value: bool) -> None:
        """Persist the visibility of one API chart series."""
        if key not in CHART_SERIES_DEFAULTS:
            return
        value = bool(value)
        if self._chart_series_visible.get(key) == value:
            return
        self._chart_series_visible[key] = value
        self._save()

    def is_grid_sum_visible(self, key: str) -> bool:
        """Whether the given metadata-sum column is shown on the grid.

        Missing keys fall back to the column's registry default (every
        column is off) so a settings.json written before a column existed
        still resolves.
        """
        default = GRID_SUM_DEFAULTS.get(key, False)
        return self._grid_sum_visible.get(key, default)

    def set_grid_sum_visible(self, key: str, value: bool) -> None:
        """Persist the visibility of one metadata-sum grid column."""
        if key not in GRID_SUM_DEFAULTS:
            return
        value = bool(value)
        if self._grid_sum_visible.get(key) == value:
            return
        self._grid_sum_visible[key] = value
        self._save()
