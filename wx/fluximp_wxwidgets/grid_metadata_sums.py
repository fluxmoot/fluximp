"""Registry of toggleable metadata-sum columns in the right panel's grid.

Single source of truth shared by the View > API Grid Sums menu and the
right panel's fork/turn grid (right_panel.ForkTurnGrid), mirroring
api_chart.py and api_metadata_display.py.  Deliberately free of wx imports so
AppSettings (constructed before the GUI toolkit is fully initialized) can
import the column keys and defaults without pulling in wx.

Each entry is ``(key, menu label, grid column index, default_visible)``.  The
key names the per-turn sum on conversation_manager.ForkTurnInfo, so the grid
needs no per-column mapping code.  Every column is off by default: the grid
is a wall of numbers to most users and the API chart (View > API Chart) is
the primary overall metrics source, so the grid is opt-in and hides itself
when every column is switched off (ForkTurnPanel).
"""

from typing import Dict, List, Tuple

# (key, menu label, grid column index, visible by default)
#
# The column index must match the grid's COL_* constants; ForkTurnGrid
# derives those from this registry (see right_panel) so the menu and the grid
# cannot drift apart.  Fork and Turn are not listed: they are always shown.
GRID_SUM_COLUMNS: List[Tuple[str, str, int, bool]] = [
    ("prompt_tokens", "Prompt", 2, False),
    ("cached_tokens", "Cached", 3, False),
    ("completion_tokens", "Compl", 4, False),
    ("reasoning_tokens", "Reason", 5, False),
    ("time", "Time (s)", 6, False),
]

GRID_SUM_KEYS: List[str] = [column[0] for column in GRID_SUM_COLUMNS]

GRID_SUM_DEFAULTS: Dict[str, bool] = {
    column[0]: column[3] for column in GRID_SUM_COLUMNS}

GRID_SUM_LABELS: Dict[str, str] = {
    column[0]: column[1] for column in GRID_SUM_COLUMNS}

GRID_SUM_COLUMN_INDEX: Dict[str, int] = {
    column[0]: column[2] for column in GRID_SUM_COLUMNS}
