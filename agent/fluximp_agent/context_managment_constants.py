#!/usr/bin/env python3
"""Single source of truth for context-management constants.

Values here govern how much content the harness injects into (and retains in)
the model context. Nothing here is a model token budget or the model's
context-window size.
"""

# ---------------------------------------------------------------------------
# View content byte budget: total size of injected view content
# (file view windows + search/list views).
#
# Tuning history (kept with the value so it moves with it):
#   360000 / 280000 / 240000 / 200000 / 120000  - explored
#   163840 (160KB)  - previous value
#   163840/2 (80KB) - too small; both GLM5.1 and Kimi K2.7 Code failed to start
#                     editing poc/sdl_vulkan_cuda_lvgl to continue phase 3 of a
#                     plan to refactor from C-style to C++20 with RAII.
# ---------------------------------------------------------------------------
MAX_FILE_VIEW_BYTES = 320000

# Fraction of MAX_FILE_VIEW_BYTES at which the view-capacity percentage is
# surfaced to the model (see FileSystemAgent.get_context_chunks).
VIEW_CAPACITY_WARN_FRACTION = 0.80


def _format_bytes(num_bytes: int) -> str:
    """Render a byte count for display, e.g. 320000 -> '320,000 bytes (312.5 KB)'."""
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes:,} bytes ({num_bytes / (1024 * 1024):.1f} MB)"
    if num_bytes >= 1024:
        return f"{num_bytes:,} bytes ({num_bytes / 1024:.1f} KB)"
    return f"{num_bytes:,} bytes"


# Derived, pre-rendered description of the budget so injected strings can
# never drift from the real value.
VIEW_BUDGET_DESC = _format_bytes(MAX_FILE_VIEW_BYTES)

# Shared truncation marker for state_service view rendering (both call sites).
VIEW_TRUNCATION_MARKER = (
    f"\n... (view truncated: exceeds view content budget of {VIEW_BUDGET_DESC})"
)
