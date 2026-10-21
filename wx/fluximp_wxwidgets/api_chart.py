"""API Chart: the per-request token series drawn in the right panel.

Single source of truth shared by the View > API Chart menu and the chart
itself (right_panel.ForkTokenChart), mirroring api_metadata_display.py.

Each series is one line on the chart and one toggleable item in the
View > API Chart submenu.  The chart plots one point per API request
(conversation_manager.RequestTokenInfo) of the displayed conversation, in
request order, so the X axis is the implicit request index.

This module also owns the policy both axes are ticked by -- see the tick
helpers at the bottom: the Y (token) axis in fixed 10k steps that widen as
the counts climb, and the X (request) axis at a few request numbers rather
than one per point.
"""

import math
from typing import Dict, List, Tuple

# (key, menu label, theme role for the line colour, visible by default)
#
# The key names the token count on conversation_manager.RequestTokenInfo (and,
# for the grid's per-turn sums, on ForkTurnInfo), so the chart needs no
# per-series mapping code.  Prompt is the only series enabled out of the box;
# the other three are opt-in.
CHART_SERIES: List[Tuple[str, str, str, bool]] = [
    ("prompt_tokens", "Prompt", "chart_line_prompt", True),
    ("cached_tokens", "Cached", "chart_line_cached", False),
    ("completion_tokens", "Completion", "chart_line_completion", False),
    ("reasoning_tokens", "Reasoning", "chart_line_reasoning", False),
]

CHART_SERIES_KEYS: List[str] = [series[0] for series in CHART_SERIES]

CHART_SERIES_DEFAULTS: Dict[str, bool] = {
    series[0]: series[3] for series in CHART_SERIES}

CHART_SERIES_LABELS: Dict[str, str] = {
    series[0]: series[1] for series in CHART_SERIES}

CHART_SERIES_LINE_ROLES: Dict[str, str] = {
    series[0]: series[2] for series in CHART_SERIES}

# Axis labels: the X axis is the request index (no per-point labels), the Y
# axis is a token count.
CHART_X_LABEL = "Request"
CHART_Y_LABEL = "Tokens"

# The Y axis starts at the smallest plotted value and its top never drops
# below this floor, rising with the highest value so no point is ever off the
# chart.
CHART_Y_MAX_FLOOR = 65536


# --------------------------------------------------------------------------
# Axis tick policy
# --------------------------------------------------------------------------
# wx.lib.plot derives its ticks from an axis range plus a "spec" (Draw takes
# no tick list), and its automatic rounding gives a 64k token axis 5k steps --
# fourteen labels and grid lines.  So both axes' ticks are generated here
# instead and fed back in by right_panel.TokenPlotCanvas, which overrides the
# plot's two tick generators; keeping the policy in this wx-free module is
# also what lets the tests pin it without a GUI.

# -- Y (tokens) axis --
#
# A fixed 10k step, widened by another 10k (10k, 20k, 30k, ...) while the
# range would need this many labels or more: a fixed step cannot grow with
# the data the way the automatic one does, so a long run would otherwise
# crowd the axis again.
CHART_Y_TICK_STEP = 10000
CHART_Y_TICK_MAX_LABELS = 10

# -- X (request) axis --
#
# A tick's position is its label (index + 1 = the request number), so only a
# few requests are ticked: always the last one, plus the ones nearest these
# fractions of the way through once there are enough requests for that to be
# readable, or just the one nearest halfway once there are enough for two.
CHART_X_TICK_FRACTIONS: Tuple[float, ...] = (0.25, 0.5, 0.75)
CHART_X_TICK_SPREAD_MIN_POINTS = 10
CHART_X_TICK_PAIR_MIN_POINTS = 6


def _tick_range(lower, upper) -> Tuple[int, int]:
    """Coerce an axis range to the ints the token axis is counted in."""
    lower = int(math.ceil(lower))
    upper = int(math.floor(upper))
    return lower, max(lower, upper)


def _count_ticks(lower: int, upper: int, step: int) -> int:
    """How many multiples of *step* lie in [lower, upper]."""
    first = -(-lower // step)  # ceil(lower / step), without the float
    last = upper // step
    return last - first + 1 if last >= first else 0


def y_tick_step(lower: int, upper: int) -> int:
    """The Y axis tick step for the range [lower, upper], in tokens.

    CHART_Y_TICK_STEP widened additively until the axis would carry fewer
    than CHART_Y_TICK_MAX_LABELS labels.
    """
    lower, upper = _tick_range(lower, upper)
    step = CHART_Y_TICK_STEP
    while _count_ticks(lower, upper, step) >= CHART_Y_TICK_MAX_LABELS:
        step += CHART_Y_TICK_STEP
    return step


def y_tick_values(lower: int, upper: int) -> List[int]:
    """The Y axis tick values in [lower, upper]: multiples of the step.

    Never empty.  A step this coarse can miss a narrow range entirely -- every
    value in one 10k band above the floor -- in which case the axis minimum is
    labelled on its own rather than leaving the axis unlabelled, which is what
    PlotCanvas' margin maths needs (it reads the first and last tick).
    """
    lower, upper = _tick_range(lower, upper)
    step = y_tick_step(lower, upper)
    values = []
    value = -(-lower // step) * step  # the first multiple of step >= lower
    while value <= upper:
        values.append(value)
        value += step
    return values or [lower]


def request_tick_indices(count: int) -> List[int]:
    """The X axis tick positions for *count* requests, ascending.

    Each tick is labelled with its request number (index + 1).  One tick per
    request would be unreadable and meaningless -- the position already says
    which request it is -- so at most four are drawn: the last request, and
    (with CHART_X_TICK_SPREAD_MIN_POINTS requests or more) the ones nearest
    25%, 50% and 75% of the way through; with only
    CHART_X_TICK_PAIR_MIN_POINTS requests, the last one plus the one nearest
    halfway; with fewer still, just the last one.
    """
    if count <= 0:
        return []
    last = count - 1
    if count < CHART_X_TICK_PAIR_MIN_POINTS:
        fractions: Tuple[float, ...] = ()
    elif count < CHART_X_TICK_SPREAD_MIN_POINTS:
        fractions = (0.5,)
    else:
        fractions = CHART_X_TICK_FRACTIONS

    indices = []
    for fraction in fractions:
        # Rounded half up rather than with round(), which is half-to-even and
        # would put the halfway tick of a ten-request run at 4, not 5.
        index = int(fraction * last + 0.5)
        if index not in indices:
            indices.append(index)
    if last not in indices:
        indices.append(last)
    return sorted(indices)
