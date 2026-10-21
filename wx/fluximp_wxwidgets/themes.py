"""Application themes: named color palettes for the FluxImp wx GUI.

A Theme is the single source of truth for every color the GUI paints.  It
is a frozen record of semantic *roles* (window chrome, status text,
conversation-history message tints, request-card tints, ...), each an
``(r, g, b)`` int tuple.

This module deliberately imports no ``wx`` so the palette tests in
test/tool_tests/themes_tests.py run headless (the same trick
markdown_render.py uses); convert to a ``wx.Colour`` with ``to_wx`` at
paint time.

``Amethyst`` is the default (``DEFAULT_THEME_KEY``).  ``Troop`` -- named for
the proof-of-concept that preceded FluxImp -- is not the default; it
reproduces, value for value, the palette the GUI hard-coded before themes
existed.

Icons are intentionally NOT themed: the Workspaces tree bitmaps
(left_panel.py), the Workflows tree bitmaps (workflow_panel.py) and the tab
status icons (tab_status_icons.py) keep their fixed colors, so no
bitmap-drawing color appears here.

The active theme is a module-level singleton, mirroring
tab_status_icons._icon_cache: main.py / MainFrame call set_active() once per
switch and every widget resolves colors through active() at construction.
Because a switch is only reachable with zero open tabs (MainFrame gates the
View > Theme submenu on that), no already-materialized card ever needs
re-theming.
"""

from dataclasses import dataclass, fields
from typing import Dict, List, Tuple

# A color role value: (red, green, blue), each 0-255.
RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class Theme:
    """One named palette.  Every field is an ``(r, g, b)`` tuple."""

    # -- Window chrome (frame, panels, collapsible-section defaults) --
    window_bg: RGB
    panel_bg: RGB
    panel_border: RGB
    section_header_bg: RGB
    section_content_bg: RGB
    text_fg: RGB
    muted_fg: RGB
    subtle_fg: RGB

    # -- Status text (right panel process status; workspace settings line) --
    status_running_fg: RGB
    status_stopped_fg: RGB
    settings_fail_fg: RGB
    settings_unchecked_fg: RGB
    settings_saved_fg: RGB

    # -- Conversation-history message roles (workflow_history_view.MessagePanel) --
    msg_user_bg: RGB
    msg_user_fg: RGB
    msg_user_border: RGB
    msg_assistant_bg: RGB
    msg_assistant_fg: RGB
    msg_assistant_border: RGB
    msg_tool_result_bg: RGB
    msg_tool_result_fg: RGB
    msg_permission_bg: RGB
    msg_permission_fg: RGB
    msg_tool_pending_bg: RGB
    msg_tool_pending_fg: RGB
    msg_tool_ok_bg: RGB
    msg_tool_ok_fg: RGB
    msg_tool_fail_bg: RGB
    msg_tool_fail_fg: RGB
    msg_default_bg: RGB
    msg_default_fg: RGB

    # -- Request cards (request_card.py + collapsible_section.py) --
    card_system_header: RGB
    card_system_content: RGB
    card_conditional_header: RGB
    card_conditional_content: RGB
    card_primary_header: RGB
    card_primary_content: RGB
    card_cancelled: RGB
    card_permission_header: RGB
    card_permission_content: RGB
    response_header: RGB
    response_content: RGB
    accent_conditional: RGB
    accent_system: RGB
    tool_header_pending: RGB
    tool_header_ok: RGB
    tool_header_fail: RGB
    tool_content: RGB
    report_conditional_header: RGB
    report_conditional_content: RGB
    report_primary_header: RGB
    report_primary_content: RGB

    # -- API Chart (right panel per-request token lines; see api_chart.py) --
    # Spliced in from the shared LIGHT_CHART/DARK_CHART set below.
    chart_bg: RGB
    chart_fg: RGB
    chart_grid: RGB
    chart_line_prompt: RGB
    chart_line_cached: RGB
    chart_line_completion: RGB
    chart_line_reasoning: RGB


# Every role name, in declaration order.  Used by the palette tests to
# assert completeness across all themes.
ROLES: List[str] = [f.name for f in fields(Theme)]


# --------------------------------------------------------------------------
# Shared API Chart colour set.  The chart (right_panel.ForkTokenChart) is one
# widget with its own small palette, and the three light themes want the same
# one just as the three dark themes want the same one -- so the values live
# here once and each theme splices its set in with ``**LIGHT_CHART`` /
# ``**DARK_CHART`` instead of repeating them three times over.
# --------------------------------------------------------------------------
LIGHT_CHART: Dict[str, RGB] = {
    "chart_bg": (255, 255, 255),
    "chart_fg": (0, 0, 0),
    "chart_grid": (200, 200, 200),
    "chart_line_prompt": (0, 90, 200),
    "chart_line_cached": (0, 140, 70),
    "chart_line_completion": (210, 105, 0),
    "chart_line_reasoning": (130, 40, 180),
}

DARK_CHART: Dict[str, RGB] = {
    "chart_bg": (32, 33, 37),
    "chart_fg": (230, 230, 235),
    "chart_grid": (80, 80, 88),
    "chart_line_prompt": (110, 170, 255),
    "chart_line_cached": (110, 215, 140),
    "chart_line_completion": (255, 175, 80),
    "chart_line_reasoning": (195, 145, 255),
}


# --------------------------------------------------------------------------
# Troop -- NOT the default (see DEFAULT_THEME_KEY).  Value-for-value the
# palette the GUI hard-coded before themes existed (kept as a regression
# guard by themes_tests.py).
# --------------------------------------------------------------------------
TROOP = Theme(
    window_bg=(240, 240, 240),
    panel_bg=(255, 255, 255),
    panel_border=(180, 180, 180),
    section_header_bg=(240, 240, 240),
    section_content_bg=(255, 255, 255),
    text_fg=(0, 0, 0),
    muted_fg=(128, 128, 128),
    subtle_fg=(80, 80, 80),

    status_running_fg=(0, 0, 0),
    status_stopped_fg=(255, 140, 0),
    settings_fail_fg=(190, 0, 0),
    settings_unchecked_fg=(170, 105, 0),
    settings_saved_fg=(0, 128, 64),

    msg_user_bg=(230, 240, 255),
    msg_user_fg=(50, 100, 150),
    msg_user_border=(204, 85, 0),
    msg_assistant_bg=(220, 245, 220),
    msg_assistant_fg=(30, 120, 30),
    msg_assistant_border=(199, 21, 133),
    msg_tool_result_bg=(255, 255, 230),
    msg_tool_result_fg=(150, 150, 50),
    msg_permission_bg=(255, 240, 230),
    msg_permission_fg=(200, 100, 50),
    msg_tool_pending_bg=(255, 250, 200),
    msg_tool_pending_fg=(200, 150, 50),
    msg_tool_ok_bg=(230, 255, 230),
    msg_tool_ok_fg=(50, 150, 50),
    msg_tool_fail_bg=(255, 230, 230),
    msg_tool_fail_fg=(200, 50, 50),
    msg_default_bg=(255, 255, 255),
    msg_default_fg=(0, 0, 0),

    card_system_header=(252, 252, 252),
    card_system_content=(248, 248, 248),
    card_conditional_header=(255, 245, 230),
    card_conditional_content=(255, 236, 210),
    card_primary_header=(255, 250, 230),
    card_primary_content=(230, 255, 230),
    card_cancelled=(255, 230, 230),
    card_permission_header=(255, 240, 230),
    card_permission_content=(255, 250, 245),
    response_header=(240, 248, 240),
    response_content=(245, 255, 245),
    accent_conditional=(210, 150, 60),
    accent_system=(160, 160, 160),
    tool_header_pending=(255, 250, 200),
    tool_header_ok=(230, 255, 230),
    tool_header_fail=(255, 230, 230),
    tool_content=(255, 255, 245),
    report_conditional_header=(255, 240, 217),
    report_conditional_content=(255, 245, 230),
    report_primary_header=(255, 250, 200),
    report_primary_content=(255, 255, 245),
    **LIGHT_CHART,
)


# --------------------------------------------------------------------------
# Slate -- cool neutral light.
# --------------------------------------------------------------------------
SLATE = Theme(
    window_bg=(238, 241, 245),
    panel_bg=(250, 251, 253),
    panel_border=(190, 196, 205),
    section_header_bg=(226, 231, 238),
    section_content_bg=(250, 251, 253),
    text_fg=(24, 29, 36),
    muted_fg=(120, 128, 140),
    subtle_fg=(90, 98, 110),

    status_running_fg=(20, 90, 60),
    status_stopped_fg=(200, 115, 20),
    settings_fail_fg=(180, 30, 30),
    settings_unchecked_fg=(165, 100, 0),
    settings_saved_fg=(0, 120, 60),

    msg_user_bg=(230, 233, 255),
    msg_user_fg=(50, 63, 150),
    msg_user_border=(204, 177, 0),
    msg_assistant_bg=(220, 245, 240),
    msg_assistant_fg=(29, 116, 99),
    msg_assistant_border=(199, 21, 57),
    msg_tool_result_bg=(254, 255, 230),
    msg_tool_result_fg=(147, 150, 50),
    msg_permission_bg=(255, 237, 230),
    msg_permission_fg=(200, 95, 50),
    msg_tool_pending_bg=(253, 255, 200),
    msg_tool_pending_fg=(159, 163, 41),
    msg_tool_ok_bg=(230, 255, 250),
    msg_tool_ok_fg=(49, 146, 127),
    msg_tool_fail_bg=(255, 230, 235),
    msg_tool_fail_fg=(196, 49, 78),
    msg_default_bg=(255, 255, 255),
    msg_default_fg=(24, 29, 36),

    card_system_header=(226, 230, 236),
    card_system_content=(236, 240, 245),
    card_conditional_header=(255, 252, 230),
    card_conditional_content=(255, 249, 210),
    card_primary_header=(254, 255, 230),
    card_primary_content=(230, 255, 250),
    card_cancelled=(255, 230, 235),
    card_permission_header=(255, 237, 230),
    card_permission_content=(255, 248, 245),
    response_header=(240, 248, 246),
    response_content=(245, 255, 253),
    accent_conditional=(210, 190, 60),
    accent_system=(140, 146, 158),
    tool_header_pending=(253, 255, 200),
    tool_header_ok=(230, 255, 250),
    tool_header_fail=(255, 230, 235),
    tool_content=(255, 255, 245),
    report_conditional_header=(255, 250, 217),
    report_conditional_content=(255, 252, 230),
    report_primary_header=(253, 255, 200),
    report_primary_content=(255, 255, 245),
    **LIGHT_CHART,
)


# --------------------------------------------------------------------------
# Amethyst -- light purple.
# --------------------------------------------------------------------------
AMETHYST = Theme(
    window_bg=(243, 240, 248),
    panel_bg=(251, 249, 254),
    panel_border=(198, 190, 212),
    section_header_bg=(232, 225, 242),
    section_content_bg=(251, 249, 254),
    text_fg=(38, 30, 50),
    muted_fg=(128, 118, 146),
    subtle_fg=(98, 88, 116),

    status_running_fg=(86, 40, 140),
    status_stopped_fg=(176, 96, 20),
    settings_fail_fg=(176, 32, 64),
    settings_unchecked_fg=(150, 96, 10),
    settings_saved_fg=(16, 120, 96),

    msg_user_bg=(242, 230, 255),
    msg_user_fg=(97, 50, 150),
    msg_user_border=(109, 204, 0),
    msg_assistant_bg=(220, 245, 232),
    msg_assistant_fg=(29, 116, 70),
    msg_assistant_border=(199, 21, 116),
    msg_tool_result_bg=(255, 250, 230),
    msg_tool_result_fg=(150, 130, 50),
    msg_permission_bg=(255, 230, 242),
    msg_permission_fg=(200, 50, 125),
    msg_tool_pending_bg=(255, 244, 200),
    msg_tool_pending_fg=(180, 153, 45),
    msg_tool_ok_bg=(230, 255, 242),
    msg_tool_ok_fg=(49, 146, 95),
    msg_tool_fail_bg=(255, 233, 230),
    msg_tool_fail_fg=(188, 66, 47),
    msg_default_bg=(255, 255, 255),
    msg_default_fg=(38, 30, 50),

    card_system_header=(236, 232, 244),
    card_system_content=(244, 241, 250),
    card_conditional_header=(243, 255, 230),
    card_conditional_content=(234, 255, 210),
    card_primary_header=(255, 250, 230),
    card_primary_content=(230, 255, 242),
    card_cancelled=(255, 233, 230),
    card_permission_header=(255, 230, 242),
    card_permission_content=(255, 245, 250),
    response_header=(240, 248, 244),
    response_content=(245, 255, 250),
    accent_conditional=(140, 210, 60),
    accent_system=(150, 142, 166),
    tool_header_pending=(255, 244, 200),
    tool_header_ok=(230, 255, 242),
    tool_header_fail=(255, 233, 230),
    tool_content=(255, 253, 245),
    report_conditional_header=(237, 255, 217),
    report_conditional_content=(243, 255, 230),
    report_primary_header=(255, 244, 200),
    report_primary_content=(255, 253, 245),
    **LIGHT_CHART,
)


# --------------------------------------------------------------------------
# Midnight -- dark.
# --------------------------------------------------------------------------
MIDNIGHT = Theme(
    window_bg=(30, 31, 34),
    panel_bg=(37, 38, 42),
    panel_border=(60, 61, 66),
    section_header_bg=(45, 46, 52),
    section_content_bg=(37, 38, 42),
    text_fg=(228, 228, 232),
    muted_fg=(150, 150, 158),
    subtle_fg=(170, 170, 178),

    # On a dark panel the "running" text must be light, not black.
    status_running_fg=(120, 215, 140),
    status_stopped_fg=(255, 175, 70),
    settings_fail_fg=(240, 120, 120),
    settings_unchecked_fg=(230, 180, 90),
    settings_saved_fg=(120, 210, 140),

    msg_user_bg=(30, 51, 81),
    msg_user_fg=(147, 192, 254),
    msg_user_border=(230, 116, 11),
    msg_assistant_bg=(27, 59, 33),
    msg_assistant_fg=(145, 208, 153),
    msg_assistant_border=(230, 101, 156),
    msg_tool_result_bg=(76, 54, 23),
    msg_tool_result_fg=(229, 180, 115),
    msg_permission_bg=(73, 37, 55),
    msg_permission_fg=(239, 163, 199),
    msg_tool_pending_bg=(52, 31, 0),
    msg_tool_pending_fg=(252, 189, 104),
    msg_tool_ok_bg=(29, 72, 38),
    msg_tool_ok_fg=(156, 230, 166),
    msg_tool_fail_bg=(77, 38, 32),
    msg_tool_fail_fg=(247, 165, 152),
    msg_default_bg=(37, 38, 42),
    msg_default_fg=(228, 228, 232),

    card_system_header=(48, 49, 54),
    card_system_content=(42, 43, 48),
    card_conditional_header=(81, 50, 30),
    card_conditional_content=(62, 32, 12),
    card_primary_header=(76, 54, 23),
    card_primary_content=(58, 40, 14),
    card_cancelled=(86, 36, 30),
    card_permission_header=(80, 46, 63),
    card_permission_content=(61, 29, 45),
    response_header=(68, 48, 18),
    response_content=(52, 35, 12),
    accent_conditional=(231, 116, 8),
    accent_system=(150, 150, 160),
    tool_header_pending=(62, 37, 0),
    tool_header_ok=(38, 76, 44),
    tool_header_fail=(92, 46, 39),
    tool_content=(57, 36, 4),
    report_conditional_header=(94, 60, 37),
    report_conditional_content=(71, 43, 26),
    report_primary_header=(93, 62, 13),
    report_primary_content=(68, 46, 15),
    **DARK_CHART,
)


# --------------------------------------------------------------------------
# Nightshade -- dark purple.
# --------------------------------------------------------------------------
NIGHTSHADE = Theme(
    window_bg=(26, 22, 34),
    panel_bg=(33, 28, 42),
    panel_border=(58, 50, 72),
    section_header_bg=(44, 37, 56),
    section_content_bg=(33, 28, 42),
    text_fg=(230, 226, 238),
    muted_fg=(156, 146, 172),
    subtle_fg=(176, 166, 192),

    status_running_fg=(150, 220, 170),
    status_stopped_fg=(240, 180, 110),
    settings_fail_fg=(245, 130, 150),
    settings_unchecked_fg=(235, 190, 120),
    settings_saved_fg=(130, 220, 180),

    msg_user_bg=(63, 41, 71),
    msg_user_fg=(215, 170, 233),
    msg_user_border=(83, 177, 71),
    msg_assistant_bg=(0, 60, 51),
    msg_assistant_fg=(110, 211, 190),
    msg_assistant_border=(239, 103, 94),
    msg_tool_result_bg=(66, 59, 22),
    msg_tool_result_fg=(207, 191, 113),
    msg_permission_bg=(75, 37, 52),
    msg_permission_fg=(242, 163, 192),
    msg_tool_pending_bg=(43, 36, 0),
    msg_tool_pending_fg=(224, 203, 101),
    msg_tool_ok_bg=(0, 72, 61),
    msg_tool_ok_fg=(113, 234, 209),
    msg_tool_fail_bg=(77, 38, 33),
    msg_tool_fail_fg=(247, 165, 153),
    msg_default_bg=(33, 28, 42),
    msg_default_fg=(230, 226, 238),

    card_system_header=(44, 38, 56),
    card_system_content=(38, 32, 48),
    card_conditional_header=(42, 66, 38),
    card_conditional_content=(24, 48, 21),
    card_primary_header=(66, 59, 22),
    card_primary_content=(50, 42, 12),
    card_cancelled=(86, 36, 30),
    card_permission_header=(81, 46, 59),
    card_permission_content=(62, 28, 42),
    response_header=(60, 51, 16),
    response_content=(45, 38, 10),
    accent_conditional=(80, 178, 67),
    accent_system=(160, 150, 178),
    tool_header_pending=(51, 43, 0),
    tool_header_ok=(0, 78, 67),
    tool_header_fail=(93, 46, 39),
    tool_content=(48, 41, 3),
    report_conditional_header=(50, 77, 47),
    report_conditional_content=(36, 57, 33),
    report_primary_header=(80, 69, 10),
    report_primary_content=(58, 51, 13),
    **DARK_CHART,
)


# --------------------------------------------------------------------------
# Carbon -- high-contrast dark (accessibility).
# --------------------------------------------------------------------------
CARBON = Theme(
    window_bg=(0, 0, 0),
    panel_bg=(10, 10, 10),
    panel_border=(80, 80, 80),
    section_header_bg=(18, 18, 18),
    section_content_bg=(10, 10, 10),
    text_fg=(255, 255, 255),
    muted_fg=(180, 180, 180),
    subtle_fg=(210, 210, 210),

    status_running_fg=(0, 255, 90),
    status_stopped_fg=(255, 200, 0),
    settings_fail_fg=(255, 80, 80),
    settings_unchecked_fg=(255, 210, 0),
    settings_saved_fg=(0, 255, 120),

    msg_user_bg=(25, 53, 80),
    msg_user_fg=(140, 195, 252),
    msg_user_border=(235, 111, 48),
    msg_assistant_bg=(31, 59, 30),
    msg_assistant_fg=(151, 207, 148),
    msg_assistant_border=(217, 105, 188),
    msg_tool_result_bg=(60, 61, 25),
    msg_tool_result_fg=(194, 196, 118),
    msg_permission_bg=(74, 41, 19),
    msg_permission_fg=(241, 172, 128),
    msg_tool_pending_bg=(38, 38, 0),
    msg_tool_pending_fg=(208, 210, 108),
    msg_tool_ok_bg=(35, 71, 34),
    msg_tool_ok_fg=(163, 229, 160),
    msg_tool_fail_bg=(77, 38, 32),
    msg_tool_fail_fg=(247, 165, 152),
    msg_default_bg=(10, 10, 10),
    msg_default_fg=(255, 255, 255),

    card_system_header=(20, 20, 20),
    card_system_content=(14, 14, 14),
    card_conditional_header=(82, 49, 34),
    card_conditional_content=(63, 31, 16),
    card_primary_header=(60, 61, 25),
    card_primary_content=(46, 46, 16),
    card_cancelled=(86, 36, 30),
    card_permission_header=(81, 50, 30),
    card_permission_content=(62, 32, 12),
    response_header=(56, 56, 22),
    response_content=(40, 40, 13),
    accent_conditional=(237, 109, 42),
    accent_system=(190, 190, 190),
    tool_header_pending=(46, 46, 0),
    tool_header_ok=(42, 76, 41),
    tool_header_fail=(92, 46, 39),
    tool_content=(43, 43, 5),
    report_conditional_header=(96, 58, 41),
    report_conditional_content=(72, 42, 29),
    report_primary_header=(72, 72, 16),
    report_primary_content=(53, 53, 16),
    **DARK_CHART,
)


# Troop is the pre-themes palette: it reproduces, value for value, the colours
# the GUI hard-coded before themes existed.  It is deliberately NOT the default
# any more, but main_frame still branches on it (platform chrome replay and the
# native appearance), so it gets a name rather than a bare literal.
TROOP_THEME_KEY = "troop"

# Registry: insertion order is the View > Theme submenu order.
THEMES: Dict[str, Theme] = {
    TROOP_THEME_KEY: TROOP,
    "slate": SLATE,
    "amethyst": AMETHYST,
    "midnight": MIDNIGHT,
    "nightshade": NIGHTSHADE,
    "carbon": CARBON,
}

# Human-readable names for the menu (key -> label).
DISPLAY_NAMES: Dict[str, str] = {
    "troop": "Troop",
    "slate": "Slate",
    "amethyst": "Amethyst",
    "midnight": "Midnight",
    "nightshade": "Nightshade",
    "carbon": "Carbon",
}

# The single source of truth for the default theme: a *key* into THEMES, not a
# DISPLAY_NAMES label.  It must be a registered key in normalised (lowercase)
# form -- AppSettings normalises the persisted value with str.strip().lower(),
# so a default that is not already normalised silently changes identity when
# settings.json is reloaded.
DEFAULT_THEME_KEY = "amethyst"

# Fail fast on a typo or a theme rename here.  Without this the mistake only
# surfaces at first paint (active()) or in the palette tests, never at import.
assert DEFAULT_THEME_KEY in THEMES, (
    f"default theme {DEFAULT_THEME_KEY!r} is not registered in THEMES")
assert set(DISPLAY_NAMES) == set(THEMES), (
    "every theme needs exactly one DISPLAY_NAMES label")


def theme_names() -> List[str]:
    """Return every theme key in menu order."""
    return list(THEMES.keys())


def get_theme(name: str) -> Theme:
    """Return the theme for *name* (case-insensitive).

    An unknown name (hand-edited settings.json, a theme removed between
    versions) falls back to the default rather than raising.
    """
    if isinstance(name, str):
        theme = THEMES.get(name.strip().lower())
        if theme is not None:
            return theme
    return THEMES[DEFAULT_THEME_KEY]


# -- Active-theme singleton -------------------------------------------------

_active_name: str = DEFAULT_THEME_KEY


def set_active(name: str) -> str:
    """Set the process-wide active theme; returns the resolved key.

    Unknown names resolve to the default so callers can persist the
    return value and stay consistent with what is actually painted.
    """
    global _active_name
    if isinstance(name, str) and name.strip().lower() in THEMES:
        _active_name = name.strip().lower()
    else:
        _active_name = DEFAULT_THEME_KEY
    return _active_name


def active_name() -> str:
    """Return the key of the active theme."""
    return _active_name


def active() -> Theme:
    """Return the active theme's palette."""
    return THEMES[_active_name]


def to_wx(rgb: RGB):
    """Convert an ``(r, g, b)`` tuple to a ``wx.Colour``.

    ``wx`` is imported here (not at module scope) so importing themes.py
    stays free of the GUI toolkit and the palette tests run headless.
    """
    import wx
    return wx.Colour(rgb[0], rgb[1], rgb[2])


def relative_luminance(rgb: RGB) -> float:
    """WCAG relative luminance of an ``(r, g, b)`` tuple (0.0 black .. 1.0 white).

    The same definition wxWidgets uses internally for its own light/dark
    decisions (``wxAuiGetRelativeLuminance``), so a theme classified dark here
    matches what wx itself would decide.
    """
    def channel(value: int) -> float:
        srgb = value / 255.0
        return srgb / 12.92 if srgb <= 0.03928 else ((srgb + 0.055) / 1.055) ** 2.4

    return (0.2126 * channel(rgb[0])
            + 0.7152 * channel(rgb[1])
            + 0.0722 * channel(rgb[2]))


def is_dark(name: str) -> bool:
    """True when *name*'s window background is a dark color.

    Derived from the palette rather than stored as a flag on Theme, so a theme
    cannot disagree with its own background.  Every theme classifies with a
    wide margin -- Troop/Slate/Amethyst are ~0.85 and Midnight/Nightshade/
    Carbon are <0.02 -- so the 0.5 threshold sits in an empty middle.

    Used to pick the native light/dark appearance for the surfaces wxWidgets
    will not let an app color directly: the window title bar, the native menu
    bar and the standard controls inside the panes (see main_frame
    .apply_appearance).
    """
    return relative_luminance(get_theme(name).window_bg) < 0.5
