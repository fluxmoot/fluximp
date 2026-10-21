"""CollapsibleSection - A reusable expandable/collapsible panel widget."""

import os
import wx
import wx.html
import html as html_lib
import logging
from typing import Optional, Callable

from fluximp_wxwidgets import themes

logger = logging.getLogger(__name__)

# Logging plan (2026-10-10 resume-desync diagnosis): widget per-step DEBUG
# lines (10+ lines per section construct) buried every real diagnosis under
# creation spam while carrying no identity — every primary card's header
# shares one label ("API Request (Fork F, Turn T)"). By default widget debug
# spam is dropped by _WidgetLoggerProxy (INFO/WARNING/ERROR pass through)
# and each widget emits ONE INFO identity line instead. Set
# FLUXIMP_WIDGET_TRACE=1 to restore the raw logger and the per-step trace for
# a specific repro.
_WIDGET_TRACE = os.environ.get('FLUXIMP_WIDGET_TRACE', '') not in ('', '0')


class _WidgetLoggerProxy:
    """Debug-dropping stand-in for a Logger (logging plan, 2026-10-10).

    debug() calls are discarded; everything else (info/warning/error/
    exception/name/...) forwards to the wrapped logger, so widget warnings
    still land in the app log while the per-step debug flood is silenced
    unless FLUXIMP_WIDGET_TRACE=1.
    """

    __slots__ = ('_base',)

    def __init__(self, base: logging.Logger):
        self._base = base

    def debug(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return getattr(self._base, name)


class CollapsibleSection(wx.Panel):
    """A reusable expandable/collapsible panel with a clickable header.

    Optimized to minimize native window handle (HWND) usage on Windows:
    - No separate _header_panel (header labels are direct children of self)
    - No separate _content_panel (content items are direct children of self)
    - Lazy content creation: collapsed sections defer HtmlWindow creation
    - This reduces each section from 6 HWNDs to 4, and saves 1+ HWND
      per collapsed section with deferred content, preventing HWND
      exhaustion crashes in long conversations.
    """


    def __init__(self, parent, label: str = "",
                 expanded: bool = True,
                 indent: int = 20,
                 header_color: wx.Colour = None,
                 content_color: wx.Colour = None,
                 logger: Optional[logging.Logger] = None,
                 section_role: str = "",
                 section_ordinal: int = 0,
                 auto_collapsible: bool = True,
                 extra_below: bool = False,
                 on_user_toggle: Optional[Callable] = None):
        # Per logging plan (2026-10-10): wrap the base logger in the
        # debug-dropping proxy unless FLUXIMP_WIDGET_TRACE=1.
        base_logger = logger or globals().get('logger')
        self._logger = (base_logger if _WIDGET_TRACE
                        else _WidgetLoggerProxy(base_logger))
        self._logger.debug(f'CollapsibleSection.__init__: START label={label!r}')
        super().__init__(parent)
        self._logger.debug('CollapsibleSection.__init__: super() done')

        self._section_role = section_role
        # §8.3 per-instance identity: this section's index among the
        # card's same-role sections. The toggle callback reports it so a
        # click stays tied to THIS section instead of cascading to every
        # sibling of that role on the card (defect B1).
        self._section_ordinal = int(section_ordinal or 0)
        self._auto_collapsible = auto_collapsible
        # Extra text on its own centered line below the header row instead
        # of inline at the row's right end (request card headers use this
        # so the identity span never crowds the pending countdown).
        self._extra_below = bool(extra_below)
        self._on_user_toggle = on_user_toggle
        self._html_controls = []  # Track HtmlWindows for deferred height updates
        self._pending_content = []  # (text, min_height) for deferred creation when collapsed

        self._expanded = expanded
        self._label = label
        self._extra_text = ""
        # Lazily-created right-justified identity span (see set_right_text);
        # None until a caller supplies text, so sections that never use it
        # pay no extra native window handle (HWND).
        self._right_label: Optional[wx.StaticText] = None
        self._indicator_text = "v" if expanded else ">"
        self._indent = indent
        self._header_color = header_color or themes.to_wx(themes.active().section_header_bg)
        self._content_color = content_color or themes.to_wx(themes.active().section_content_bg)
        self._logger.debug('CollapsibleSection.__init__: attrs set, calling _build_ui')

        self._build_ui()
        # Logging plan (2026-10-10 resume-desync diagnosis): one identity
        # line per section (INFO passes the proxy) — id() disambiguates
        # same-label sections; role+ordinal say which card part this is
        # (the information the 02:10 app log's per-step spam never carried).
        self._logger.info(
            'CollapsibleSection: created id=%s label=%r role=%r ordinal=%s',
            id(self), self._label, self._section_role, self._section_ordinal)
        self._logger.debug('CollapsibleSection.__init__: COMPLETE')

    def _build_ui(self):
        """Build the collapsible section UI.

        Minimizes native window handles by placing header labels directly
        on the section panel instead of creating a separate _header_panel,
        and using sizers directly instead of a separate _content_panel.
        """
        self._logger.debug('CollapsibleSection._build_ui: START')
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Set section background to header color
        # (header labels sit on this; content items have their own backgrounds)
        self.SetBackgroundColour(self._header_color)

        # Header area - labels directly in main_sizer (no separate _header_panel)
        self._logger.debug('CollapsibleSection._build_ui: creating header sizer')
        self._header_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Combine indicator and label into a single StaticText to save 1 HWND per section.
        # Format: 'v Content' or '> Content'
        self._logger.debug('CollapsibleSection._build_ui: creating header label')
        self._header_label = wx.StaticText(
            self, label=f'{self._indicator_text} {self._label}'
        )
        header_font = self._header_label.GetFont()
        header_font.SetWeight(wx.FONTWEIGHT_BOLD)
        self._header_label.SetFont(header_font)
        self._header_label.Bind(wx.EVT_LEFT_DOWN, self._on_header_click)
        self._header_label.SetCursor(wx.Cursor(wx.CURSOR_HAND))
        self._header_sizer.Add(self._header_label, 0, wx.LEFT | wx.TOP | wx.BOTTOM, 5)

        self._header_sizer.AddStretchSpacer(1)

        self._logger.debug('CollapsibleSection._build_ui: creating extra label')
        self._extra_label = wx.StaticText(
            self, label=self._extra_text
        )
        extra_font = self._extra_label.GetFont()
        self._extra_label.SetFont(extra_font)
        self._extra_label.SetForegroundColour(themes.to_wx(themes.active().muted_fg))
        if self._extra_below:
            # Own centered line beneath the header row (request card
            # headers): the extra text (e.g. the "pending: Ns" countdown)
            # is the only text on that line, so the right-justified
            # identity span never crowds it back onto the header row.
            # Starts hidden; set_extra() shows it while pending/cancelled
            # and hides it again once the response clears the text, so a
            # completed card spends no vertical space on the line.
            self._extra_label.Hide()
        else:
            self._header_sizer.Add(self._extra_label, 0, wx.RIGHT | wx.TOP | wx.BOTTOM, 5)

        main_sizer.Add(self._header_sizer, 0, wx.EXPAND)

        if self._extra_below:
            extra_row = wx.BoxSizer(wx.HORIZONTAL)
            extra_row.AddStretchSpacer(1)
            extra_row.Add(self._extra_label, 0, wx.TOP | wx.BOTTOM, 2)
            extra_row.AddStretchSpacer(1)
            main_sizer.Add(extra_row, 0, wx.EXPAND)

        # Bind click on extra label too
        self._extra_label.Bind(wx.EVT_LEFT_DOWN, self._on_header_click)
        self._extra_label.SetCursor(wx.Cursor(wx.CURSOR_HAND))

        # Content area - no separate _content_panel, just sizers directly
        self._logger.debug('CollapsibleSection._build_ui: creating content sizers')
        self._content_inner_sizer = wx.BoxSizer(wx.HORIZONTAL)
        # Use spacer instead of a panel for indent - avoids creating an extra
        # window handle per CollapsibleSection.
        self._content_inner_sizer.Add((self._indent, -1), 0, wx.EXPAND)
        self._content_sizer = wx.BoxSizer(wx.VERTICAL)
        # Dedicated sizer for HtmlWindows - allows destroying/recreating
        # them on collapse/expand without disturbing other content (like btn_panels)
        self._html_sizer = wx.BoxSizer(wx.VERTICAL)
        self._content_sizer.Add(self._html_sizer, 0, wx.EXPAND)
        self._content_inner_sizer.Add(self._content_sizer, 1, wx.EXPAND)

        # Track the sizer item for show/hide (instead of _content_panel.Hide())
        # Note: we do NOT call ShowItems(False) here. Instead, we rely on
        # the sizer item's Show(False) to prevent content from being visible.
        # This preserves each window's individual shown state, so explicitly-
        # hidden windows (like resolved permission buttons) remain hidden
        # when the section is later expanded.
        self._content_sizer_item = main_sizer.Add(self._content_inner_sizer, 0, wx.EXPAND)

        if not self._expanded:
            self._content_sizer_item.Show(False)

        self.SetSizer(main_sizer)
        self._logger.debug('CollapsibleSection._build_ui: COMPLETE')

    def _on_header_click(self, event: wx.Event):
        """Handle click on the header - toggle expanded state."""
        self.toggle()

    def toggle(self):
        """Toggle the expanded state."""
        if not self._auto_collapsible:
            return
        if self._expanded:
            self.collapse()
        else:
            self.expand()

    def expand(self, notify: bool = True):
        """Expand the section - show content."""
        if self._expanded:
            return

        self._expanded = True
        self._indicator_text = "v"
        self._header_label.SetLabel(f'{self._indicator_text} {self._label}')

        # Create any pending content that was deferred
        self._create_pending_content()

        # Show content sizer item - individual windows' shown states
        # are preserved, so explicitly-hidden windows (like resolved
        # permission buttons) remain hidden when expanded.
        self._content_sizer_item.Show(True)

        # §8.5: heights are measured synchronously at build time (fixed §8.2
        # width before SetPage); no deferred height update is needed.
        # The controller/view re-measure on the next flush.

        if notify and self._on_user_toggle:
            # §8.3: the callback carries the section's per-instance
            # ordinal so one click stays tied to THIS section (defect B1).
            self._on_user_toggle(self._section_role, self._section_ordinal,
                                 True)

        self._update_layout()

    def collapse(self, notify: bool = True):
        """Collapse the section - hide content."""
        if not self._expanded:
            return

        self._expanded = False
        self._indicator_text = ">"
        self._header_label.SetLabel(f'{self._indicator_text} {self._label}')

        # Destroy HtmlWindows to free native window handles (HWNDs).
        # Their text is stored in _pending_content for recreation on expand().
        self._destroy_content()

        # Hide the content sizer item - individual windows' shown states
        # are preserved, so explicitly-hidden windows (like resolved
        # permission buttons) remain hidden when re-expanded.
        self._content_sizer_item.Show(False)

        if notify and self._on_user_toggle:
            # §8.3: the callback carries the section's per-instance
            # ordinal so one click stays tied to THIS section (defect B1).
            self._on_user_toggle(self._section_role, self._section_ordinal,
                                 False)

        self._update_layout()

    def _create_pending_content(self):
        """Create any content that was deferred because the section was collapsed."""
        for text, min_height in self._pending_content:
            self._create_rich_text_widget(text, min_height)
        self._pending_content.clear()

    def _destroy_content(self):
        """Destroy HtmlWindows in the content area to free native window handles (HWNDs).

        Stores their text data for recreation when the section is expanded.
        This is critical on Windows where HWND exhaustion causes crashes in
        long conversations with many CollapsibleSections.
        """
        for ctrl, text, min_height in self._html_controls:
            # Store text for recreation on expand
            self._pending_content.append((text, min_height))
            try:
                if not ctrl.IsBeingDeleted():
                    self._html_sizer.Detach(ctrl)
                    ctrl.Destroy()
            except RuntimeError:
                pass  # Window already destroyed
        self._html_controls.clear()

    def _update_layout(self):
        """Update layout after expand/collapse."""
        self.Layout()

    @property
    def is_expanded(self) -> bool:
        """Whether the section is currently expanded."""
        return self._expanded

    @property
    def section_ordinal(self) -> int:
        """Index among the card's same-role sections (§8.3 identity)."""
        return self._section_ordinal

    def set_label(self, label: str):
        """Set the header label text."""
        self._label = label
        self._header_label.SetLabel(f'{self._indicator_text} {self._label}')
        self.Layout()

    def set_indicator(self, indicator: str):
        """Set the indicator symbol."""
        self._indicator_text = indicator
        self._header_label.SetLabel(f'{self._indicator_text} {self._label}')
        self.Layout()

    def set_extra(self, extra: str):
        """Set the extra text in the header.

        extra_below sections show the text on its own centered line below
        the header row; that line collapses away while the text is empty
        (e.g. the request card's set_extra("") once the response arrives),
        so a completed header spends no vertical space on it.
        """
        self._extra_text = extra
        self._extra_label.SetLabel(extra)
        if self._extra_below:
            self._extra_label.Show(bool(extra))
        self.Layout()

    def set_right_text(self, text: str):
        """Set the right-justified identity span in the header row.

        Lazily creates a gray, non-bold StaticText styled like the
        footer/extra labels (not the bold title). The span is inserted
        between the stretch spacer and _extra_label, so the _extra_label
        text (e.g. the pending/cancelled status) stays rightmost. Sections
        that never call this do not pay the extra native window handle.
        """
        if not text and self._right_label is None:
            # Nothing to show yet; do not create an empty label.
            return
        if self._right_label is None:
            self._right_label = wx.StaticText(self)
            self._right_label.SetForegroundColour(themes.to_wx(themes.active().muted_fg))
            # Insert after the stretch spacer (item 0 is the title, item 1
            # the stretch spacer), keeping the span rightmost on the row;
            # in inline-extra mode it lands before _extra_label, which then
            # stays the true rightmost text.
            self._header_sizer.Insert(
                2, self._right_label, 0, wx.LEFT | wx.TOP | wx.BOTTOM, 5)
            # Match the other header labels' click/cursor behavior (the
            # request cards' auto_collapsible=False keeps this a no-op).
            self._right_label.Bind(wx.EVT_LEFT_DOWN, self._on_header_click)
            self._right_label.SetCursor(wx.Cursor(wx.CURSOR_HAND))
        self._right_label.SetLabel(text)
        # Hide the span when it becomes empty (e.g. every API metadata
        # item it carries toggled off) so the header row does not reserve
        # a blank gap; a later non-empty set_right_text re-shows it.
        self._right_label.Show(bool(text))
        self.Layout()

    def get_content_panel(self) -> wx.Panel:
        """Get the panel where widgets can be added.

        Returns self (the CollapsibleSection) since there is no separate
        content panel. Content items are direct children of this section.
        """
        return self

    def get_content_sizer(self) -> wx.BoxSizer:
        """Get the content sizer where widgets can be added directly."""
        return self._content_sizer

    def add_rich_text(self, text: str, min_height: int = 30) -> Optional[wx.html.HtmlWindow]:
        """Add an HtmlWindow to the content area for selectable/copyable text.

        If the section is currently collapsed, the content is deferred and
        None is returned. The HtmlWindow will be created when the section
        is expanded. This saves native window handles (HWNDs) for collapsed
        sections, preventing exhaustion in long conversations.
        """
        if self._expanded:
            return self._create_rich_text_widget(text, min_height)
        else:
            # Defer creation until expand() is called - saves HWNDs
            self._pending_content.append((text, min_height))
            return None

    def _create_rich_text_widget(self, text: str, min_height: int = 30) -> wx.html.HtmlWindow:
        """Create an HtmlWindow widget for rich text content."""
        ctrl = wx.html.HtmlWindow(
            self,
            style=wx.html.HW_SCROLLBAR_NEVER
        )

        # Build HTML content - escape and format plain text
        escaped = html_lib.escape(text)
        formatted = escaped.replace('\n', '<br>\n')
        bg_hex = self._content_color.GetAsString(wx.C2S_HTML_SYNTAX)
        fg_hex = themes.to_wx(themes.active().text_fg).GetAsString(wx.C2S_HTML_SYNTAX)

        html_content = (
            f'<html><body bgcolor="{bg_hex}">'
            f'<font face="sans-serif" size="3" color="{fg_hex}">{formatted}</font>'
            f'</body></html>'
        )

        ctrl.SetPage(html_content)
        ctrl.SetBackgroundColour(self._content_color)

        # Track for HWND freeing on collapse (text preserved in _destroy_content).
        self._html_controls.append((ctrl, text, min_height))

        # Height is measured after Layout in WorkflowHistoryView._measure_widget
        # (which reads the HtmlWindow's internal representation at the laid-out
        # §8.2 width); measuring here at the control's default width produced
        # over-tall min heights and excessive whitespace (§8.5).

        self._html_sizer.Add(ctrl, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # If section is collapsed, hide the newly created widget
        if not self._expanded:
            ctrl.Hide()

        return ctrl


    def set_header_color(self, color: wx.Colour):
        """Set the header background color."""
        self._header_color = color
        self.SetBackgroundColour(color)
        self.Refresh()

    def set_content_color(self, color: wx.Colour):
        """Set the content area background color."""
        self._content_color = color
        # Update existing HtmlWindows
        for ctrl, text, min_height in self._html_controls:
            try:
                if ctrl.IsBeingDeleted():
                    continue
            except RuntimeError:
                continue
            ctrl.SetBackgroundColour(color)
        self.Refresh()

    def set_indent(self, indent: int):
        """Set the left indent for content area in pixels."""
        self._indent = indent
        if self._content_inner_sizer.GetItemCount() > 0:
            indent_item = self._content_inner_sizer.GetItem(0)
            if indent_item:
                # Spacer items use AssignSpacer to change size
                indent_item.AssignSpacer((indent, -1))
                self.Layout()
