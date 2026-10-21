"""Right panel - Fork/turn grid for navigating workflow history."""

import wx
from fluximp_agent import fs_agents
from fluximp_wxwidgets.workflow_config import WorkflowConfigManager, STOCK_WORKFLOW_NAME
import wx.grid
from typing import Optional, Callable, List

from fluximp_agent.model_config import get_provider_names, create_provider
from fluximp_agent.curated_models import filter_provider_names, filter_model_names
from fluximp_wxwidgets.app_settings import AppSettings
from fluximp_wxwidgets.conversation_manager import (
    Conversation,
    RequestTokenInfo,
)


from fluximp_wxwidgets import themes
from fluximp_wxwidgets.api_chart import (
    CHART_SERIES_DEFAULTS,
    CHART_SERIES_KEYS,
    CHART_SERIES_LABELS,
    CHART_SERIES_LINE_ROLES,
    CHART_X_LABEL,
    CHART_Y_LABEL,
    CHART_Y_MAX_FLOOR,
    request_tick_indices,
    y_tick_values,
)
from fluximp_wxwidgets.grid_metadata_sums import (
    GRID_SUM_COLUMNS,
    GRID_SUM_COLUMN_INDEX,
    GRID_SUM_DEFAULTS,
    GRID_SUM_KEYS,
)
from wx.lib.plot.plotcanvas import PlotCanvas
from wx.lib.plot.polyobjects import PlotGraphics, PolyLine


class ForkTurnGrid(wx.grid.Grid):
    """Grid displaying fork/turn status for the current conversation."""

    # Column indices.  Fork and Turn are always shown; the sum columns are
    # toggleable (View > API Grid Sums) and come from the registry.
    COL_FORK = 0
    COL_TURN = 1
    COL_PROMPT = GRID_SUM_COLUMN_INDEX["prompt_tokens"]
    COL_CACHED = GRID_SUM_COLUMN_INDEX["cached_tokens"]
    COL_COMPLETION = GRID_SUM_COLUMN_INDEX["completion_tokens"]
    COL_REASONING = GRID_SUM_COLUMN_INDEX["reasoning_tokens"]
    COL_TIME = GRID_SUM_COLUMN_INDEX["time"]

    def __init__(self, parent):
        super().__init__(parent)

        # Initialize with single row, will be updated dynamically
        self.CreateGrid(1, 7)

        # Which toggleable metadata-sum columns are shown (View > API
        # Grid Sums); the grid hides itself entirely when none are.
        self._columns_visible: dict = dict(GRID_SUM_DEFAULTS)

        self.SetColLabelValue(self.COL_FORK, "Fork")
        self.SetColLabelValue(self.COL_TURN, "Turn")
        self.SetColLabelValue(self.COL_PROMPT, "Prompt")
        self.SetColLabelValue(self.COL_CACHED, "Cached")
        self.SetColLabelValue(self.COL_COMPLETION, "Compl")
        self.SetColLabelValue(self.COL_REASONING, "Reason")
        self.SetColLabelValue(self.COL_TIME, "Time (s)")

        # Set column sizes
        self.SetColSize(self.COL_FORK, 50)
        self.SetColSize(self.COL_TURN, 50)
        self.SetColSize(self.COL_PROMPT, 60)
        self.SetColSize(self.COL_CACHED, 60)
        self.SetColSize(self.COL_COMPLETION, 60)
        self.SetColSize(self.COL_REASONING, 60)
        self.SetColSize(self.COL_TIME, 70)

        # Disable cell editing
        self.EnableEditing(False)

        # Hide row labels
        self.SetRowLabelSize(0)

        # Store the last selected cell to prevent re-selecting
        self._last_selected = None

        self.Bind(wx.grid.EVT_GRID_SELECT_CELL, self._on_cell_selected)

        # Start with every sum column hidden (the registry default), which
        # also hides the whole grid until one is enabled.
        self._apply_column_visibility()

    def set_columns_visible(self, visibility: dict):
        """Show/hide sum columns; keys absent from *visibility* keep state."""
        for key in GRID_SUM_KEYS:
            if key in visibility:
                self._columns_visible[key] = bool(visibility[key])
        self._apply_column_visibility()

    def has_visible_sum_columns(self) -> bool:
        """True when at least one sum column is shown (else the grid hides)."""
        return any(self._columns_visible.get(key) for key in GRID_SUM_KEYS)

    def _apply_column_visibility(self):
        """Show/hide each sum column to match _columns_visible."""
        for key, _label, column, _default in GRID_SUM_COLUMNS:
            if self._columns_visible.get(key):
                self.ShowCol(column)
            else:
                self.HideCol(column)

    def _hide_disabled_columns(self):
        """Re-hide disabled sum columns after AutoSize() resized them."""
        for key, _label, column, _default in GRID_SUM_COLUMNS:
            if not self._columns_visible.get(key):
                self.HideCol(column)

    def _on_cell_selected(self, event):
        """Handle cell selection."""
        row = event.GetRow()
        fork = self.GetCellValue(row, self.COL_FORK)
        turn = self.GetCellValue(row, self.COL_TURN)

        # Skip if values are empty (grid may be clearing/updating)
        if fork.strip() == '' or turn.strip() == '':
            event.Skip()
            return

        # Get parent panel and call its handler
        parent = self.GetParent()
        if hasattr(parent, '_on_grid_selected'):
            parent._on_grid_selected(int(fork), int(turn))

        self._last_selected = (row, event.GetCol())
        event.Skip()

    def update_from_conversation(self, conversation: Optional[Conversation]):
        """Update grid from conversation data."""
        if not conversation:
            self.ClearGrid()
            return

        fork_turns = conversation.get_fork_turns()

        # Resize grid to match data
        current_rows = self.GetNumberRows()
        needed_rows = len(fork_turns)

        if needed_rows > current_rows:
            self.AppendRows(needed_rows - current_rows)
        elif needed_rows < current_rows:
            self.DeleteRows(0, current_rows - needed_rows)

        # Populate grid
        for i, info in enumerate(fork_turns):
            self.SetCellValue(i, self.COL_FORK, str(info.fork))
            self.SetCellValue(i, self.COL_TURN, str(info.turn))
            self.SetCellValue(i, self.COL_PROMPT, str(info.prompt_tokens) if info.prompt_tokens else "")
            self.SetCellValue(i, self.COL_CACHED, str(info.cached_tokens) if info.cached_tokens else "")
            self.SetCellValue(i, self.COL_COMPLETION, str(info.completion_tokens) if info.completion_tokens else "")
            self.SetCellValue(i, self.COL_REASONING, str(info.reasoning_tokens) if info.reasoning_tokens else "")
            self.SetCellValue(i, self.COL_TIME, f"{info.time_seconds:.1f}" if info.time_seconds else "")

        self.AutoSize()
        # AutoSize() may resize every column; re-hide the disabled sum columns
        # so the View > API Grid Sums toggles survive a data refresh.
        self._hide_disabled_columns()


class TokenPlotCanvas(PlotCanvas):
    """PlotCanvas whose axes are laid out by the API chart's tick policy.

    wx.lib.plot always derives its ticks from an axis range plus a "spec"
    (Draw takes no tick list), and its automatic rounding gives a 64k token
    axis 5k steps -- fourteen labels and grid lines.  So both axes are
    generated by api_chart instead, by overriding the two tick generators:
    the Y axis in fixed 10k steps (widened when they would crowd) and the X
    axis at the request positions ForkTokenChart picks, each labelled with
    its request number.

    Neither tick list may be empty, however empty the chart: PlotCanvas._Draw
    reads xticks[-1] and yticks[0]/[-1] to size the plot's margins, so the X
    axis falls back to one label-less tick (see set_request_ticks).
    """

    # The one tick an unlabelled X axis gets, so _Draw's margin maths has
    # something to measure.  Its text is empty and the bottom ticks and values
    # are switched off whenever it is in use, so it paints nothing.
    _NO_TICK = (0.0, '')

    def __init__(self, parent):
        super().__init__(parent, style=wx.NO_BORDER)
        self._x_ticks: List[tuple] = [self._NO_TICK]

    def set_request_ticks(self, count: int):
        """Tick the X axis at the request numbers api_chart picks for *count*.

        With no requests there is nothing to label, so the bottom ticks and
        their values are switched off rather than left to draw one stray tick
        mark in the middle of an empty chart.
        """
        ticks = [(float(index), str(index + 1))
                 for index in request_tick_indices(count)]
        self._x_ticks = ticks or [self._NO_TICK]
        self._set_bottom_display(bool(ticks))

    def _set_bottom_display(self, show: bool):
        """Show or hide the bottom ticks and their values.

        Both setters redraw the plot, so this only touches them when the value
        actually changes -- which is only on the transition between an empty
        and a non-empty chart.
        """
        if (self.enableTicks.bottom == show
                and self.enableAxesValues.bottom == show):
            return
        self.enableTicks = (show, False, False, False)
        self.enableAxesValues = (show, True, False, False)

    # -- the two tick generators wx.lib.plot draws from --------------------
    # Both are handed the axis range about to be drawn (PlotCanvas._Draw) and
    # both ignore the xSpec/ySpec "spec" the base class would consult.

    def _xticks(self, *args):
        return list(self._x_ticks)

    def _yticks(self, lower, upper):
        return [(float(value), str(value))
                for value in y_tick_values(lower, upper)]


class ForkTokenChart(wx.Panel):
    """Per-request token line chart for the displayed conversation.

    One point per API request (conversation_manager.RequestTokenInfo) in
    request order, so the X axis is the implicit request index: it is
    labelled "Request" and ticked at a few request numbers
    (api_chart.request_tick_indices) rather than at every point, and one
    line is drawn per enabled series (api_chart.CHART_SERIES).  The Y axis
    is tokens, labelled in the fixed 10k steps of api_chart.y_tick_values:
    it starts at the smallest plotted value and its top never drops below
    CHART_Y_MAX_FLOOR, rising with the highest value, so a growing run is
    always fully on the chart.
    """

    # Smallest side (px) worth drawing; below this ForkTurnPanel hides it.
    MIN_SIDE = 120

    def __init__(self, parent):
        super().__init__(parent)
        self._requests: List[RequestTokenInfo] = []
        self._series_visible: dict = dict(CHART_SERIES_DEFAULTS)

        sizer = wx.BoxSizer(wx.VERTICAL)
        self.plot = TokenPlotCanvas(self)
        # Antialiasing is a rendering-quality choice, not a theme colour, so
        # it is set here rather than in apply_theme (which is re-run on every
        # theme switch and would only redraw for no reason).  Without it
        # PlotCanvas draws with the raw wx.DC and the 2px series lines come
        # out visibly jagged; with it every line, tick and label is drawn
        # through a wx.GCDC (GDI+) instead, which smooths the edges.
        self.plot.enableAntiAliasing = True
        sizer.Add(self.plot, 1, wx.EXPAND)
        self.SetSizer(sizer)

        self.apply_theme()

    # -- theme -------------------------------------------------------------
    def apply_theme(self):
        """(Re)paint the chart from the active theme, then redraw it.

        Called at construction and again on a theme switch
        (MainFrame._apply_theme).  The chart resolves its colours from the
        theme's own chart roles rather than inheriting the pane's, so it has
        to repaint itself after the pane-wide re-colour walk.
        """
        theme = themes.active()
        bg = themes.to_wx(theme.chart_bg)
        fg = themes.to_wx(theme.chart_fg)
        self.SetBackgroundColour(bg)
        self.plot.SetBackgroundColour(bg)
        self.plot.SetForegroundColour(fg)
        # PlotCanvas clears its offscreen buffer with its own background and
        # blits it onto the inner canvas window, so both must match or a
        # theme switch leaves a differently coloured margin.
        try:
            self.plot.canvas.SetBackgroundColour(bg)
        except Exception:
            pass
        axis_pen = wx.Pen(fg, 1, wx.PENSTYLE_SOLID)
        self.plot.gridPen = wx.Pen(
            themes.to_wx(theme.chart_grid), 1, wx.PENSTYLE_DOT)
        self.plot.axesPen = axis_pen
        self.plot.tickPen = axis_pen
        self.plot.fontSizeAxis = 9
        self.plot.enableLegend = False
        self.plot.enableTitle = False
        self.plot.enableZoom = False
        self.plot.enableDrag = False
        # The vertical grid is off: the X axis is ticked at a few requests
        # (TokenPlotCanvas.set_request_ticks, which also owns whether the
        # bottom ticks and their values are shown at all -- an empty chart
        # has neither).  Vertical lines through the plot would only repeat
        # what those few ticks already say.
        self.plot.enableGrid = (False, True)
        self.plot.xSpec = 'min'
        self._redraw()

    # -- data --------------------------------------------------------------
    def set_series_visible(self, visibility: dict):
        """Show/hide series; keys absent from *visibility* keep their state."""
        for key in CHART_SERIES_KEYS:
            if key in visibility:
                self._series_visible[key] = bool(visibility[key])
        self._redraw()

    def set_requests(self, requests):
        """Feed the chart the conversation's per-request token rows."""
        self._requests = list(requests or [])
        self._redraw()

    def has_visible_series(self) -> bool:
        """True when at least one series is enabled (else the chart hides)."""
        return any(self._series_visible.get(key) for key in CHART_SERIES_KEYS)

    def _redraw(self):
        """Rebuild the plot from the current data, series and theme."""
        theme = themes.active()
        count = len(self._requests)

        lines = []
        values: List[int] = []
        for key in CHART_SERIES_KEYS:
            if not self._series_visible.get(key):
                continue
            points = [(index, getattr(row, key, 0) or 0)
                      for index, row in enumerate(self._requests)]
            values.extend(value for _index, value in points)
            lines.append(PolyLine(
                points,
                colour=themes.to_wx(
                    getattr(theme, CHART_SERIES_LINE_ROLES[key])),
                width=2,
                legend=CHART_SERIES_LABELS[key]))

        # X: one slot per request, index 0..count-1, padded by half a slot so
        # the end points are not clipped; a degenerate range makes Draw a
        # no-op, hence the 0.5 minimum span for zero/one rows.
        x_axis = (-0.5, max(0.5, count - 0.5))
        # Y: min of the plotted values (0 with no data), top at least the
        # floor so early runs keep a stable scale.
        y_min = min(values) if values else 0
        y_max = max([CHART_Y_MAX_FLOOR] + values)
        if y_max <= y_min:
            y_max = y_min + 1

        # Tick the X axis at this run's request numbers before drawing: a
        # tick's position is its label, so it is set from the data.
        self.plot.set_request_ticks(count)

        graphics = PlotGraphics(
            lines, xLabel=CHART_X_LABEL, yLabel=CHART_Y_LABEL)
        self.plot.Draw(graphics, xAxis=x_axis, yAxis=(y_min, y_max))


class ForkTurnPanel(wx.Panel):
    """Panel holding the per-request token chart above the fork/turn grid."""

    # Padding around the chart, and the smallest grid height kept when
    # deciding whether the chart still fits vertically.
    CHART_MARGIN = 5
    GRID_MIN_HEIGHT = 80

    def __init__(self, parent):
        super().__init__(parent)

        sizer = wx.BoxSizer(wx.VERTICAL)

        self.info_label = wx.StaticText(self, label="No workflow selected")
        sizer.Add(self.info_label, 0, wx.EXPAND | wx.ALL, 5)

        # The chart sits above the grid: square, sized to the panel width by
        # _relayout_chart (which also hides it when it cannot fit).
        self.chart = ForkTokenChart(self)
        # A [stretch, chart, stretch] row rather than a bare sizer item: the
        # chart is square, so its width has to track its height (and the panel
        # width) instead of being EXPANDed to the panel's full width.  A
        # hidden chart collapses the row to zero height, so no space is
        # reserved when the chart is switched off.
        chart_row = wx.BoxSizer(wx.HORIZONTAL)
        chart_row.AddStretchSpacer(1)
        chart_row.Add(self.chart, 0)
        chart_row.AddStretchSpacer(1)
        sizer.Add(chart_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM,
                  self.CHART_MARGIN)

        self.grid = ForkTurnGrid(self)
        sizer.Add(self.grid, 1, wx.EXPAND)

        self.SetSizer(sizer)

        self.conversation: Optional[Conversation] = None
        self.on_turn_selected: Optional[Callable[[int, int], None]] = None

        self._chart_side = 0
        self._chart_shown = True
        # Whether the grid is currently shown (it hides when every sum column
        # is off -- View > API Grid Sums).
        self._grid_shown = True
        self.Bind(wx.EVT_SIZE, self._on_size)

    def _on_size(self, event):
        self._relayout_chart()
        event.Skip()

    def _relayout_chart(self):
        """Keep the chart square and width-driven, or hide it when it can't fit.

        The chart is not sizer-proportioned: its height tracks its width so it
        stays square as the panel widens.  It is dropped entirely when either
        no series is enabled (the View > API Chart toggles) or the panel is too
        short to leave the grid a usable height -- a rare edge case.

        The grid below is shown only while at least one metadata-sum column is
        enabled (View > API Grid Sums); with none it collapses, and the
        chart may then use the height its reserve would have taken.
        """
        grid_show = self.grid.has_visible_sum_columns()
        show = self.chart.has_visible_series()
        side = 0
        if show:
            size = self.GetClientSize()
            width = size.width - 2 * self.CHART_MARGIN
            reserve = self.GRID_MIN_HEIGHT if grid_show else 0
            available = (size.height - self.info_label.GetSize().height
                         - 2 * self.CHART_MARGIN - reserve)
            side = min(width, available)
            if side < ForkTokenChart.MIN_SIDE:
                show = False
                side = 0
        if (show == self._chart_shown and side == self._chart_side
                and grid_show == self._grid_shown):
            return
        self._chart_shown = show
        self._chart_side = side
        self._grid_shown = grid_show
        if show:
            self.chart.SetMinSize((side, side))
        self.chart.Show(show)
        self.grid.Show(grid_show)
        self.Layout()

    def set_series_visible(self, visibility: dict):
        """Apply the View > API Chart series toggles (chart hides when none)."""
        self.chart.set_series_visible(visibility)
        self._relayout_chart()

    def set_grid_columns_visible(self, visibility: dict):
        """Apply the View > API Grid Sums toggles (grid hides when none)."""
        self.grid.set_columns_visible(visibility)
        self._relayout_chart()

    def _on_grid_selected(self, fork: int, turn: int):
        """Handle grid cell selection."""
        if self.on_turn_selected:
            self.on_turn_selected(fork, turn)

    def set_conversation(self, conversation: Optional[Conversation]):
        """Set the conversation to display.

        No per-conversation callback binding: the displayed tab forwards
        grid refreshes here (RightPanel.refresh_fork_grid) instead — a
        background tab's facts never reach the shared grid.
        """
        self.conversation = conversation
        self.grid.update_from_conversation(conversation)
        self._update_chart()
        self._update_info()

    def _update_grid(self):
        """Update the grid in the main thread."""
        self.grid.update_from_conversation(self.conversation)
        self._update_chart()
        self._update_info()

    def _update_chart(self):
        """Feed the chart the displayed conversation's per-request rows.

        The rows are per API request (Conversation.get_request_tokens), not
        the grid's per-turn sums, so a turn holding several requests shows as
        several points.
        """
        requests = (self.conversation.get_request_tokens()
                    if self.conversation else [])
        self.chart.set_requests(requests)

    def _update_info(self):
        """Update the info label."""
        if self.conversation:
            self.info_label.SetLabel(
                f"{self.conversation.workflow_name}: {self.conversation.conversation_id}\n"
                f"Forks: {self.conversation.fork_turn_state.active_fork}, "
                f"Turns: {self.conversation.fork_turn_state.active_turn}"
            )
        else:
            self.info_label.SetLabel("No workflow selected")


class RightPanel(wx.Panel):
    """Right panel containing the fork/turn grid."""

    # Agent types available in the GUI (scrutinator is internal use only)
    AGENT_TYPES = sorted(fs_agents.AGENT_DEFINITIONS.keys())  # default, overridden by config_mgr

    # Default agent type to select
    DEFAULT_AGENT_TYPE = 'consultant'

    def __init__(self, parent, config_mgr: WorkflowConfigManager = None,
                 app_settings: Optional[AppSettings] = None):
        super().__init__(parent)
        self.config_mgr = config_mgr
        self.app_settings = app_settings
        if config_mgr:
            self.agent_types = sorted(config_mgr.get_workflow_agent_names(STOCK_WORKFLOW_NAME))
        else:
            self.agent_types = sorted(fs_agents.AGENT_DEFINITIONS.keys())
        # Assert scrutinator is not in the GUI agent types
        assert 'scrutinator' not in self.agent_types, "scrutinator is for internal use only"

        sizer = wx.BoxSizer(wx.VERTICAL)
        self.fork_turn_panel = ForkTurnPanel(self)
        sizer.Add(self.fork_turn_panel, 1, wx.EXPAND)

        # Apply the persisted View > API Chart series toggles to the chart.
        self.fork_turn_panel.set_series_visible(self.chart_series_visibility())

        # Apply the persisted View > API Grid Sums column toggles to the
        # grid (every column off by default, so the grid starts hidden).
        self.fork_turn_panel.set_grid_columns_visible(
            self.grid_sum_visibility())

        # Config selector (Agent, Provider, Model) - labels above dropdowns
        # Use a grid sizer: 2 rows (labels, then dropdowns), 3 columns
        config_sizer = wx.FlexGridSizer(2, 3, 5, 10)  # 2 rows, 3 cols, vgap=5, hgap=10
        config_sizer.AddGrowableCol(2)  # Model column grows

        # Row 0: Labels
        config_sizer.Add(wx.StaticText(self, label="Agent:"), 0, wx.ALIGN_LEFT)
        config_sizer.Add(wx.StaticText(self, label="Provider:"), 0, wx.ALIGN_LEFT)
        config_sizer.Add(wx.StaticText(self, label="Model:"), 0, wx.ALIGN_LEFT)

        # Row 1: Dropdowns
        self.agent_combo = wx.ComboBox(self, choices=self.agent_types, style=wx.CB_READONLY, size=(90, -1))
        default_agent_idx = self.agent_types.index(self.DEFAULT_AGENT_TYPE)
        self.agent_combo.SetSelection(default_agent_idx)
        self.agent_combo.Bind(wx.EVT_COMBOBOX, self._on_agent_changed)
        self.agent_combo.Enable(False)
        config_sizer.Add(self.agent_combo, 0, wx.EXPAND)

        # Providers that have an API key set (set at startup via
        # set_providers_with_keys); provider_names is the curated-filtered
        # subset actually offered in the combo.
        self.providers_with_keys: List[str] = []
        self.provider_names = get_provider_names()
        self.provider_combo = wx.ComboBox(self, choices=self.provider_names, style=wx.CB_READONLY, size=(110, -1))
        self.provider_combo.Bind(wx.EVT_COMBOBOX, self._on_provider_changed)
        self.provider_combo.Enable(False)
        config_sizer.Add(self.provider_combo, 0, wx.EXPAND)

        self.model_combo = wx.ComboBox(self, choices=[], style=wx.CB_READONLY, size=(200, -1))
        self.model_combo.Bind(wx.EVT_COMBOBOX, self._on_model_changed)
        self.model_combo.Enable(False)
        config_sizer.Add(self.model_combo, 1, wx.EXPAND)

        # Log path row (agent stdout log path + copy button) - above the combo boxes
        log_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.log_path_label = wx.StaticText(self, label="", style=wx.ST_ELLIPSIZE_MIDDLE)
        self.log_path_label.SetForegroundColour(themes.to_wx(themes.active().subtle_fg))
        self.log_copy_button = wx.Button(self, label="Copy", size=(70, -1))
        self.log_copy_button.Bind(wx.EVT_BUTTON, self._on_copy_log_path)
        self.log_copy_button.Enable(False)
        log_sizer.Add(self.log_path_label, 1, wx.ALIGN_CENTER_VERTICAL)
        log_sizer.Add(self.log_copy_button, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 5)
        sizer.Add(log_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)
        sizer.Add(config_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Status indicator label
        self.status_label = wx.StaticText(self, label="")
        self.status_label.SetForegroundColour(themes.to_wx(themes.active().status_stopped_fg))
        sizer.Add(self.status_label, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        self.SetSizer(sizer)

        self.current_conversation: Optional[Conversation] = None
        # Owning ConversationTab of the displayed conversation (main_frame
        # passes it on open/switch; the displayed tab forwards live state —
        # the panel binds nothing).
        self.current_tab = None

    def set_providers_with_keys(self, provider_names: List[str]):
        """Record the providers that have an API key set, then apply filtering.

        Called at startup with the keyring-scanned list.  The raw list is kept
        so the curated filter can be re-applied later (App Settings toggle)
        without another keyring scan.
        """
        self.providers_with_keys = list(provider_names)
        self.apply_curated_filter()

    def has_providers_with_curated_filter(self) -> bool:
        """True when the curated filter still leaves at least one keyed provider.

        The App Settings toggle and the startup gate use this to decide
        whether enabling the filter would leave the GUI with no usable
        provider.
        """
        return bool(filter_provider_names(self.providers_with_keys, True))

    def apply_curated_filter(self):
        """Recompute the Provider combo from the curated App Settings value.

        The displayed providers are the keyed providers
        (self.providers_with_keys), further narrowed to the curated set when
        the App Settings checkbox is on.  self.provider_names is the index
        source used by set_conversation/set_current_provider, so a provider
        filtered out here can no longer be set.  The displayed conversation
        is re-read (full set_conversation) so a live toggle takes effect
        immediately.
        """
        limit = (self.app_settings is not None
                 and self.app_settings.limit_providers_and_models_to_curated)
        provider_names = filter_provider_names(self.providers_with_keys, limit)
        # Defensive: never leave the combo with no providers.  Startup and the
        # App Settings toggle both gate this, but a stray call must not brick
        # the panel.
        if not provider_names:
            provider_names = list(self.providers_with_keys)

        current = self.provider_combo.GetValue()
        self.provider_names = provider_names
        self.provider_combo.SetItems(self.provider_names)
        if current in self.provider_names:
            self.provider_combo.SetSelection(self.provider_names.index(current))
        elif self.provider_names:
            self.provider_combo.SetSelection(0)

        if self.current_conversation is not None:
            self.set_conversation(self.current_conversation, tab=self.current_tab)

    def set_current_provider(self, conversation: Conversation, provider_name: str):
        """THE single authority for setting Conversation.current_provider.

        The provider must be one that has an API key set (self.provider_names).
        If the incoming provider isn't allowed but the conversation already
        holds an allowed provider, keep it; otherwise fall back to the first
        allowed provider.
        """
        if provider_name in self.provider_names:
            conversation.current_provider = provider_name
        elif conversation.current_provider not in self.provider_names:
            if self.provider_names:
                conversation.current_provider = self.provider_names[0]

    def _update_model_choices(self, provider_name: str, select_model: Optional[str] = None):
        """Update model dropdown with models for the given provider.

        When the App Settings curated filter is on, the provider's models are
        narrowed to its curated set (filter_model_names); the stored model is
        then either found in that list or the selection falls back to the
        provider default / first entry.
        """
        provider = create_provider(provider_name)
        models = provider.get_model_names()
        if (self.app_settings is not None
                and self.app_settings.limit_providers_and_models_to_curated):
            models = filter_model_names(provider_name, models, True)
        self.model_combo.SetItems(models)

        if select_model and select_model in models:
            self.model_combo.SetSelection(models.index(select_model))
        elif models:
            # Default to the provider's default model
            default = provider.get_default_model()
            if default in models:
                self.model_combo.SetSelection(models.index(default))
            else:
                self.model_combo.SetSelection(0)

    def _update_status_label(self):
        """Update the status indicator label.

        Aliveness is the attached Popen OR the io dir's agent lock
        (process_lock.py): an unattached-but-alive orphaned agent (a
        previous GUI session's process) shows as running-unattached — its
        records stream through the watcher's held-lock arm and Send
        absorbs it instead of double-spawning. A probe failure
        ('unknown') degrades to the plain stopped label, never a false
        dead. The stopped text's "will restart on next send" promise is
        shown only when a control actually provides the restart — the
        displayed tab's Send enabled or Resume visible
        (get_restart_available); a stopped non-resumable workflow shows
        the plain stopped form and promises nothing (§7.8).
        """
        if not self.current_conversation:
            self.status_label.SetLabel("")
            return

        process = self.current_conversation.process
        if process is not None and process.poll() is None:
            self.status_label.SetForegroundColour(themes.to_wx(themes.active().status_running_fg))
            self.status_label.SetLabel(f"✓ Process running (PID: {process.pid})")
            return
        if process is not None:
            # Process has exited - clear it and show stopped status
            self.current_conversation.process = None
        from fluximp_agent.process_lock import probe
        try:
            verdict = probe(self.current_conversation.io_dir, 'agent')
        except Exception:
            verdict = 'unknown'
        if verdict == 'held':
            # Unattached agent owns the io dir: alive, absorbed, and being
            # scanned by the watcher's held-lock arm.
            self.status_label.SetForegroundColour(themes.to_wx(themes.active().status_running_fg))
            self.status_label.SetLabel("✓ Agent running (unattached)")
            return
        self.status_label.SetForegroundColour(themes.to_wx(themes.active().status_stopped_fg))
        # Stopped: the "will restart on next send" promise only when a
        # control actually provides it — Send enabled (a send respawns
        # through _ensure_process_running) or Resume visible (the resume
        # restarts). A stopped non-resumable workflow has neither (§7.8)
        # — plain stopped, no promise.
        restart_available = False
        tab = self.current_tab
        if tab is not None:
            restart_available = tab.get_restart_available()
        if restart_available:
            self.status_label.SetLabel(
                "\u26a0 Process stopped - will restart on next send")
        else:
            self.status_label.SetLabel("\u26a0 Process stopped")

    def enable_config_controls(self, enabled: bool):
        """Enable or disable the config dropdowns.

        One owner for the combos' enabled state now: the displayed tab's
        config gate — Send's idle gate OR unlocked whenever no process of
        any kind is running (the recovery unlock for a mid-turn process
        death) — applied at tab sync (set_conversation) and forwarded live
        by the displayed tab (apply_displayed_tab_state,
        note_process_starting). Never by background-tab callbacks: the
        panel dropped all per-conversation bindings.
        """
        self.agent_combo.Enable(enabled)
        self.provider_combo.Enable(enabled)
        self.model_combo.Enable(enabled)
        # Force immediate visual update on each control
        self.agent_combo.Update()
        self.provider_combo.Update()
        self.model_combo.Update()

    def apply_displayed_tab_state(self, config_enabled: bool):
        """Forward from the displayed tab: apply its config lock state.

        The forwarded bool is the Agent/Provider/Model combos' gate —
        Send's idle gate OR unlocked whenever no process of any kind is
        running (conversation_tab._config_gate_enabled) — no longer the
        input/Send lock itself, so a mid-turn process death can never
        deadlock the Provider/Model switch that recovers it.

        Replaces the old per-conversation on_agent_active_changed binding,
        which carried no tab identity — a background tab's transition
        disabled these combos for whichever tab was displayed, and only a
        switch away/back resynced. The forwarding tab's recompute is the
        sole traffic now; the status refresh keeps the old handler's
        live-display behavior.
        """
        self.enable_config_controls(config_enabled)
        self._update_status_label()

    def _on_agent_changed(self, event: wx.Event):
        """Handle agent type selection change - terminate process immediately."""
        if not self.current_conversation:
            return

        new_agent_type = self.agent_combo.GetValue()
        self.current_conversation.agent_type = new_agent_type
        # Update agent_configs with config paths for custom/overridden agents in the workflow
        if self.config_mgr:
            config_paths = self.config_mgr.get_conversation_agent_configs(
                self.current_conversation.workflow_name, new_agent_type)
            self.current_conversation.agent_configs = [str(p) for p in config_paths]
        self.current_conversation._save_config()

        # Terminate process if running (will restart with new config on next send)
        if self.current_conversation.process is not None:
            self.current_conversation.terminate()

        self._update_status_label()

    def _on_provider_changed(self, event: wx.Event):
        """Handle provider selection change - terminate process immediately."""
        if not self.current_conversation:
            return

        self.set_current_provider(
            self.current_conversation, self.provider_combo.GetValue())
        new_provider = self.current_conversation.current_provider
        # Keep the combo in sync in case the setter kept the previous
        # provider instead of the incoming one.
        if new_provider in self.provider_names:
            self.provider_combo.SetSelection(self.provider_names.index(new_provider))

        # Get the model to use for this provider
        stored_model = self.current_conversation.provider_models.get(new_provider)
        if stored_model:
            self._update_model_choices(new_provider, select_model=stored_model)
        else:
            self._update_model_choices(new_provider)

        # Update current model to selected one
        self.current_conversation.current_model = self.model_combo.GetValue()
        self.current_conversation._save_config()

        # Terminate process if running (will restart with new config on next send)
        if self.current_conversation.process is not None:
            self.current_conversation.terminate()

        self._update_status_label()

    def _on_model_changed(self, event: wx.Event):
        """Handle model selection change - terminate process immediately."""
        if not self.current_conversation:
            return

        new_model = self.model_combo.GetValue()
        self.current_conversation.current_model = new_model
        self.current_conversation.provider_models[self.current_conversation.current_provider] = new_model
        self.current_conversation._save_config()

        # Terminate process if running (will restart with new config on next send)
        if self.current_conversation.process is not None:
            self.current_conversation.terminate()

        self._update_status_label()

    def _update_log_path(self):
        """Update the agent stdout log path label and copy button state."""
        if self.current_conversation and self.current_conversation.io_dir:
            path = str(self.current_conversation.io_dir.resolve() / "agent_stdout.log")
            self.log_path_label.SetLabel(path)
            self.log_path_label.SetToolTip(path)
            self.log_copy_button.Enable(True)
        else:
            self.log_path_label.SetLabel("")
            self.log_path_label.SetToolTip("")
            self.log_copy_button.Enable(False)
        # Re-layout so the ellipsized label resizes correctly to fit between
        # the left edge and the Copy button (otherwise it overlaps the button
        # until the window is manually resized).
        wx.CallAfter(self.Layout)

    def _on_copy_log_path(self, event: wx.Event):
        """Copy the agent stdout log path to the clipboard."""
        if not self.current_conversation or not self.current_conversation.io_dir:
            return
        path = str(self.current_conversation.io_dir.resolve() / "agent_stdout.log")
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject(path))
            wx.TheClipboard.Close()
            self.status_label.SetForegroundColour(themes.to_wx(themes.active().status_running_fg))
            self.status_label.SetLabel("✓ Log path copied to clipboard")

    def note_process_starting(self):
        """Displayed-tab forward: a process start is beginning.

        The combos lock with the input/Send pair through the starting
        window; the stamp is live-only state (nothing on the conversation
        stores it), so only this forward can show it.
        """
        self.enable_config_controls(False)
        self.status_label.SetLabel("⏳ Starting process...")

    def note_process_attached(self):
        """Displayed-tab forward: a process was attached — show running status."""
        convo = self.current_conversation
        if convo is not None and convo.process is not None:
            self.status_label.SetForegroundColour(themes.to_wx(themes.active().status_running_fg))
            pid = convo.process.pid
            self.status_label.SetLabel(f"✓ Process running (PID: {pid})")

    def set_conversation(self, conversation: Optional[Conversation], tab=None):
        """Set the conversation (and its owning tab) to display.

        The panel holds no per-conversation callbacks — they carried no
        tab identity, so background tabs' agent/process transitions used to
        overwrite the combos and status of whichever tab was displayed, and
        switching away/back (this method) was the only resync. Live updates
        now arrive only as forwards from the displayed tab
        (apply_displayed_tab_state / note_process_starting /
        note_process_attached / refresh_fork_grid), each gated by its own
        identity check, so this method stays the full durable re-read for
        open/switch.
        """
        self.current_conversation = conversation
        self.current_tab = tab
        self.fork_turn_panel.set_conversation(conversation)

        if conversation:
            # Update agent combo with agents from this conversation's workflow
            if self.config_mgr:
                self.agent_types = sorted(self.config_mgr.get_workflow_agent_names(conversation.workflow_name))
                self.agent_combo.Clear()
                for name in self.agent_types:
                    self.agent_combo.Append(name)
                # Ensure the agent_type is valid for this workflow; if not,
                # correct it to the first available agent in the workflow
                if conversation.agent_type not in self.agent_types and self.agent_types:
                    conversation.agent_type = self.agent_types[0]
                # Update agent_configs with config paths for custom/overridden agents in the workflow
                config_paths = self.config_mgr.get_conversation_agent_configs(
                    conversation.workflow_name, conversation.agent_type)
                conversation.agent_configs = [str(p) for p in config_paths]
            try:
                idx = self.agent_types.index(conversation.agent_type)
                self.agent_combo.SetSelection(idx)
            except ValueError:
                if self.agent_types:
                    self.agent_combo.SetSelection(0)
            # Update provider combo (reconcile the conversation's provider
            # against the providers that have an API key set)
            previous_provider = conversation.current_provider
            self.set_current_provider(conversation, conversation.current_provider)
            provider_reconciled = conversation.current_provider != previous_provider
            if conversation.current_provider in self.provider_names:
                self.provider_combo.SetSelection(
                    self.provider_names.index(conversation.current_provider))
            elif self.provider_names:
                self.provider_combo.SetSelection(0)

            # Update model choices based on selected provider (use stored model if available)
            selected_provider = self.provider_combo.GetValue()
            self._update_model_choices(selected_provider, select_model=conversation.current_model)

            # The model combo is authoritative: adopt the displayed model when
            # the provider was reconciled (the stored model belonged to the old
            # provider) or when a stored model isn't among the offered
            # (possibly curated-filtered) choices, so the conversation and the
            # combo never diverge.  An empty stored model is left alone so the
            # history load can restore it from the conversation's metadata
            # (conversation_tab.py).
            if (provider_reconciled
                    or (conversation.current_model
                        and conversation.current_model != self.model_combo.GetValue())):
                conversation.current_model = self.model_combo.GetValue()
                conversation._save_config()

            # Update status label
            self._update_status_label()

            # Combos follow the displayed tab's config gate (relaxed §2.2):
            # ask the tab — its gate folds Send's idle gate with the
            # no-process-running unlock. A mid-turn process death leaves the
            # turn's io facts non-terminal forever (no one lives to write the
            # terminal record), and the Provider/Model switch that is the
            # recovery for that state must never be deadlocked behind it.
            # The conversation's is_agent_active (tab-less fallback below)
            # omits those gates and is stale during live conditional runs.
            self.enable_config_controls(
                self.current_tab.get_config_enabled()
                if self.current_tab is not None
                else not conversation.is_agent_active)
            self._update_log_path()
        else:
            self.agent_combo.Enable(False)
            self.provider_combo.Enable(False)
            self.model_combo.Enable(False)
            self.status_label.SetLabel("")
            self._update_log_path()

    def refresh_fork_grid(self):
        """Displayed-tab forward: rebuild the fork/turn grid now.

        _update_grid re-reads the displayed conversation, so the rebuild is
        always for whichever tab the panel is showing.
        """
        self.fork_turn_panel._update_grid()

    def chart_series_visibility(self) -> dict:
        """The persisted per-series API chart visibility (View > API Chart)."""
        if self.app_settings is None:
            return dict(CHART_SERIES_DEFAULTS)
        return {key: self.app_settings.is_chart_series_visible(key)
                for key in CHART_SERIES_KEYS}

    def apply_chart_series_visibility(self):
        """Push the persisted View > API Chart toggles to the chart."""
        self.fork_turn_panel.set_series_visible(self.chart_series_visibility())

    def grid_sum_visibility(self) -> dict:
        """The persisted per-column grid sum visibility (View > API Grid Sums)."""
        if self.app_settings is None:
            return dict(GRID_SUM_DEFAULTS)
        return {key: self.app_settings.is_grid_sum_visible(key)
                for key in GRID_SUM_KEYS}

    def apply_grid_sum_visibility(self):
        """Push the persisted View > API Grid Sums toggles to the grid."""
        self.fork_turn_panel.set_grid_columns_visible(
            self.grid_sum_visibility())

    def set_callbacks(self, on_turn_selected=None):
        """Set callback functions."""
        self.fork_turn_panel.on_turn_selected = on_turn_selected

    def update_agent_types(self, workflow_name: str = None):
        """Refresh the agent dropdown with agents from the specified workflow.

        If the current agent is no longer in the workflow, corrects it to the
        first available agent and updates agent_configs accordingly.
        """
        if not self.config_mgr:
            return
        if workflow_name is None and self.current_conversation:
            workflow_name = self.current_conversation.workflow_name
        if workflow_name is None:
            workflow_name = STOCK_WORKFLOW_NAME
        self.agent_types = sorted(self.config_mgr.get_workflow_agent_names(workflow_name))
        # Preserve current selection if it's still valid for this workflow
        current = self.agent_combo.GetValue()
        self.agent_combo.Clear()
        for name in self.agent_types:
            self.agent_combo.Append(name)
        if current in self.agent_types:
            self.agent_combo.SetValue(current)
        elif self.agent_types:
            self.agent_combo.SetSelection(0)
            # Current agent is no longer in this workflow - correct conversation.agent_type
            if self.current_conversation:
                self.current_conversation.agent_type = self.agent_types[0]
                config_paths = self.config_mgr.get_conversation_agent_configs(
                    workflow_name, self.current_conversation.agent_type)
                self.current_conversation.agent_configs = [str(p) for p in config_paths]
                self.current_conversation._save_config()
