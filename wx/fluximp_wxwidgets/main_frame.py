"""Main frame with AUI layout and menu bar."""

import wx
import wx.aui
import logging
from pathlib import Path
from typing import Optional, Callable
import json

from fluximp_wxwidgets.workspace import WorkspaceManager, Workspace, ConversationStub
from fluximp_wxwidgets.conversation_manager import Conversation
from fluximp_wxwidgets.file_watcher import FileWatcher
from fluximp_wxwidgets.left_panel import LeftPanel
from fluximp_wxwidgets.right_panel import RightPanel
from fluximp_wxwidgets.conversation_tab import ConversationTab
from fluximp_wxwidgets.tab_status_icons import get_status_icon
from fluximp_wxwidgets.workflow_config import WorkflowConfigManager, STOCK_WORKFLOW_NAME
from fluximp_wxwidgets.agent_tab import AgentTab
from fluximp_wxwidgets.event_tab import EventTab
from fluximp_wxwidgets.workspace_settings_tab import WorkspaceSettingsTab
from fluximp_wxwidgets.app_settings import AppSettings
from fluximp_wxwidgets.app_settings_tab import AppSettingsTab, NoCuratedProvidersDialog
from fluximp_wxwidgets.api_metadata_display import API_METADATA_ITEMS
from fluximp_wxwidgets.api_chart import CHART_SERIES
from fluximp_wxwidgets.grid_metadata_sums import GRID_SUM_COLUMNS
from fluximp_wxwidgets import themes

logger = logging.getLogger(__name__)

# The wxAUI dock-art colour ids are read from the wx.aui constants by name in
# MainFrame._apply_aui_theme; they are deliberately not hard-coded here.  The
# enum's first five entries are *sizes* -- sash, caption, gripper, pane border,
# pane button -- so the colours start at 5 and run BACKGROUND, SASH,
# ACTIVE_CAPTION, ACTIVE_CAPTION_GRADIENT, INACTIVE_CAPTION,
# INACTIVE_CAPTION_GRADIENT, ACTIVE_CAPTION_TEXT, INACTIVE_CAPTION_TEXT,
# BORDER, GRIPPER (ids 5..14).  Assuming 0..10 from the roles' positions is how
# an earlier cut repainted the pane captions with the body text colour and left
# the rest of the window untouched.


def _theme_window_tree(window, bg, fg):
    """Recursively apply *bg*/*fg* to *window* and every descendant.

    Native controls paint their own background and do not inherit their
    parent's, so a pane's colours only reach the controls inside it if each
    one is set explicitly.  Icon bitmaps are unaffected: they are images
    with baked-in colours, not a control colour, which is exactly why the
    tree icons keep their fixed palette.

    Buttons are deliberately skipped.  Giving a button an explicit background
    makes the platform draw it itself, which loses the native themed
    appearance -- including the hover highlight (a shade of blue on Windows).
    A theme that set it would look *less* native than the unthemed GUI, and
    the loss is not recoverable by setting the colour back.  The native
    appearance (see apply_appearance) is what darkens them where dark mode
    reaches them.
    """
    pending = [window]
    while pending:
        current = pending.pop()
        if not isinstance(current, wx.Button):
            try:
                current.SetBackgroundColour(bg)
                current.SetForegroundColour(fg)
                current.Refresh()
            except Exception:
                # A control that refuses a colour, or a window destroyed
                # mid-teardown, must not abort the rest of the walk.
                pass
        try:
            pending.extend(current.GetChildren())
        except Exception:
            pass


def _capture_widget_colours(window) -> dict:
    """Record each widget's own colours, before any theme paints them.

    Keyed by the widget itself, because the point is to put each one back
    individually: Troop has to reproduce a pane whose background is the
    platform's dialog grey *and* trees and a grid inside it that are white,
    and a single colour cannot express that.

    Buttons are skipped for the same reason _theme_window_tree skips them: an
    explicit background costs a button its native appearance (including the
    hover highlight), so they must never be painted -- not even back to a
    value we recorded ourselves.
    """
    snapshot: dict = {}
    pending = [window]
    while pending:
        current = pending.pop()
        if not isinstance(current, wx.Button):
            try:
                snapshot[current] = (current.GetBackgroundColour(),
                                     current.GetForegroundColour())
            except Exception:
                # A control that will not report a colour, or a window
                # destroyed mid-teardown, must not abort the walk.
                pass
        try:
            pending.extend(current.GetChildren())
        except Exception:
            pass
    return snapshot


def _restore_widget_colours(window, snapshot) -> None:
    """Replay the colours recorded by _capture_widget_colours.

    A widget with no record is left alone: it did not exist when the snapshot
    was taken, so its colours did not come from the platform and there is
    nothing to restore.
    """
    pending = [window]
    while pending:
        current = pending.pop()
        colours = snapshot.get(current)
        if colours is not None:
            try:
                current.SetBackgroundColour(colours[0])
                current.SetForegroundColour(colours[1])
                current.Refresh()
            except Exception:
                pass
        try:
            pending.extend(current.GetChildren())
        except Exception:
            pass


def _clear_notebook_backgrounds(window) -> list:
    """Drop any explicit background from notebooks and their pages.

    A wx.Notebook's tab strip is drawn natively, so its colour can be neither
    read nor set -- but its page area is made of wx widgets this code paints.
    When the colour painted on a page is not the one an untouched page would
    use, the page area and the tab strip disagree, which is exactly the
    reported symptom: the little tab was right while the page area behind it
    was a grey that matched neither.

    Clearing the explicit colour is how wx spells "no background of my own" --
    an invalid colour clears the flag rather than painting black -- so the
    widget falls back to the platform default and the page area matches the tab
    strip by construction, on every platform, without this code needing to know
    what either colour is.

    Returns a description of each widget it cleared, for the log: if the page
    area is still wrong afterwards, that value is what was painted before.
    """
    cleared = []
    pending = [window]
    while pending:
        current = pending.pop()
        if isinstance(current, wx.Notebook):
            targets = [current]
            try:
                targets += [current.GetPage(i)
                            for i in range(current.GetPageCount())]
            except Exception:
                pass
            for target in targets:
                try:
                    was = target.GetBackgroundColour().GetAsString(
                        wx.C2S_HTML_SYNTAX)
                    target.SetBackgroundColour(wx.NullColour)
                    target.Refresh()
                    cleared.append(f"{target.__class__.__name__}={was}")
                except Exception:
                    pass
        try:
            pending.extend(current.GetChildren())
        except Exception:
            pass
    return cleared


def _theme_grid(grid, *, cell_bg, cell_fg, label_bg, label_fg, line, widget_bg):
    """Paint a wx.grid.Grid's cells, labels and grid lines.

    Every colour is a parameter because this serves two cases that differ in
    all of them: applying a theme, and replaying the colours the platform gave
    the grid (Troop).  The cells are deliberately not the pane's
    background colour -- on Windows they are white while the pane is grey, and
    painting the pane's grey onto them is what made the grid grey.

    Cells are set explicitly as well as by default: a cell keeps the colours
    it was created with, so the defaults alone would only cover rows appended
    after this runs.

    The grid's own *highlighting* is deliberately left alone.  wxGrid paints
    the current cell, that cell's column label and any selected column with
    colours of its own -- the system highlight by default -- and wxWidgets 3.3
    highlights the current column's label even when nothing is selected.
    There is no API to switch that off (it is deliberate: Excel and Calc
    behave the same way), and suppressing it would mean either painting the
    labels ourselves or SetUseNativeColLabels, both of which cost more than
    the cosmetic difference is worth.  So the "Fork" column label -- column 0,
    where the grid cursor sits once the grid takes focus -- can show a
    different colour from the rest, and resets on restart.
    """
    if grid is None:
        return
    try:
        grid.SetDefaultCellBackgroundColour(cell_bg)
        grid.SetDefaultCellTextColour(cell_fg)
        grid.SetLabelBackgroundColour(label_bg)
        grid.SetLabelTextColour(label_fg)
        grid.SetGridLineColour(line)
        grid.SetBackgroundColour(widget_bg)
        for row in range(grid.GetNumberRows()):
            for col in range(grid.GetNumberCols()):
                grid.SetCellBackgroundColour(row, col, cell_bg)
                grid.SetCellTextColour(row, col, cell_fg)
    except Exception as exc:
        logger.warning("Theme: grid re-colour failed: %s", exc)

    try:
        grid.ForceRefresh()
    except Exception as exc:
        logger.warning("Theme: grid refresh failed: %s", exc)



class NewWorkflowDialog(wx.Dialog):
    """Dialog for creating a new workflow."""

    def __init__(self, parent, workspaces: list, config_mgr: WorkflowConfigManager,
                 preselected_workspace=None):
        super().__init__(parent, title="New Workflow", size=(400, 180))

        self.workspaces = workspaces

        sizer = wx.BoxSizer(wx.VERTICAL)

        # Workspace selection
        ws_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ws_sizer.Add(wx.StaticText(self, label="Workspace:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        self.workspace_combo = wx.ComboBox(
            self,
            choices=[ws.name for ws in workspaces],
            style=wx.CB_READONLY
        )
        if preselected_workspace:
            idx = next((i for i, ws in enumerate(workspaces) if ws == preselected_workspace), 0)
            self.workspace_combo.SetSelection(idx)
        elif workspaces:
            self.workspace_combo.SetSelection(0)
        ws_sizer.Add(self.workspace_combo, 1, wx.EXPAND)
        sizer.Add(ws_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # Workflow type selection - populate from config manager
        workflow_names = [wf.name for wf in config_mgr.get_workflows()]
        type_sizer = wx.BoxSizer(wx.HORIZONTAL)
        type_sizer.Add(wx.StaticText(self, label="Workflow Type:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        self.type_combo = wx.ComboBox(
            self,
            choices=workflow_names,
            style=wx.CB_READONLY
        )
        if workflow_names:
            self.type_combo.SetSelection(0)
        type_sizer.Add(self.type_combo, 1, wx.EXPAND)
        sizer.Add(type_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Buttons
        btn_sizer = wx.StdDialogButtonSizer()
        btn_sizer.AddButton(wx.Button(self, wx.ID_OK))
        btn_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
        btn_sizer.Realize()
        sizer.Add(btn_sizer, 0, wx.ALIGN_RIGHT | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        self.SetSizer(sizer)

        # Disable OK if no workspaces
        if not workspaces:
            self.FindWindowById(wx.ID_OK).Enable(False)

    def get_selected_workspace(self):
        """Get the selected workspace object."""
        idx = self.workspace_combo.GetSelection()
        if idx >= 0 and idx < len(self.workspaces):
            return self.workspaces[idx]
        return None

    def get_selected_workflow_type(self):
        """Get the selected workflow type string."""
        return self.type_combo.GetValue()


class MainFrame(wx.Frame):
    """Main application frame with AUI layout."""

    def __init__(self, parent=None, title="FluxImp GUI", base_dir: Optional[Path] = None,
                 app_settings: Optional[AppSettings] = None):
        # Initialize manager with base directory first (needed for config path)
        self.base_dir = base_dir or Path.home() / ".fluximp"
        self.manager = WorkspaceManager(base_dir=self.base_dir)
        self.config_mgr = WorkflowConfigManager(base_dir=self.base_dir)
        # Application-level settings (main.py creates these before the frame
        # so its startup curated-provider gate can read/persist them).
        self.app_settings = app_settings or AppSettings(self.base_dir)

        # Load window state before creating window
        window_state = self._load_window_state()

        # Create frame with default size
        super().__init__(parent, title=title, size=window_state.get('size', (1400, 900)))

        # Restore position if available
        if 'pos' in window_state:
            self.SetPosition(window_state['pos'])

        # Restore maximized state
        if window_state.get('maximized', False):
            self.Maximize(True)
        
        # Track open conversation tabs
        self.open_tabs: dict[str, ConversationTab] = {}
        self.open_agent_tabs: dict[str, AgentTab] = {}
        self.open_event_tabs: dict[str, EventTab] = {}
        self.open_workspace_settings_tabs: dict[str, WorkspaceSettingsTab] = {}
        self.app_settings_tab: Optional[AppSettingsTab] = None
        self.current_workspace: Workspace = None
        
        # Centralized file watcher for all conversations
        self.file_watcher = FileWatcher()
        self.file_watcher.start()

        
        # Create AUI manager
        self.aui_mgr = wx.aui.AuiManager(self)
        
        # Create UI components
        self._create_menu_bar()
        self._create_toolbars()
        self._create_center_pane()
        
        # Add panes to AUI manager
        self.aui_mgr.Update()
        
        # Bind events
        self.Bind(wx.EVT_CLOSE, self._on_close)

        # Capture the platform's own chrome colours first, while every widget
        # still has the colours the platform gave it: Troop replays them so it
        # matches the GUI as it was before themes existed.
        self._platform_chrome = self._capture_platform_chrome()

        # Paint the persisted theme on the window chrome.  Tabs opened later
        # are built with the active theme, so no card re-theming is needed.
        self._apply_theme()

    def _get_window_state_path(self) -> Path:
        """Get path to window state file."""
        return self.base_dir / "window_state.json"

    def _load_window_state(self) -> dict:
        """Load window state from config file."""
        state_path = self._get_window_state_path()
        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    data = json.load(f)
                    # Convert list back to tuple for position and size
                    if 'pos' in data:
                        data['pos'] = tuple(data['pos'])
                    if 'size' in data:
                        data['size'] = tuple(data['size'])
                    return data
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _save_window_state(self):
        """Save window state to config file."""
        state = {
            'maximized': self.IsMaximized()
        }

        # Only save position/size if not maximized
        if not self.IsMaximized():
            state['pos'] = self.GetPosition().Get()
            state['size'] = self.GetSize().Get()
        else:
            # When maximized, load previous non-maximized state and keep it
            prev_state = self._load_window_state()
            if 'pos' in prev_state and 'size' in prev_state:
                state['pos'] = prev_state['pos']
                state['size'] = prev_state['size']

        # Ensure directory exists
        state_path = self._get_window_state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(state_path, 'w') as f:
                json.dump(state, f)
        except IOError:
            pass  # Silently fail if we can't write the file

    def _create_menu_bar(self):
        """Create the menu bar."""
        menubar = wx.MenuBar()
        
        # File menu
        file_menu = wx.Menu()
        
        id_add_workspace = wx.NewIdRef()
        file_menu.Append(id_add_workspace, "Add Workspace...\tCtrl+Shift+W")
        file_menu.Bind(wx.EVT_MENU, self._on_add_workspace, id=id_add_workspace)
        
        id_new_workflow = wx.NewIdRef()
        file_menu.Append(id_new_workflow, "New Workflow...")
        file_menu.Bind(wx.EVT_MENU, self._on_new_workflow, id=id_new_workflow)
        
        id_app_settings = wx.NewIdRef()
        file_menu.Append(id_app_settings, "Settings...")
        file_menu.Bind(wx.EVT_MENU, self._on_app_settings, id=id_app_settings)
        
        file_menu.AppendSeparator()
        
        id_exit = wx.ID_EXIT
        file_menu.Append(id_exit, "Exit\tAlt+F4")
        file_menu.Bind(wx.EVT_MENU, self._on_exit, id=id_exit)
        
        menubar.Append(file_menu, "&File")
        
        # View menu
        view_menu = wx.Menu()
        # API Request Card toggle (app-wide, persisted in settings.json).
        # When off (the default) the workflow history shows only the last
        # completed request card after the last You/Assistant panel plus the
        # pending card(s), keeping the history compact (see the history
        # controller's suppression pass).
        self.id_view_api_cards = wx.NewIdRef()
        self.item_view_api_cards = view_menu.AppendCheckItem(
            self.id_view_api_cards, "API Request Card")
        view_menu.Check(self.id_view_api_cards,
                        self.app_settings.api_request_cards_visible)
        view_menu.Bind(wx.EVT_MENU, self._on_toggle_api_cards,
                       id=self.id_view_api_cards)

        # API Request Card Metadata submenu: one toggleable item per API Request
        # metadata item (app-wide, persisted in settings.json).  Non-API
        # system-agent metadata (subdirectory, overall success/failure,
        # overall run time) is intentionally excluded — it is always shown.
        api_metadata_menu = wx.Menu()
        self.api_metadata_items: dict = {}
        # Bulk section at the top of the submenu: Enable All / Disable All
        # are one-shot actions (no check marks of their own) that set every
        # per-item toggle below and persist in a single write, so sweeping
        # the whole set during testing is one click.
        self.id_api_metadata_enable_all = wx.NewIdRef()
        api_metadata_menu.Append(
            self.id_api_metadata_enable_all, "Enable All")
        api_metadata_menu.Bind(
            wx.EVT_MENU, self._on_set_all_api_metadata,
            id=self.id_api_metadata_enable_all)
        self.id_api_metadata_disable_all = wx.NewIdRef()
        api_metadata_menu.Append(
            self.id_api_metadata_disable_all, "Disable All")
        api_metadata_menu.Bind(
            wx.EVT_MENU, self._on_set_all_api_metadata,
            id=self.id_api_metadata_disable_all)

        api_metadata_menu.AppendSeparator()
        for key, label, _default in API_METADATA_ITEMS:
            item_id = wx.NewIdRef()
            item = api_metadata_menu.AppendCheckItem(item_id, label)
            item.Check(self.app_settings.is_api_metadata_visible(key))
            api_metadata_menu.Bind(
                wx.EVT_MENU, self._on_toggle_api_metadata, id=item_id)
            self.api_metadata_items[key] = (item_id, item)
        self.item_api_metadata_submenu = view_menu.AppendSubMenu(
            api_metadata_menu, "API Request Card Metadata")
        self.item_api_metadata_submenu.Enable(
            self.app_settings.api_request_cards_visible)
        view_menu.AppendSeparator()
        # API Chart submenu: one toggleable item per per-request token series
        # the right panel's chart plots (app-wide, persisted in settings.json).
        # Prompt is on by default; with every item off the chart hides itself
        # (RightPanel/ForkTurnPanel).  Unlike API Request Card Metadata there
        # is no bulk Enable/Disable All section.
        api_chart_menu = wx.Menu()
        self.api_chart_items: dict = {}
        for key, label, _role, _default in CHART_SERIES:
            item_id = wx.NewIdRef()
            item = api_chart_menu.AppendCheckItem(item_id, label)
            item.Check(self.app_settings.is_chart_series_visible(key))
            api_chart_menu.Bind(
                wx.EVT_MENU, self._on_toggle_api_chart_series, id=item_id)
            self.api_chart_items[key] = (item_id, item)
        view_menu.AppendSubMenu(api_chart_menu, "API Chart")
        # API Grid Sums submenu: one toggleable item per metadata-sum
        # column the right panel's grid shows (app-wide, persisted in
        # settings.json).  Every column is off by default -- the grid is a
        # wall of numbers to most users and the chart is the primary metrics
        # source -- and with every item off the grid hides itself
        # (RightPanel/ForkTurnPanel).  Like API Chart there is no bulk
        # Enable/Disable All section.
        grid_sum_menu = wx.Menu()
        self.grid_sum_items: dict = {}
        for key, label, _column, _default in GRID_SUM_COLUMNS:
            item_id = wx.NewIdRef()
            item = grid_sum_menu.AppendCheckItem(item_id, label)
            item.Check(self.app_settings.is_grid_sum_visible(key))
            grid_sum_menu.Bind(
                wx.EVT_MENU, self._on_toggle_grid_sum, id=item_id)
            self.grid_sum_items[key] = (item_id, item)
        self.item_grid_sum_submenu = view_menu.AppendSubMenu(
            grid_sum_menu, "API Grid Sums")

        view_menu.AppendSeparator()
        self.id_view_left = wx.NewIdRef()
        self.item_view_left = view_menu.AppendCheckItem(self.id_view_left, "Show Left Panel")
        view_menu.Check(self.id_view_left, True)
        view_menu.Bind(wx.EVT_MENU, self._on_toggle_left_panel, id=self.id_view_left)
        
        self.id_view_right = wx.NewIdRef()
        self.item_view_right = view_menu.AppendCheckItem(self.id_view_right, "Show Right Panel")
        view_menu.Check(self.id_view_right, True)
        view_menu.Bind(wx.EVT_MENU, self._on_toggle_right_panel, id=self.id_view_right)
        # Theme submenu (fluximp_wxwidgets.themes): one radio item per
        # theme, checked = the active theme.  Switching is gated on having
        # no open tabs -- an already-built card keeps the colors it was
        # constructed with -- so the submenu is rebuilt each time the View
        # menu opens (see _rebuild_theme_menu): with tabs open it collapses
        # to a single disabled "Close all tabs first" line.
        self.view_menu = view_menu
        self.theme_menu = wx.Menu()
        self.theme_menu_items: dict = {}
        view_menu.AppendSubMenu(self.theme_menu, "Theme")
        # Bound ONCE, not per-rebuild: _rebuild_theme_menu allocates fresh
        # item ids every time, so a per-item Bind would accumulate handler
        # entries for the life of the process.  _on_select_theme dispatches
        # by id through theme_menu_items instead.
        self.theme_menu.Bind(wx.EVT_MENU, self._on_select_theme)
        self.Bind(wx.EVT_MENU_OPEN, self._on_menu_open)
        menubar.Append(view_menu, "&View")
        
        self.SetMenuBar(menubar)
    
    def _create_toolbars(self):
        """Create left and right toolbars."""
        # Left panel - workspace tree
        self.left_panel = LeftPanel(self, self.manager, self.config_mgr)
        self.left_panel.set_callbacks(
            on_conversation_selected=self._on_conversation_selected,
            on_workspace_selected=self._on_workspace_selected,
            on_add_workspace=self._on_add_workspace,
            on_new_workflow_dialog=self._on_new_workflow,
            on_new_workflow=self._on_new_workflow_from_tree,
            on_check_open_tab=self._has_open_tab,
            on_agent_double_click=self._on_agent_double_click,
            on_agents_changed=self._on_agents_changed,
            on_sequence_group_double_click=self._on_sequence_group_double_click,
            on_start_conversation=self._on_start_conversation_from_workflow,
            on_create_workflow=self._on_create_workflow_from_tree,
            on_workspace_settings=self._on_workspace_settings_from_tree,
        )
        
        self.aui_mgr.AddPane(
            self.left_panel,
            wx.aui.AuiPaneInfo()
            .Name("left")
            .Caption("Workspaces")
            .Left()
            .Layer(1)
            .Position(1)
            .BestSize((270, -1))
            .MinSize((200, -1))
            .CloseButton(False)
            .MaximizeButton(False)
        )
        
        # Right panel - fork/turn grid
        self.right_panel = RightPanel(self, self.config_mgr, app_settings=self.app_settings)
        self.right_panel.set_callbacks(
            on_turn_selected=self._on_turn_selected
        )
        
        self.aui_mgr.AddPane(
            self.right_panel,
            wx.aui.AuiPaneInfo()
            .Name("right")
            .Caption("Fork/Turn History")
            .Right()
            .Layer(1)
            .Position(1)
            .BestSize((420, -1))
            .MinSize((250, -1))
            .CloseButton(False)
            .MaximizeButton(False)
        )
    
    def _create_center_pane(self):
        """Create the center tabbed notebook."""
        self.notebook = wx.aui.AuiNotebook(
            self,
            style=wx.aui.AUI_NB_DEFAULT_STYLE |
                  wx.aui.AUI_NB_TAB_EXTERNAL_MOVE |
                  wx.aui.AUI_NB_WINDOWLIST_BUTTON |
                  wx.aui.AUI_NB_CLOSE_ON_ALL_TABS
        )
        
        # Bind tab context menu
        self.notebook.Bind(wx.aui.EVT_AUINOTEBOOK_TAB_RIGHT_DOWN, self._on_tab_right_click)
        
        # Bind tab close
        self.notebook.Bind(wx.aui.EVT_AUINOTEBOOK_PAGE_CLOSE, self._on_tab_close)
        
        # Bind tab change
        self.notebook.Bind(wx.aui.EVT_AUINOTEBOOK_PAGE_CHANGED, self._on_tab_changed)
        
        self.aui_mgr.AddPane(
            self.notebook,
            wx.aui.AuiPaneInfo()
            .Name("center")
            .Center()
            .PaneBorder(False)
            .CaptionVisible(False)
            .CloseButton(False)
        )
    
    def _on_add_workspace(self, event=None):
        """Handle Add Workspace menu item."""
        dlg = wx.DirDialog(
            self,
            "Choose a workspace directory",
            style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST
        )
        
        if dlg.ShowModal() == wx.ID_OK:
            path = Path(dlg.GetPath())
            
            # Check if already added
            if self.manager.get_workspace_by_path(path):
                wx.MessageBox(
                    "This workspace is already in the list.",
                    "Workspace Exists",
                    wx.OK | wx.ICON_INFORMATION
                )
            else:
                # Ask for name
                name_dlg = wx.TextEntryDialog(
                    self,
                    "Enter a name for this workspace (optional):",
                    "Workspace Name",
                    path.name
                )
                
                name = None
                if name_dlg.ShowModal() == wx.ID_OK:
                    name = name_dlg.GetValue().strip() or None
                name_dlg.Destroy()
                
                workspace = self.manager.create_workspace(path, name)
                self.left_panel.add_workspace(workspace)
        
        dlg.Destroy()
    
    def _on_new_workflow(self, event=None, preselected_workspace=None):
        """Handle New Workflow - shows dialog to select workspace and type."""
        workspaces = self.manager.get_workspaces()
        if not workspaces:
            wx.MessageBox(
                "Please add a workspace first.",
                "No Workspace",
                wx.OK | wx.ICON_INFORMATION
            )
            return

        dlg = NewWorkflowDialog(
            self, workspaces, self.config_mgr,
            preselected_workspace=preselected_workspace or self.current_workspace
        )
        if dlg.ShowModal() == wx.ID_OK:
            workspace = dlg.get_selected_workspace()
            workflow_type = dlg.get_selected_workflow_type()
            if workspace and workflow_type:
                self._create_workflow(workspace, workflow_type)
        dlg.Destroy()

    def _on_new_workflow_from_tree(self, workspace: Workspace):
        """Handle New Workflow from workspace tree context menu."""
        self._on_new_workflow(preselected_workspace=workspace)

    def _create_workflow(self, workspace: Workspace, workflow_type: str):
        """Create a new workflow in the specified workspace."""
        workflow_config = self.config_mgr.get_workflow(workflow_type)
        if workflow_config is None:
            wx.MessageBox(
                f"Workflow '{workflow_type}' not found.",
                "Error", wx.OK | wx.ICON_ERROR
            )
            return

        # Validate that the workflow has at least one agent
        if not workflow_config.agents:
            dlg = wx.MessageDialog(
                self,
                f"Workflow '{workflow_type}' has no agents defined. "
                "Add agents to the workflow in the Workflows tab before starting a conversation.",
                "No Agents Defined",
                wx.OK | wx.ICON_WARNING
            )
            dlg.ShowModal()
            dlg.Destroy()
            return

        # Determine default agent type: first agent in the workflow
        default_agent = workflow_config.agents[0]

        stub = workspace.create_conversation(
            workflow_name=workflow_type,
            agent_type=default_agent
        )
        self.left_panel.add_conversation(workspace, stub)

        def _configure(conversation: Conversation):
            # Set agent_configs for custom/overridden agents in the workflow
            if self.config_mgr:
                config_paths = self.config_mgr.get_conversation_agent_configs(
                    workflow_type, conversation.agent_type)
                conversation.agent_configs = [str(p) for p in config_paths]
                conversation.workflow_name = workflow_type
                conversation._save_config()

        self._open_conversation_tab(stub, configure=_configure)
        self.current_workspace = workspace

    def _on_conversation_selected(self, stub: ConversationStub):
        """Handle conversation selected from tree."""
        self._open_conversation_tab(stub)

    def _on_agent_double_click(self, agent_name: str):
        """Handle agent double-click - open agent tab in center notebook."""
        # Check if tab already open
        if agent_name in self.open_agent_tabs:
            # Find and select the existing tab
            for i in range(self.notebook.GetPageCount()):
                page = self.notebook.GetPage(i)
                if page is self.open_agent_tabs[agent_name]:
                    self.notebook.SetSelection(i)
                    return
        # Create new agent tab
        tab = AgentTab(self.notebook, agent_name, self.config_mgr)
        self.notebook.AddPage(tab, tab.get_title(), select=True)
        self.open_agent_tabs[agent_name] = tab
    

    def _on_workspace_settings_from_tree(self, workspace: Workspace):
        """Open (or activate) the settings tab for the workspace."""
        ws_id = workspace.workspace_id
        if ws_id in self.open_workspace_settings_tabs:
            # Find and select the existing tab
            for i in range(self.notebook.GetPageCount()):
                if self.notebook.GetPage(i) is self.open_workspace_settings_tabs[ws_id]:
                    self.notebook.SetSelection(i)
                    return
        # Create new settings tab
        tab = WorkspaceSettingsTab(self.notebook, workspace, self.config_mgr)
        self.notebook.AddPage(tab, tab.get_title(), select=True)
        self.open_workspace_settings_tabs[ws_id] = tab

    def _on_app_settings(self, event=None):
        """Open (or activate) the application settings tab."""
        if self.app_settings_tab is not None:
            for i in range(self.notebook.GetPageCount()):
                if self.notebook.GetPage(i) is self.app_settings_tab:
                    self.notebook.SetSelection(i)
                    return
        tab = AppSettingsTab(
            self.notebook, self.app_settings,
            on_limit_to_curated_changed=self._on_limit_to_curated_changed)
        self.notebook.AddPage(tab, tab.get_title(), select=True)
        self.app_settings_tab = tab

    def _on_limit_to_curated_changed(self, desired: bool) -> bool:
        """Apply the curated-filter setting, gating on a usable provider.

        Returns True when the requested value was applied.  Turning the
        filter on when it would leave no provider with an API key set shows
        the same turn-off-or-exit dialog used at startup: 'turn off' leaves
        the setting off (returns False so the checkbox reverts), 'exit'
        closes the frame.
        """
        if desired and not self.right_panel.has_providers_with_curated_filter():
            dlg = NoCuratedProvidersDialog(self)
            result = dlg.ShowModal()
            dlg.Destroy()
            if result != wx.ID_YES:
                # Exit chosen: close the frame (the setting is moot).
                self.Close()
                return False
            # "Turn Off Curated Filtering" chosen: keep the filter off and
            # return False so the checkbox reverts to the persisted value.
            self.app_settings.limit_providers_and_models_to_curated = False
            self.right_panel.apply_curated_filter()
            return False

        self.app_settings.limit_providers_and_models_to_curated = desired
        self.right_panel.apply_curated_filter()
        return True

    def _on_start_conversation_from_workflow(self, workflow_name: str):
        """Handle Start Conversation from Workflows tab."""
        if not self.current_workspace:
            workspaces = self.manager.get_workspaces()
            if not workspaces:
                wx.MessageBox(
                    "Please add a workspace first.",
                    "No Workspace",
                    wx.OK | wx.ICON_INFORMATION
                )
                return
            self.current_workspace = workspaces[0]
        self._create_workflow(self.current_workspace, workflow_name)

    def _on_create_workflow_from_tree(self, workspace: Workspace):
        """Create the workspace's default workflow from the context menu."""
        workflow_name = workspace.get_default_workflow()
        if self.config_mgr and self.config_mgr.get_workflow(workflow_name) is None:
            wx.MessageBox(
                f"Workspace default workflow '{workflow_name}' is no longer defined "
                f"in the Workflows tab. Using '{STOCK_WORKFLOW_NAME}' instead.",
                "Default Workflow Missing",
                wx.OK | wx.ICON_INFORMATION
            )
            workflow_name = STOCK_WORKFLOW_NAME
        self._create_workflow(workspace, workflow_name)

    def _on_agents_changed(self):
        """Handle agents changed - refresh right panel dropdown and agent configs."""
        self.config_mgr.reload()
        workflow_name = None
        if self.right_panel.current_conversation:
            workflow_name = self.right_panel.current_conversation.workflow_name
        self.right_panel.update_agent_types(workflow_name)

    def _on_sequence_group_double_click(self, workflow_name: str, event_key: str, group_name: str):
        """Handle sequence group double-click - open event tab in center notebook."""
        tab_key = f"{workflow_name}:{event_key}:{group_name}"
        # Check if tab already open
        if tab_key in self.open_event_tabs:
            for i in range(self.notebook.GetPageCount()):
                page = self.notebook.GetPage(i)
                if page is self.open_event_tabs[tab_key]:
                    self.notebook.SetSelection(i)
                    return
        # Create new event tab
        tab = EventTab(
            self.notebook, workflow_name, event_key, group_name, self.config_mgr,
            on_save_callback=self._on_event_tab_saved,
        )
        title = f"Event: {group_name} ({workflow_name})"
        self.notebook.AddPage(tab, title, select=True)
        self.open_event_tabs[tab_key] = tab

    def _on_event_tab_saved(self):
        """Handle event tab save - reload config and refresh tree."""
        self.config_mgr.reload()
        if self.left_panel.workflow_panel:
            self.left_panel.workflow_panel.tree.refresh_tree()

    def _on_workspace_selected(self, workspace: Workspace):
        """Handle workspace selected from tree."""
        self.current_workspace = workspace
    
    def _on_turn_selected(self, fork: int, turn: int):
        """Handle turn selected from right panel."""
        # Get current tab's conversation
        current_tab = self._get_current_tab()
        if current_tab:
            # Navigate to this fork/turn
            # This could show the messages for that specific fork/turn
            pass
    
    def _open_conversation_tab(self, stub: ConversationStub,
                               configure: Optional[Callable[[Conversation], None]] = None):
        """Open or activate a conversation tab from its stub.

        Phase 2: builds a parse-free shell Conversation (no blocking io
        parse), creates the tab, and defers the history load (load_existing)
        until after the page is added so the backlog streams through the
        file watcher instead.
        """
        conv_id = stub.conversation_id

        # Check if already open
        if conv_id in self.open_tabs:
            # Find and activate the tab
            for i in range(self.notebook.GetPageCount()):
                if self.notebook.GetPage(i) == self.open_tabs[conv_id]:
                    self.notebook.SetSelection(i)
                    return

        # Build the parse-free shell Conversation synchronously.
        conversation = stub.workspace.build_shell_conversation(conv_id)
        if configure is not None:
            configure(conversation)

        # Freeze notebook to prevent janky visual artifacts during tab creation
        self.notebook.Freeze()
        try:
            # Create new tab
            tab = ConversationTab(
                self.notebook, conversation,
                file_watcher=self.file_watcher,
                config_mgr=self.config_mgr,
                right_panel=self.right_panel,
                app_settings=self.app_settings)
            self.open_tabs[conv_id] = tab

            title = tab.get_title()
            self.notebook.AddPage(
                tab, title, select=True,
                bitmap=get_status_icon(tab.get_status()))
            tab.on_status_changed = lambda t=tab: self._apply_tab_status(t)

            # Force layout to compute proper sizes before showing
            tab.Layout()
        finally:
            self.notebook.Thaw()
        # Force a synchronous paint of the new tab (header, history view,
        # reload status line) and the notebook's tab strip (title + close
        # button) now, while the event queue is still empty. The deferred
        # load_existing starts the watcher reload immediately afterwards;
        # without these paints the queued reload events starve the first
        # paint and the tab chrome does not appear until finalization.
        tab.Update()
        self.notebook.Refresh()
        self.notebook.Update()
        # AuiNotebook's tab strip is an AUI-managed child window that a plain
        # Refresh/Update on the notebook does not always repaint while the
        # reload CallAfter chain keeps the queue busy; force the whole frame to
        # paint synchronously so the tab strip (title + close button) appears
        # immediately (§5.10.1's "tab-close control remains live").
        self.Refresh()
        self.Update()

        # Update right panel
        self.right_panel.set_conversation(conversation, tab=tab)

        # Register conversation with file watcher and start the reload
        # streaming driver deferred (until after the page is added).
        # Adopt the app-wide API Request Card toggle before the reload
        # streams so the backlog finalizes into the right mode.
        tab.history_controller.set_api_cards_visible(
            self.app_settings.api_request_cards_visible)
        self.file_watcher.register_conversation(conversation)
        wx.CallAfter(tab.history_controller.load_existing)
    
    def _apply_tab_status(self, tab: ConversationTab):
        """Update a conversation tab's header status icon (Option F).

        Wired as the tab's on_status_changed callback; the tab dedupes, so
        this fires only on a real status change and the notebook bitmap is
        repainted on genuine transitions only.
        """
        try:
            if self.open_tabs.get(tab.conversation.conversation_id) is not tab:
                return
            # SetPageBitmap takes a page *index*, not the page window (unlike
            # AddPage, whose first arg is the window); resolve it here.
            idx = self.notebook.GetPageIndex(tab)
            if idx < 0:
                return
            self.notebook.SetPageBitmap(idx, get_status_icon(tab.get_status()))
        except RuntimeError:
            # Tab or notebook destroyed mid-update (teardown race).
            pass

    def _get_current_tab(self) -> ConversationTab:
        """Get the currently active conversation tab."""
        idx = self.notebook.GetSelection()
        if idx >= 0:
            page = self.notebook.GetPage(idx)
            if isinstance(page, ConversationTab):
                return page

    def _has_open_tab(self, conversation_id: str) -> bool:
        """Check if a conversation has an open tab."""
        return conversation_id in self.open_tabs
    
    def _on_tab_right_click(self, event: wx.aui.AuiNotebookEvent):
        """Handle right-click on tab."""
        tab_idx = event.GetSelection()
        if tab_idx < 0:
            return
        
        menu = wx.Menu()
        
        id_close = wx.NewIdRef()
        menu.Append(id_close, "Close")
        menu.Bind(wx.EVT_MENU, lambda e: self.notebook.DeletePage(tab_idx), id=id_close)
        
        id_close_others = wx.NewIdRef()
        menu.Append(id_close_others, "Close Others")
        menu.Bind(wx.EVT_MENU, lambda e: self._close_other_tabs(tab_idx), id=id_close_others)
        
        id_close_right = wx.NewIdRef()
        menu.Append(id_close_right, "Close Tabs to the Right")
        menu.Bind(wx.EVT_MENU, lambda e: self._close_tabs_to_right(tab_idx), id=id_close_right)
        
        id_close_left = wx.NewIdRef()
        menu.Append(id_close_left, "Close Tabs to the Left")
        menu.Bind(wx.EVT_MENU, lambda e: self._close_tabs_to_left(tab_idx), id=id_close_left)
        
        self.PopupMenu(menu)
        menu.Destroy()
    
    def _close_other_tabs(self, keep_idx: int):
        """Close all tabs except the specified one."""
        # Close tabs after
        for i in range(self.notebook.GetPageCount() - 1, keep_idx, -1):
            self._close_tab_at_index(i)
        
        # Close tabs before
        for i in range(keep_idx - 1, -1, -1):
            self._close_tab_at_index(i)
    
    def _close_tabs_to_right(self, keep_idx: int):
        """Close all tabs to the right of the specified one."""
        for i in range(self.notebook.GetPageCount() - 1, keep_idx, -1):
            self._close_tab_at_index(i)
    
    def _close_tabs_to_left(self, keep_idx: int):
        """Close all tabs to the left of the specified one."""
        for i in range(keep_idx - 1, -1, -1):
            self._close_tab_at_index(i)
    

    def _close_tab_at_index(self, idx: int):
        """Close tab at index and clean up."""
        page = self.notebook.GetPage(idx)
        if isinstance(page, ConversationTab):
            # Quiesce the file watcher for this conversation BEFORE tearing
            # down the tab, so the watcher thread stops posting callbacks
            # that could race native window destruction during teardown.
            conv_id = page.conversation.conversation_id
            if conv_id in self.open_tabs:
                self.file_watcher.unregister_conversation(conv_id)
            page.close()
            # Remove from open_tabs (the tab owns the Conversation now)
            if conv_id in self.open_tabs:
                del self.open_tabs[conv_id]
        elif isinstance(page, EventTab):
            for tab_key, tab in list(self.open_event_tabs.items()):
                if tab == page:
                    del self.open_event_tabs[tab_key]
                    break
        elif isinstance(page, AgentTab):
            for agent_name, tab in list(self.open_agent_tabs.items()):
                if tab == page:
                    del self.open_agent_tabs[agent_name]
                    break
        elif isinstance(page, WorkspaceSettingsTab):
            # Remove from open_workspace_settings_tabs
            for ws_id, tab in list(self.open_workspace_settings_tabs.items()):
                if tab == page:
                    del self.open_workspace_settings_tabs[ws_id]
                    break
        elif isinstance(page, AppSettingsTab):
            if self.app_settings_tab is page:
                self.app_settings_tab = None
        self.notebook.DeletePage(idx)
    

    def _on_tab_close(self, event: wx.aui.AuiNotebookEvent):
        """Handle tab close button - unload conversation/agent to free memory."""
        idx = event.GetSelection()
        page = self.notebook.GetPage(idx)

        if isinstance(page, ConversationTab):
            # Quiesce the file watcher for this conversation BEFORE tearing
            # down the tab, so the watcher thread stops posting callbacks
            # that could race native window destruction during teardown.
            conv_id = page.conversation.conversation_id
            if conv_id in self.open_tabs:
                self.file_watcher.unregister_conversation(conv_id)
            page.close()
            # Remove from open_tabs (the tab owns the Conversation now)
            if conv_id in self.open_tabs:
                del self.open_tabs[conv_id]
        elif isinstance(page, EventTab):
            # Remove from open_event_tabs
            for tab_key, tab in list(self.open_event_tabs.items()):
                if tab == page:
                    del self.open_event_tabs[tab_key]
                    break
        elif isinstance(page, AgentTab):
            # Remove from open_agent_tabs
            for agent_name, tab in list(self.open_agent_tabs.items()):
                if tab == page:
                    del self.open_agent_tabs[agent_name]
                    break
        elif isinstance(page, WorkspaceSettingsTab):
            # Remove from open_workspace_settings_tabs
            for ws_id, tab in list(self.open_workspace_settings_tabs.items()):
                if tab == page:
                    del self.open_workspace_settings_tabs[ws_id]
                    break
        elif isinstance(page, AppSettingsTab):
            if self.app_settings_tab is page:
                self.app_settings_tab = None

        event.Skip()

    
    def _on_tab_changed(self, event: wx.aui.AuiNotebookEvent):
        """Handle tab change - update right panel."""
        idx = event.GetSelection()
        if idx >= 0:
            page = self.notebook.GetPage(idx)
            if isinstance(page, ConversationTab):
                # Update right panel with current conversation
                self.right_panel.set_conversation(page.conversation, tab=page)
                # Background tabs adopt the app-wide API Request Card toggle
                # only when they become active.
                page.history_controller.set_api_cards_visible(
                    self.app_settings.api_request_cards_visible)

        event.Skip()
    
    def _on_toggle_api_cards(self, event: wx.Event):
        """Toggle rendering of every API Request card in the history.

        Persists the new state (app-wide).  The change applies immediately
        to the active conversation tab; background tabs adopt it when they
        next become active (see _on_tab_changed).  With the cards switched
        off the API Request Card Metadata submenu is greyed out — its items
        have no cards to affect.
        """
        visible = self.item_view_api_cards.IsChecked()
        self.app_settings.api_request_cards_visible = visible
        self.item_api_metadata_submenu.Enable(visible)
        tab = self._get_current_tab()
        if tab is not None:
            tab.history_controller.set_api_cards_visible(visible)

    def _on_toggle_left_panel(self, event: wx.Event):
        """Toggle left panel visibility."""
        pane = self.aui_mgr.GetPane("left")
        pane.Show(not pane.IsShown())
        self.aui_mgr.Update()
    
    def _on_toggle_right_panel(self, event: wx.Event):
        """Toggle right panel visibility."""
        pane = self.aui_mgr.GetPane("right")
        pane.Show(not pane.IsShown())
        self.aui_mgr.Update()

    def _on_toggle_api_metadata(self, event: wx.Event):
        """Toggle visibility of one API Request metadata item.

        Persists the new state (app-wide) and pushes it to every open
        conversation's history view so already-built cards update live.
        """
        event_id = event.GetId()
        for key, (item_id, item) in self.api_metadata_items.items():
            if int(item_id) == event_id:
                self.app_settings.set_api_metadata_visible(
                    key, item.IsChecked())
                self._refresh_api_metadata_display()
                return

    def _on_set_all_api_metadata(self, event: wx.Event):
        """Enable or disable every API Request metadata item at once.

        The View > API Request Card Metadata submenu's bulk section (Enable
        All / Disable All) routes here: every per-item check is set, the
        state is persisted in a single write and every open conversation is
        refreshed via the same path as a single toggle.
        """
        enable = event.GetId() == int(self.id_api_metadata_enable_all)
        for _key, (_item_id, item) in self.api_metadata_items.items():
            item.Check(enable)
        self.app_settings.set_all_api_metadata_visible(enable)
        self._refresh_api_metadata_display()

    def _on_toggle_api_chart_series(self, event: wx.Event):
        """Toggle one per-request token series on the right panel's chart.

        Persists the new state (app-wide) and pushes it to the chart, which
        hides itself when every series is switched off.
        """
        event_id = event.GetId()
        for key, (item_id, item) in self.api_chart_items.items():
            if int(item_id) == event_id:
                self.app_settings.set_chart_series_visible(
                    key, item.IsChecked())
                panel = getattr(self, "right_panel", None)
                if panel is not None:
                    panel.apply_chart_series_visibility()
                return

    def _on_toggle_grid_sum(self, event: wx.Event):
        """Toggle one metadata-sum column on the right panel's grid.

        Persists the new state (app-wide) and pushes it to the grid, which
        hides itself when every column is switched off.
        """
        event_id = event.GetId()
        for key, (item_id, item) in self.grid_sum_items.items():
            if int(item_id) == event_id:
                self.app_settings.set_grid_sum_visible(key, item.IsChecked())
                panel = getattr(self, "right_panel", None)
                if panel is not None:
                    panel.apply_grid_sum_visibility()
                return

    def _refresh_api_metadata_display(self):
        """Push API metadata visibility changes to every open conversation."""
        for tab in list(self.open_tabs.values()):
            view = getattr(tab, 'history_view', None)
            if view is None:
                continue
            try:
                view.refresh_api_metadata_display()
            except RuntimeError:
                continue
    
    def _on_menu_open(self, event: wx.MenuEvent):
        """Rebuild the View > Theme submenu just before the View menu opens."""
        try:
            menu = event.GetMenu()
            # wxPython normally hands back the same Python object for a given
            # C++ menu, so `is` holds; the title comparison is a fallback if
            # that identity ever fails.  The theme submenu's own title
            # ("Theme") can never match "&View", so the submenu is never
            # rebuilt while it is itself open; a popup menu (empty title)
            # never matches either.
            if menu is not None and (
                    menu is self.view_menu
                    or menu.GetTitle() == self.view_menu.GetTitle()):
                self._rebuild_theme_menu()
                # Re-assert the API Request Card Metadata gate in case the
                # API Request Card toggle changed while the menu was closed.
                self.item_api_metadata_submenu.Enable(
                    self.app_settings.api_request_cards_visible)
        except RuntimeError:
            # Menu/frame torn down mid-open (teardown race).
            pass
        event.Skip()

    def _rebuild_theme_menu(self):
        """Populate the View > Theme submenu for the current tab state.

        With any tab open the theme cannot be switched (an open card keeps
        the colors it was built with), so the submenu collapses to a single
        disabled "Close all tabs first" line; with no tabs open it lists one
        radio item per theme, checked = the active theme.
        """
        menu = self.theme_menu
        # Rebuild from scratch on every open: cheap, and it keeps the radio
        # checks and the open-tab gate from ever drifting out of sync.
        # wxPython binds wxMenu's item-deleting overload as DestroyItem: the
        # bare Destroy() name resolves to wxEvtHandler.Destroy(), which takes
        # no arguments (that is the call a popup menu makes to delete itself
        # below), so Destroy(item) raises "too many arguments".
        while menu.GetMenuItemCount() > 0:
            menu.DestroyItem(menu.FindItemByPosition(0))
        self.theme_menu_items = {}

        if self.notebook.GetPageCount() > 0:
            menu.Append(wx.ID_ANY, "Close all tabs first").Enable(False)
            return

        for name in themes.theme_names():
            item_id = wx.NewIdRef()
            item = menu.AppendRadioItem(
                item_id, themes.DISPLAY_NAMES.get(name, name))
            item.Check(name == themes.active_name())
            self.theme_menu_items[int(item_id)] = name

    def _on_select_theme(self, event: wx.Event):
        """Persist and apply the chosen theme (reachable only with no tabs)."""
        name = self.theme_menu_items.get(int(event.GetId()))
        if name is None:
            return
        self.app_settings.theme = name
        themes.set_active(self.app_settings.theme)
        self._apply_theme()

    def _capture_platform_chrome(self) -> dict:
        """Snapshot the colours the platform gave each surface, before we paint.

        Troop reproduces the GUI as it was before themes existed, and that GUI
        left these colours to the platform -- but not all of them to the *same*
        value, which is why this is per-surface rather than one colour.  On
        Windows the pane background is the dialog grey while the trees, the
        grid's cells and the notebook's page area are white, so replaying one
        captured grey over all of them is what turned the trees, the grid and
        the page area grey while the pane itself was finally right.

        Recorded at construction, before the first _apply_theme, while every
        widget still has the colours the platform gave it.  Replayed by
        _apply_theme for the Troop theme, which also lets a switch *back*
        from a dark theme restore the real thing rather than leaving the dark
        colours behind.
        """
        grid_colours = None
        grid = self._fork_turn_grid()
        if grid is not None:
            try:
                grid_colours = {
                    "cell_bg": grid.GetDefaultCellBackgroundColour(),
                    "cell_fg": grid.GetDefaultCellTextColour(),
                    "label_bg": grid.GetLabelBackgroundColour(),
                    "label_fg": grid.GetLabelTextColour(),
                    "line": grid.GetGridLineColour(),
                    "widget_bg": grid.GetBackgroundColour(),
                }
            except Exception as exc:
                logger.warning("Theme: grid colour capture failed: %s", exc)
        panes = {}
        for pane_name in ("left", "right"):
            pane = self.aui_mgr.GetPane(pane_name)
            if pane is not None and pane.window is not None:
                panes[pane_name] = _capture_widget_colours(pane.window)
        return {
            "frame": (self.GetBackgroundColour(), self.GetForegroundColour()),
            "notebook": (self.notebook.GetBackgroundColour(),
                         self.notebook.GetForegroundColour()),
            "grid": grid_colours,
            "panes": panes,
        }

    def _fork_turn_grid(self):
        """The right panel's fork/turn grid, or None when it is not built."""
        panel = getattr(self, "right_panel", None)
        return getattr(getattr(panel, "fork_turn_panel", None), "grid", None)

    def _fork_turn_chart(self):
        """The right panel's API chart, or None when it is not built."""
        panel = getattr(self, "right_panel", None)
        return getattr(getattr(panel, "fork_turn_panel", None), "chart", None)

    def _apply_theme(self):
        """Re-color the window chrome for the active theme.

        Only chrome is themed live: a switch is reachable only with zero
        open tabs (see _rebuild_theme_menu), so no card needs re-theming --
        tabs opened afterwards resolve colors at construction.

        Every surface the frame owns is themed except the icon bitmaps: the
        panes and everything inside them, the grid, the notebook, and then the
        two surfaces no wx widget colour can reach -- the AUI art (see
        _apply_aui_theme) and the native appearance (see _apply_appearance).

        Troop is the exception, and it is why this branches rather than reading
        every colour out of one palette.  Troop reproduces the GUI as it was
        before themes existed, and that GUI left these colours to the platform
        -- but not all of them to the *same* value.  On Windows the pane
        background is the dialog grey while the trees, the grid's cells and the
        notebook's page area are white, so no single colour can reproduce it:
        painting the pane's grey over all of them is exactly what turned the
        trees, the grid and the page area grey.  Troop therefore replays the
        per-widget colours captured at construction (see
        _capture_platform_chrome).
        """
        theme = themes.active()
        platform = getattr(self, "_platform_chrome", None)
        grid = self._fork_turn_grid()

        if themes.active_name() == themes.TROOP_THEME_KEY and platform:
            self.SetBackgroundColour(platform["frame"][0])
            self.SetForegroundColour(platform["frame"][1])
            cleared = []
            for pane_name in ("left", "right"):
                pane = self.aui_mgr.GetPane(pane_name)
                if pane is None or pane.window is None:
                    continue
                _restore_widget_colours(
                    pane.window, platform["panes"].get(pane_name, {}))
                cleared.extend(_clear_notebook_backgrounds(pane.window))
            if cleared:
                logger.info(
                    "Theme: cleared the explicit background of %d notebook "
                    "widget(s) so each page area matches its native tab "
                    "strip: %s", len(cleared), ", ".join(cleared))
            if grid is not None and platform["grid"]:
                _theme_grid(grid, **platform["grid"])
            self.notebook.SetBackgroundColour(platform["notebook"][0])
            self.notebook.SetForegroundColour(platform["notebook"][1])
        else:
            bg = themes.to_wx(theme.panel_bg)
            fg = themes.to_wx(theme.text_fg)
            self.SetBackgroundColour(themes.to_wx(theme.window_bg))
            self.SetForegroundColour(fg)
            # Both panes are themed, children included: a native control
            # (tree, notebook, grid, text field) paints its own background and
            # does not inherit the panel's, so each has to be told explicitly.
            # The tree *icons* are the one exception -- they are bitmaps drawn
            # with fixed colours and no Set*Colour call reaches them.
            for pane_name in ("left", "right"):
                pane = self.aui_mgr.GetPane(pane_name)
                if pane is not None and pane.window is not None:
                    _theme_window_tree(pane.window, bg, fg)
            if grid is not None:
                _theme_grid(
                    grid,
                    cell_bg=bg,
                    cell_fg=fg,
                    label_bg=themes.to_wx(theme.section_header_bg),
                    label_fg=fg,
                    line=themes.to_wx(theme.panel_border),
                    widget_bg=bg)
            self.notebook.SetBackgroundColour(bg)
            self.notebook.SetForegroundColour(fg)

        # The chart paints itself from the theme's own chart roles, so it is
        # repainted here rather than left to the pane-wide walk above (which
        # would give it the pane's background) or to the captured platform
        # colours (which are not its own).
        chart = self._fork_turn_chart()
        if chart is not None:
            chart.apply_theme()

        self.notebook.Refresh()
        # The area behind the panes, the pane captions and the notebook's tab
        # strip are painted by the AUI art providers, not by the widgets.
        self._apply_aui_theme(theme)
        # The title bar, the native menu bar and the standard controls inside
        # the panes are drawn by the OS or by wx itself, so they need the
        # native appearance rather than any wx widget colour call.
        self._apply_appearance(themes.active_name())
        self.Refresh()
        self.Update()

    def _apply_aui_theme(self, theme) -> None:
        """Theme the AUI chrome: the area behind the panes, the pane
        captions ("Workspaces", "Fork/Turn History"), the sashes between
        panes and the notebook's tab strip.

        Those surfaces are painted by the AUI art providers, not by the
        widgets themselves, so without this a theme switch would only
        re-color the panels' own backgrounds and the window would look
        almost untouched.

        Every step logs what it did.  wxAuiDefaultDockArt::SetColour
        silently ignores a colour id it does not recognise, so a silent
        no-op here is otherwise indistinguishable from a repaint failure
        (and an unlogged one at that -- which is exactly how the first cut
        of this method shipped a no-op).

        Nothing in this method may raise: a theme switch must never be able
        to break the window, and a wx build without the AUI art bindings
        keeps the platform look (which is exactly Troop's appearance).
        """
        bg = themes.to_wx(theme.panel_bg)
        fg = themes.to_wx(theme.text_fg)
        caption_bg = themes.to_wx(theme.section_header_bg)
        border = themes.to_wx(theme.panel_border)

        dock_art = None
        try:
            dock_art = self.aui_mgr.GetArtProvider()
        except AttributeError as exc:
            logger.warning("AUI theme: no dock art provider (%s)", exc)

        if dock_art is not None:
            # The colour ids come from the wx.aui constants by name, never
            # from hard-coded numbers: the enum's first five entries are
            # *sizes*, so the colours start at 5, and a build that renumbered
            # them would otherwise be repainted on the wrong surfaces.
            for attr_name, colour in (
                    ("AUI_DOCKART_BACKGROUND_COLOUR", bg),
                    ("AUI_DOCKART_SASH_COLOUR", border),
                    ("AUI_DOCKART_BORDER_COLOUR", border),
                    ("AUI_DOCKART_GRIPPER_COLOUR", border),
                    ("AUI_DOCKART_ACTIVE_CAPTION_COLOUR", caption_bg),
                    ("AUI_DOCKART_ACTIVE_CAPTION_GRADIENT_COLOUR", caption_bg),
                    ("AUI_DOCKART_ACTIVE_CAPTION_TEXT_COLOUR", fg),
                    ("AUI_DOCKART_INACTIVE_CAPTION_COLOUR", caption_bg),
                    ("AUI_DOCKART_INACTIVE_CAPTION_GRADIENT_COLOUR", caption_bg),
                    ("AUI_DOCKART_INACTIVE_CAPTION_TEXT_COLOUR", fg)):
                art_id = getattr(wx.aui, attr_name, None)
                if art_id is None:
                    logger.warning("AUI theme: wx.aui.%s is missing", attr_name)
                    continue
                try:
                    dock_art.SetColour(art_id, colour)
                except Exception as exc:
                    # A theme switch must never raise; log and press on.
                    logger.warning("AUI theme: SetColour(%s=%s) failed: %s",
                                   attr_name, art_id, exc)
            # Read-back, at debug: the only way to tell an id this build
            # actually knows from one it silently ignores.
            try:
                got = dock_art.GetColour(
                    wx.aui.AUI_DOCKART_ACTIVE_CAPTION_COLOUR)
                logger.debug(
                    "AUI theme: active caption colour now %s (wanted %s)",
                    got.GetAsString(wx.C2S_HTML_SYNTAX),
                    caption_bg.GetAsString(wx.C2S_HTML_SYNTAX))
            except Exception as exc:
                logger.debug("AUI theme: caption colour read-back failed: %s",
                             exc)

        # Recolor the notebook's existing tab art in place.  A new art
        # provider is deliberately NOT installed: wxAuiNotebook takes
        # ownership of the art it is given, and a Python-held reference to
        # an owned art is a double-free hazard at teardown.
        tab_art = None
        try:
            tab_art = self.notebook.GetArtProvider()
        except AttributeError as exc:
            logger.warning("AUI theme: notebook has no tab art (%s)", exc)
        if tab_art is not None:
            try:
                tab_art.SetColour(bg)
                tab_art.SetActiveColour(caption_bg)
                logger.debug("AUI theme: notebook tab art recoloured")
            except Exception as exc:
                logger.warning("AUI theme: tab art recolour failed: %s", exc)

        # The art changes only take effect on the next paint of the manager's
        # window and of the windows it manages.
        try:
            self.aui_mgr.Update()
        except Exception as exc:
            logger.warning("AUI theme: aui Update failed: %s", exc)
        self.Refresh()
        self.Update()

    def _apply_appearance(self, theme_name: str) -> None:
        """Ask wx for the native light/dark appearance matching *theme_name*.

        Thin per-frame wrapper around the module-level apply_appearance() so
        _apply_theme reads like its sibling _apply_* calls; see that function
        for why this is the portable route to the title bar and the menu bar,
        which no wx widget colour can reach.
        """
        apply_appearance(theme_name)

    def _on_exit(self, event: wx.Event):
        """Handle Exit menu item."""
        self.Close()
    
    def _on_close(self, event: wx.CloseEvent):
        """Handle window close."""
        # Save window state before closing
        self._save_window_state()

        # Stop the file watcher first
        self.file_watcher.stop()

        # Close all tabs gracefully
        for tab in list(self.open_tabs.values()):
            tab.close()

        # Uninitialize AUI manager
        self.aui_mgr.UnInit()

        event.Skip()


def _resolve_appearance_enum(app):
    """Find the wxApp::Appearance enum wherever this build exposes it.

    The enum is declared inside wxAppBase, so wxPython surfaces it as
    wx.PyApp.Appearance -- reachable as wx.App.Appearance and on the app
    instance too -- and *not* as wx.Appearance.  Searching only the wx module
    is why the first cut never found it and skipped the portable route
    entirely, which is what left a switch back to a light theme unable to
    clear the dark appearance.

    Returns the enum, or None on a wx build that predates it (pre-4.3), where
    the caller falls through to the MSW-only opt-in and then to a no-op.
    """
    for holder in (getattr(wx, "App", None),
                   getattr(wx, "PyApp", None),
                   app,
                   wx):
        enum = getattr(holder, "Appearance", None)
        if enum is not None and getattr(enum, "Light", None) is not None:
            return enum
    return None


def _appearance_result_ok(result) -> bool:
    """Interpret SetAppearance's AppearanceResult return value.

    wxAppBase::SetAppearance returns AppearanceResult::{Failure, Ok,
    CannotChange} and the base implementation returns Failure, so the return
    value -- not an exception -- is how "this build cannot do it" is reported.
    Compared by name because the enum's Python surface varies between bindings;
    a binding that exposes no return value at all is treated as success, since
    the call did not raise.
    """
    text = str(getattr(result, "name", result)).lower()
    return "failure" not in text and "cannot" not in text


def apply_appearance(theme_name: str) -> None:
    """Ask wxWidgets for the native light or dark appearance for *theme_name*.

    This is the portable route to the surfaces wxWidgets does not let an
    application colour directly: the window title bar, the native menu bar and
    the standard controls inside the panes (scrollbars, tree expanders, combo
    popups).  wxWidgets implements it per platform internally -- a dark title
    bar and menu bar on MSW, the dark GTK theme variant on Linux, NSAppearance
    on macOS -- so the call is identical everywhere and no OS-specific code is
    needed here.

    wx 4.3 / wxWidgets 3.3 added it.  On an older wx the attribute is simply
    absent, so this degrades to a no-op and the app keeps its own
    (theme-coloured) widgets; every lookup is by getattr for exactly that
    reason.

    Call it before creating any windows for full effect -- main.py does, right
    after resolving the persisted theme and before MainFrame exists.  Calling
    it again for a live window (a theme switch) is best-effort: wx documents
    that a later change may not reach every already-created native control, so
    the theme's own colours stay the source of truth for the widgets.

    Both directions are requested.  SetAppearance() takes Light as well as
    Dark, and asking for Light is the only portable way to *leave* dark mode;
    see _resolve_appearance_enum for why that direction used to be skipped.
    """
    dark = themes.is_dark(theme_name)
    # Troop is the GUI as it was before themes existed, and that GUI never
    # asked wx for an appearance at all -- so this asks for *System* rather
    # than forcing Light.  Forcing Light would make Troop differ from the
    # unthemed GUI on a machine whose desktop is dark, and "match the unthemed
    # GUI" is the whole point of Troop.
    if theme_name == themes.TROOP_THEME_KEY:
        wanted_name = "System"
    else:
        wanted_name = "Dark" if dark else "Light"
    app = wx.GetApp()

    # Portable route first: wxApp::SetAppearance() is the documented,
    # cross-platform entry point, and the only one that can force a *light*
    # appearance regardless of the system setting (see the fallback below).
    #
    # The enum is resolved rather than read off wx because it is declared
    # inside wxAppBase: wxPython surfaces it as wx.PyApp.Appearance (and
    # therefore as wx.App.Appearance and on the app instance), never as
    # wx.Appearance.  Reading only the wx module is why an earlier cut skipped
    # this route entirely -- and with the fallback below gated on `if dark:`,
    # a switch back to a light theme then made no call at all, leaving the
    # native chrome dark until the app was restarted.
    enum = _resolve_appearance_enum(app)
    if enum is not None:
        wanted = getattr(enum, wanted_name, None)
        for label, holder in (("wx.GetApp()", app),
                              ("wx.App", getattr(wx, "App", None)),
                              ("wx.PyApp", getattr(wx, "PyApp", None))):
            func = (getattr(holder, "SetAppearance", None)
                    if holder is not None else None)
            if not callable(func):
                continue
            try:
                result = func(wanted)
            except Exception as exc:
                logger.info("Theme: %s.SetAppearance(%s) failed: %s",
                            label, wanted_name, exc)
                continue
            if _appearance_result_ok(result):
                logger.info(
                    "Theme: native appearance set to %s via %s.SetAppearance "
                    "-> %s", wanted_name, label, result)
                return
            logger.info(
                "Theme: %s.SetAppearance(%s) -> %s (not applied); trying the "
                "next route", label, wanted_name, result)

    # wxMSW's own opt-in, attempted purely by name so a GTK or macOS build
    # simply has no such attribute and this is a no-op there: no platform
    # check, no ctypes, just the next wx API in the list.
    #
    # Called for *both* light and dark themes.  wx/msw/app.h declares only
    # DarkMode_Auto = 0 and DarkMode_Always = 1 -- there is no "force light"
    # flag -- so Auto ("use dark mode if the system is using it") is the
    # closest available "stop forcing dark", and calling it on the light
    # branch is what clears a previously-set DarkMode_Always.
    enable = getattr(app, "MSWEnableDarkMode", None)
    flag = getattr(app, "DarkMode_Always" if dark else "DarkMode_Auto", None)
    if callable(enable) and flag is not None:
        try:
            enable(flag)
        except Exception as exc:
            logger.info("Theme: MSWEnableDarkMode(%s) failed: %s", flag, exc)
        else:
            logger.info("Theme: native appearance set to %s via "
                        "MSWEnableDarkMode(%s)", wanted_name, flag)
            return

    logger.info(
        "Theme: no usable native appearance call on this wx "
        "(Appearance enum: %s, MSWEnableDarkMode: %s); native chrome keeps "
        "the platform look", enum is not None, callable(enable))
