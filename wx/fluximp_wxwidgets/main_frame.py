"""Main frame with AUI layout and menu bar."""

import wx
import wx.aui
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

    def __init__(self, parent=None, title="Fluximp Agent GUI", base_dir: Optional[Path] = None,
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
        
        self.id_view_left = wx.NewIdRef()
        self.item_view_left = view_menu.AppendCheckItem(self.id_view_left, "Show Left Panel")
        view_menu.Check(self.id_view_left, True)
        view_menu.Bind(wx.EVT_MENU, self._on_toggle_left_panel, id=self.id_view_left)
        
        self.id_view_right = wx.NewIdRef()
        self.item_view_right = view_menu.AppendCheckItem(self.id_view_right, "Show Right Panel")
        view_menu.Check(self.id_view_right, True)
        view_menu.Bind(wx.EVT_MENU, self._on_toggle_right_panel, id=self.id_view_right)
        
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
                right_panel=self.right_panel)
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

        event.Skip()
    
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
