"""Left panel - Tabbed workspace/workflow management for Fluximp GUI."""

import wx
from pathlib import Path
from typing import Optional, Callable

from fluximp_wxwidgets.workspace import WorkspaceManager, Workspace, ConversationStub
from fluximp_wxwidgets.conversation_manager import Conversation
from fluximp_wxwidgets.workflow_config import WorkflowConfigManager
from fluximp_wxwidgets.workflow_panel import WorkflowPanel


class WorkspaceTreeCtrl(wx.TreeCtrl):
    """Tree control for displaying workspaces and conversations."""
    
    # Tree item data types
    TYPE_WORKSPACE = "workspace"
    TYPE_CONVERSATION = "conversation"
    
    def __init__(self, parent, manager: WorkspaceManager):
        super().__init__(parent, style=wx.TR_DEFAULT_STYLE | wx.TR_HIDE_ROOT)
        
        self.manager = manager
        
        # Image list for icons
        self.img_list = wx.ImageList(16, 16)
        self.img_folder = self.img_list.Add(self._create_folder_bitmap())
        self.img_chat = self.img_list.Add(self._create_chat_bitmap())
        self.SetImageList(self.img_list)
        
        # Root is hidden
        self.root = self.AddRoot("Workspaces")
        
        # Event handlers
        self.Bind(wx.EVT_TREE_ITEM_ACTIVATED, self._on_item_activated)
        self.Bind(wx.EVT_TREE_ITEM_RIGHT_CLICK, self._on_item_right_click)
        self.Bind(wx.EVT_TREE_SEL_CHANGED, self._on_selection_changed)
        
        # Callbacks
        self.on_conversation_selected: Optional[Callable[[ConversationStub], None]] = None
        self.on_new_workflow: Optional[Callable[[Workspace], None]] = None
        self.on_workspace_selected: Optional[Callable[[Workspace], None]] = None
        self.on_check_open_tab: Optional[Callable[[str], bool]] = None
        self.on_create_workflow: Optional[Callable[[Workspace], None]] = None
        self.on_workspace_settings: Optional[Callable[[Workspace], None]] = None

        # Build initial tree
        self._refresh_tree()
    
    def _create_folder_bitmap(self) -> wx.Bitmap:
        """Create a simple folder icon bitmap."""
        bmp = wx.Bitmap(16, 16)
        dc = wx.MemoryDC(bmp)
        dc.SetBackground(wx.Brush(wx.Colour(240, 240, 240)))
        dc.Clear()
        dc.SetPen(wx.Pen(wx.Colour(180, 180, 180)))
        dc.SetBrush(wx.Brush(wx.Colour(255, 220, 100)))
        dc.DrawRoundedRectangle(2, 4, 12, 10, 2)
        dc.SelectObject(wx.NullBitmap)
        return bmp
    
    def _create_chat_bitmap(self) -> wx.Bitmap:
        """Create a simple chat icon bitmap."""
        bmp = wx.Bitmap(16, 16)
        dc = wx.MemoryDC(bmp)
        dc.SetBackground(wx.Brush(wx.Colour(240, 240, 240)))
        dc.Clear()
        dc.SetPen(wx.Pen(wx.Colour(100, 150, 220)))
        dc.SetBrush(wx.Brush(wx.Colour(150, 200, 255)))
        dc.DrawEllipse(2, 2, 12, 10)
        dc.SelectObject(wx.NullBitmap)
        return bmp
    
    def _refresh_tree(self):
        """Refresh the entire tree from the manager."""
        self.DeleteChildren(self.root)
        
        for workspace in self.manager.get_workspaces():
            self._add_workspace_node(workspace)
    
    def _add_workspace_node(self, workspace: Workspace) -> wx.TreeItemId:
        """Add a workspace node to the tree."""
        item = self.AppendItem(self.root, workspace.name, image=self.img_folder)
        self.SetItemData(item, (self.TYPE_WORKSPACE, workspace))

        # Add conversation stubs (lightweight, not full conversations)
        for conv_id in workspace.get_conversation_ids():
            stub = workspace.conversations.get(conv_id)
            if stub:
                self._add_conversation_node(item, stub)

        return item
    
    def _add_conversation_node(self, parent: wx.TreeItemId,
                                stub: ConversationStub) -> wx.TreeItemId:
        """Add a conversation stub node under a workspace."""
        # Format timestamp nicely
        timestamp = stub.start_time.strftime("%Y-%m-%d %H:%M")
        label = f"{stub.workflow_name} {timestamp}"

        item = self.AppendItem(parent, label, image=self.img_chat)
        self.SetItemData(item, (self.TYPE_CONVERSATION, stub))
        return item
    
    def add_workspace(self, workspace: Workspace):
        """Add a new workspace to the tree."""
        self._add_workspace_node(workspace)
    
    def add_conversation(self, workspace: Workspace, stub: ConversationStub):
        """Add a new conversation stub to the tree."""
        # Find the workspace node
        item, cookie = self.GetFirstChild(self.root)
        while item.IsOk():
            data = self.GetItemData(item)
            if data and data[0] == self.TYPE_WORKSPACE and data[1] == workspace:
                self._add_conversation_node(item, stub)
                self.Expand(item)
                return
            item, cookie = self.GetNextChild(self.root, cookie)

    def remove_conversation(self, stub: ConversationStub):
        """Remove a conversation node from the tree."""
        item = self._find_conversation_item(stub)
        if item and item.IsOk():
            self.Delete(item)

    def _find_conversation_item(self, stub: ConversationStub) -> Optional[wx.TreeItemId]:
        """Find the tree item for a conversation stub."""
        item, cookie = self.GetFirstChild(self.root)
        while item.IsOk():
            child, child_cookie = self.GetFirstChild(item)
            while child.IsOk():
                data = self.GetItemData(child)
                if data and data[0] == self.TYPE_CONVERSATION and data[1] == stub:
                    return child
                child, child_cookie = self.GetNextChild(item, child_cookie)
            item, cookie = self.GetNextChild(self.root, cookie)
        return None
    
    def _on_item_activated(self, event: wx.TreeEvent):
        """Handle double-click on an item - lazy load conversation if needed."""
        try:
            if self.IsBeingDeleted():
                return
        except RuntimeError:
            return

        item = event.GetItem()
        if not item.IsOk():
            return

        try:
            data = self.GetItemData(item)
        except RuntimeError:
            return
        if not data:
            return

        item_type, item_data = data

        if item_type == self.TYPE_CONVERSATION and self.on_conversation_selected:
            # item_data is a ConversationStub - the frame builds the parse-free
            # shell Conversation and streams the backlog via the watcher (Phase
            # 2, no blocking parse on the GUI thread).
            stub: ConversationStub = item_data
            self.on_conversation_selected(stub)
        elif item_type == self.TYPE_WORKSPACE and self.on_workspace_selected:
            self.on_workspace_selected(item_data)

    def _on_selection_changed(self, event: wx.TreeEvent):
        """Handle selection change (single-click, keyboard nav)."""
        try:
            if self.IsBeingDeleted():
                return
        except RuntimeError:
            return

        item = event.GetItem()
        if not item.IsOk():
            return

        try:
            data = self.GetItemData(item)
        except RuntimeError:
            return
        if not data:
            return

        item_type, item_data = data

        if item_type == self.TYPE_WORKSPACE and self.on_workspace_selected:
            self.on_workspace_selected(item_data)

    def _on_item_right_click(self, event: wx.TreeEvent):
        """Handle right-click on an item."""
        try:
            if self.IsBeingDeleted():
                return
        except RuntimeError:
            return

        item = event.GetItem()
        if not item.IsOk():
            return

        try:
            self.SelectItem(item)
            data = self.GetItemData(item)
        except RuntimeError:
            return
        
        if not data:
            return
        
        item_type, item_data = data
        
        if item_type == self.TYPE_WORKSPACE:
            self._show_workspace_menu(item, item_data, event.GetPoint())
        elif item_type == self.TYPE_CONVERSATION:
            self._show_conversation_menu(item, item_data, event.GetPoint())
    
    def _show_workspace_menu(self, item: wx.TreeItemId, workspace: Workspace, 
                             position: wx.Point):
        """Show context menu for a workspace."""
        menu = wx.Menu()
        
        # Count conversations for warning
        conv_count = len(workspace.conversations)
        
        menu.Append(wx.ID_ANY, f"Path: {workspace.path}")
        menu.AppendSeparator()
        
        id_new_default = wx.NewIdRef()
        menu.Append(id_new_default, f"New {workspace.get_default_workflow()}")

        id_new_workflow = wx.NewIdRef()
        menu.Append(id_new_workflow, "New Workflow...")
        menu.Bind(wx.EVT_MENU, lambda e: self._on_new_workflow(workspace), id=id_new_workflow)

        id_settings = wx.NewIdRef()
        menu.Append(id_settings, "Settings...")
        menu.Bind(wx.EVT_MENU, lambda e: self._on_workspace_settings(workspace), id=id_settings)
        menu.Bind(wx.EVT_MENU, lambda e: self._on_create_default_workflow(workspace), id=id_new_default)
        
        menu.AppendSeparator()
        
        id_remove = wx.NewIdRef()
        menu.Append(id_remove, "Remove from List")
        menu.Bind(wx.EVT_MENU, lambda e: self._on_remove_workspace(item, workspace), id=id_remove)
        
        id_delete = wx.NewIdRef()
        menu.Append(id_delete, f"Delete Workspace ({conv_count} workflows)")
        menu.Bind(wx.EVT_MENU, lambda e: self._on_delete_workspace(item, workspace), id=id_delete)
        
        self.PopupMenu(menu, position)
        menu.Destroy()
    
    def _show_conversation_menu(self, item: wx.TreeItemId, stub: ConversationStub,
                                position: wx.Point):
        """Show context menu for a conversation."""
        menu = wx.Menu()

        id_delete = wx.NewIdRef()
        menu.Append(id_delete, f"Delete {stub.workflow_name}")
        menu.Bind(wx.EVT_MENU,
                 lambda e: self._on_delete_conversation(item, stub),
                 id=id_delete)

        self.PopupMenu(menu, position)
        menu.Destroy()

    def _on_new_workflow(self, workspace: Workspace):
        """Handle New Workflow from context menu - notify callback."""
        if self.on_new_workflow:
            self.on_new_workflow(workspace)

    def _on_create_default_workflow(self, workspace: Workspace):
        """Create the workspace's default workflow via callback."""
        if self.on_create_workflow:
            self.on_create_workflow(workspace)

    def _on_workspace_settings(self, workspace: Workspace):
        """Open the workspace settings tab via callback."""
        if self.on_workspace_settings:
            self.on_workspace_settings(workspace)
    
    def _on_remove_workspace(self, item: wx.TreeItemId, workspace: Workspace):
        """Remove workspace from list (doesn't delete files)."""
        # Refuse if any conversation in this workspace has an open tab: the
        # tab owns the Conversation object, so removing the workspace under it
        # would orphan the tab's conversation reference.
        if self.on_check_open_tab:
            for conv_id in workspace.get_conversation_ids():
                if self.on_check_open_tab(conv_id):
                    dlg = wx.MessageDialog(
                        self,
                        f"Cannot remove workspace '{workspace.name}' because one or more "
                        "workflows have open tabs. Close all tabs for this workspace first.",
                        "Cannot Remove Workspace",
                        wx.OK | wx.ICON_INFORMATION
                    )
                    dlg.ShowModal()
                    dlg.Destroy()
                    return
        self.manager.remove_workspace(workspace.workspace_id)
        self.Delete(item)
    
    def _on_delete_workspace(self, item: wx.TreeItemId, workspace: Workspace):
        """Delete workspace and all its conversations."""
        # Check if any conversation in this workspace has an open tab
        if self.on_check_open_tab:
            for conv_id in workspace.get_conversation_ids():
                if self.on_check_open_tab(conv_id):
                    dlg = wx.MessageDialog(
                        self,
                        f"Cannot delete workspace '{workspace.name}' because one or more "
                        "workflows have open tabs. Close all tabs for this workspace first.",
                        "Cannot Delete Workspace",
                        wx.OK | wx.ICON_INFORMATION
                    )
                    dlg.ShowModal()
                    dlg.Destroy()
                    return

        conv_count = len(workspace.conversations)

        dlg = wx.MessageDialog(
            self,
            f"This will permanently delete the workspace '{workspace.name}' "
            f"and all {conv_count} workflow(s) within it.\n\n"
            f"Path: {workspace.path}\n\n"
            "This action cannot be undone. Are you sure?",
            "Delete Workspace?",
            wx.YES_NO | wx.ICON_WARNING
        )

        if dlg.ShowModal() == wx.ID_YES:
            # Delete workspace (handles files and removal from manager)
            self.manager.delete_workspace(workspace.workspace_id)
            self.Delete(item)

        dlg.Destroy()
    
    def _on_delete_conversation(self, item: wx.TreeItemId, stub: ConversationStub):
        """Delete a conversation."""
        # Check if this conversation has an open tab
        if self.on_check_open_tab and self.on_check_open_tab(stub.conversation_id):
            dlg = wx.MessageDialog(
                self,
                "Cannot delete a workflow that has an open tab. Close the tab first.",
                "Cannot Delete",
                wx.OK | wx.ICON_INFORMATION
            )
            dlg.ShowModal()
            dlg.Destroy()
            return

        dlg = wx.MessageDialog(
            self,
            f"Delete this {stub.workflow_name} workflow? This action cannot be undone.",
            f"Delete {stub.workflow_name}?",
            wx.YES_NO | wx.ICON_WARNING
        )

        if dlg.ShowModal() == wx.ID_YES:
            # Find the workspace
            for workspace in self.manager.get_workspaces():
                if stub.conversation_id in workspace.conversations:
                    workspace.delete_conversation(stub.conversation_id)
                    break
            self.Delete(item)

        dlg.Destroy()


class LeftPanel(wx.Panel):
    """Left panel containing tabbed workspace/workflow management."""

    def __init__(self, parent, manager: WorkspaceManager, config_mgr: WorkflowConfigManager = None):
        super().__init__(parent)
        self.manager = manager
        self.config_mgr = config_mgr
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.notebook = wx.Notebook(self)
        # --- Tab 1: Workspaces ---
        ws_panel = wx.Panel(self.notebook)
        ws_sizer = wx.BoxSizer(wx.VERTICAL)
        toolbar = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_add_workspace = wx.Button(ws_panel, label="Add Workspace...")
        toolbar.Add(self.btn_add_workspace, 0, wx.RIGHT, 5)
        self.btn_add_workspace.Bind(wx.EVT_BUTTON, self._on_add_workspace)
        self.btn_new_workflow = wx.Button(ws_panel, label="New Workflow...")
        toolbar.Add(self.btn_new_workflow, 0)
        self.btn_new_workflow.Bind(wx.EVT_BUTTON, self._on_new_workflow_dialog)
        ws_sizer.Add(toolbar, 0, wx.EXPAND | wx.ALL, 5)
        ws_sizer.Add(wx.StaticLine(ws_panel), 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)
        self.tree = WorkspaceTreeCtrl(ws_panel, manager)
        ws_sizer.Add(self.tree, 1, wx.EXPAND | wx.ALL, 5)
        ws_panel.SetSizer(ws_sizer)
        self.notebook.AddPage(ws_panel, "Workspaces")
        # --- Tab 2: Workflows ---
        self.workflow_panel = None
        if config_mgr:
            self.workflow_panel = WorkflowPanel(self.notebook, config_mgr)
            self.notebook.AddPage(self.workflow_panel, "Workflows")
        sizer.Add(self.notebook, 1, wx.EXPAND)
        self.SetSizer(sizer)
        # Callbacks
        self.on_add_workspace = None
        self.on_new_workflow_dialog = None

    def _on_add_workspace(self, event: wx.Event):
        if self.on_add_workspace:
            self.on_add_workspace()

    def _on_new_workflow_dialog(self, event: wx.Event):
        if self.on_new_workflow_dialog:
            self.on_new_workflow_dialog()


    def set_callbacks(self, on_conversation_selected, on_workspace_selected,
                      on_add_workspace=None, on_new_workflow_dialog=None,
                      on_new_workflow=None, on_check_open_tab=None,
                      on_agent_double_click=None, on_agents_changed=None,
                      on_sequence_group_double_click=None, on_start_conversation=None,
                      on_create_workflow=None, on_workspace_settings=None):
        """Set callback functions."""
        self.tree.on_conversation_selected = on_conversation_selected
        self.tree.on_workspace_selected = on_workspace_selected
        self.on_add_workspace = on_add_workspace
        self.tree.on_new_workflow = on_new_workflow
        self.on_new_workflow_dialog = on_new_workflow_dialog
        self.tree.on_check_open_tab = on_check_open_tab
        self.tree.on_create_workflow = on_create_workflow
        self.tree.on_workspace_settings = on_workspace_settings
        if self.workflow_panel:
            self.workflow_panel.set_callbacks(
                on_agent_double_click=on_agent_double_click,
                on_agents_changed=on_agents_changed,
                on_sequence_group_double_click=on_sequence_group_double_click,
                on_start_conversation=on_start_conversation,
            )

    def add_workspace(self, workspace: Workspace):
        self.tree.add_workspace(workspace)

    def add_conversation(self, workspace: Workspace, stub: ConversationStub):
        self.tree.add_conversation(workspace, stub)
