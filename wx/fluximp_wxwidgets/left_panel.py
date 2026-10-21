"""Left panel - Tabbed workspace/workflow management for FluxImp GUI."""

import wx
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

from fluximp_wxwidgets.workspace import WorkspaceManager, Workspace, ConversationStub
from fluximp_wxwidgets.conversation_manager import Conversation
from fluximp_wxwidgets.conversation_grouping import (
    group_conversations_by_month,
    month_label,
)
from fluximp_wxwidgets.workflow_config import WorkflowConfigManager
from fluximp_wxwidgets.workflow_panel import WorkflowPanel
from fluximp_wxwidgets.icon_bitmaps import icon_bundle


class WorkspaceTreeCtrl(wx.TreeCtrl):
    """Tree control for displaying workspaces and conversations."""
    
    # Tree item data types
    TYPE_WORKSPACE = "workspace"
    TYPE_CONVERSATION = "conversation"
    TYPE_MONTH = "month"
    
    def __init__(self, parent, manager: WorkspaceManager):
        super().__init__(parent, style=wx.TR_DEFAULT_STYLE | wx.TR_HIDE_ROOT)
        
        self.manager = manager
        
        # Image list for icons
        # Icons: transparent and DPI-aware (see icon_bitmaps).  SetImages
        # takes a wx.BitmapBundle per image, so the tree scales them for a
        # HiDPI display instead of showing a fixed 16 physical pixels; the
        # indices below are what AppendItem's image= parameter takes.
        self.SetImages([
            icon_bundle(self._draw_folder),
            icon_bundle(self._draw_chat),
            icon_bundle(self._draw_calendar),
        ])
        self.img_folder = 0
        self.img_chat = 1
        self.img_calendar = 2
        
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
    
    @staticmethod
    def _draw_folder(gc) -> None:
        """Paint the folder icon into a 16x16 logical box.

        No background is painted: the bitmap is transparent (see
        icon_bitmaps), so whatever is not drawn here shows the tree's own
        colour through, which is what lets one icon work on a light and on a
        dark tree.
        """
        gc.SetPen(wx.Pen(wx.Colour(180, 180, 180)))
        gc.SetBrush(wx.Brush(wx.Colour(255, 220, 100)))
        gc.DrawRoundedRectangle(2, 4, 12, 10, 2)

    @staticmethod
    def _draw_chat(gc) -> None:
        """Paint the chat icon into a 16x16 logical box (see _draw_folder)."""
        gc.SetPen(wx.Pen(wx.Colour(100, 150, 220)))
        gc.SetBrush(wx.Brush(wx.Colour(150, 200, 255)))
        gc.DrawEllipse(2, 2, 12, 10)
    
    @staticmethod
    def _draw_calendar(gc) -> None:
        """Paint a calendar icon into a 16x16 logical box (see _draw_folder).

        Used for the YYYY-MM month nodes that collect older conversations.
        """
        gc.SetPen(wx.Pen(wx.Colour(150, 150, 150)))
        gc.SetBrush(wx.Brush(wx.Colour(235, 235, 240)))
        gc.DrawRoundedRectangle(2, 3, 12, 11, 2)
        # Header band and the two binding rings above it.
        gc.SetBrush(wx.Brush(wx.Colour(150, 170, 220)))
        gc.DrawRectangle(3, 4, 10, 3)
        gc.SetPen(wx.Pen(wx.Colour(150, 150, 150)))
        gc.StrokeLine(5, 2, 5, 4)
        gc.StrokeLine(11, 2, 11, 4)

    def _refresh_tree(self):
        """Refresh the entire tree from the manager."""
        self.DeleteChildren(self.root)
        
        for workspace in self.manager.get_workspaces():
            self._add_workspace_node(workspace)
    
    def _add_workspace_node(self, workspace: Workspace) -> wx.TreeItemId:
        """Add a workspace node and its conversations to the tree.

        Conversations are listed newest-first.  Recent ones (current UTC
        month, plus the previous UTC month early in the month) sit directly
        under the workspace; everything older is collected into collapsed
        YYYY-MM month nodes, newest month first.
        """
        item = self.AppendItem(self.root, workspace.name, image=self.img_folder)
        self.SetItemData(item, (self.TYPE_WORKSPACE, workspace))

        # Add conversation stubs (lightweight, not full conversations)
        stubs = []
        for conv_id in workspace.get_conversation_ids():
            stub = workspace.conversations.get(conv_id)
            if stub:
                stubs.append(stub)

        recent, month_groups = group_conversations_by_month(
            stubs, datetime.now(timezone.utc))

        for stub in recent:
            self._add_conversation_node(item, stub)

        for key, month_stubs in month_groups:
            month_item = self._add_month_node(item, key, len(month_stubs))
            for stub in month_stubs:
                self._add_conversation_node(month_item, stub)

        return item
    
    def _conversation_label(self, stub: ConversationStub) -> str:
        """Return the tree label for a conversation stub."""
        timestamp = stub.start_time.strftime("%Y-%m-%d %H:%M")
        return f"{stub.workflow_name} {timestamp}"

    def _add_conversation_node(self, parent: wx.TreeItemId,
                                stub: ConversationStub) -> wx.TreeItemId:
        """Add a conversation stub node under a workspace or month node."""
        item = self.AppendItem(parent, self._conversation_label(stub),
                               image=self.img_chat)
        self.SetItemData(item, (self.TYPE_CONVERSATION, stub))
        return item

    def _add_month_node(self, parent: wx.TreeItemId, key: str,
                        count: int) -> wx.TreeItemId:
        """Add a collapsed YYYY-MM month node under a workspace."""
        item = self.AppendItem(parent, month_label(key, count),
                               image=self.img_calendar)
        self.SetItemData(item, (self.TYPE_MONTH, key))
        # AppendItem leaves the node collapsed, which is what we want.
        return item
    
    def add_workspace(self, workspace: Workspace):
        """Add a new workspace to the tree."""
        self._add_workspace_node(workspace)
    
    def add_conversation(self, workspace: Workspace, stub: ConversationStub):
        """Add a new conversation stub to the tree.

        A freshly created conversation is always in the current month, so it
        is a "recent" entry and belongs at the very top of the workspace's
        children - above any older recent entries and above the month nodes.
        """
        # Find the workspace node
        item, cookie = self.GetFirstChild(self.root)
        while item.IsOk():
            data = self.GetItemData(item)
            if data and data[0] == self.TYPE_WORKSPACE and data[1] == workspace:
                new_item = self.PrependItem(item, self._conversation_label(stub),
                                            image=self.img_chat)
                self.SetItemData(new_item, (self.TYPE_CONVERSATION, stub))
                self.Expand(item)
                return
            item, cookie = self.GetNextChild(self.root, cookie)

    def remove_conversation(self, stub: ConversationStub):
        """Remove a conversation node from the tree."""
        item = self._find_conversation_item(stub)
        if item and item.IsOk():
            self._delete_conversation_node(item)

    def _delete_conversation_node(self, item: wx.TreeItemId):
        """Delete a conversation node, pruning a month node left empty.

        A conversation sits either directly under its workspace (a "recent"
        entry) or inside a YYYY-MM month node; in the latter case an emptied
        month node is removed too, so the tree keeps no empty groups.
        """
        parent = self.GetItemParent(item)
        self.Delete(item)
        if parent.IsOk():
            parent_data = self.GetItemData(parent)
            if parent_data and parent_data[0] == self.TYPE_MONTH:
                first_child, _ = self.GetFirstChild(parent)
                if not first_child.IsOk():
                    self.Delete(parent)

    def _find_conversation_item(self, stub: ConversationStub) -> Optional[wx.TreeItemId]:
        """Find the tree item for a conversation stub.

        Conversation nodes live either directly under a workspace (the
        "recent" entries) or one level deeper inside a YYYY-MM month node,
        so both levels are searched.
        """
        item, cookie = self.GetFirstChild(self.root)
        while item.IsOk():
            child, child_cookie = self.GetFirstChild(item)
            while child.IsOk():
                data = self.GetItemData(child)
                if data and data[0] == self.TYPE_CONVERSATION and data[1] == stub:
                    return child
                if data and data[0] == self.TYPE_MONTH:
                    grandchild, gc_cookie = self.GetFirstChild(child)
                    while grandchild.IsOk():
                        gdata = self.GetItemData(grandchild)
                        if (gdata and gdata[0] == self.TYPE_CONVERSATION
                                and gdata[1] == stub):
                            return grandchild
                        grandchild, gc_cookie = self.GetNextChild(child, gc_cookie)
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
        # TYPE_MONTH nodes are containers only: no context menu.
    
    def _show_workspace_menu(self, item: wx.TreeItemId, workspace: Workspace, 
                             position: wx.Point):
        """Show context menu for a workspace."""
        menu = wx.Menu()
        
        # Count conversations for warning
        conv_count = len(workspace.conversations)
        
        id_new_default = wx.NewIdRef()
        menu.Append(id_new_default, f"New {workspace.get_default_workflow()}")

        id_new_workflow = wx.NewIdRef()
        menu.Append(id_new_workflow, "New Workflow...")
        menu.Bind(wx.EVT_MENU, lambda e: self._on_new_workflow(workspace), id=id_new_workflow)

        menu.AppendSeparator()

        id_settings = wx.NewIdRef()
        menu.Append(id_settings, "Settings...")
        menu.Bind(wx.EVT_MENU, lambda e: self._on_workspace_settings(workspace), id=id_settings)
        menu.Bind(wx.EVT_MENU, lambda e: self._on_create_default_workflow(workspace), id=id_new_default)

        id_remove = wx.NewIdRef()
        menu.Append(id_remove, "Remove from Tree")
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
        """Remove workspace from tree (doesn't delete files)."""
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
            self._delete_conversation_node(item)

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
