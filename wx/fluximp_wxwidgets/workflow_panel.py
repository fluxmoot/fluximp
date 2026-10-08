"""Workflow panel - Tree control for managing workflows and agents."""

import wx
from typing import Optional, Callable

from fluximp_wxwidgets.workflow_config import (
    WorkflowConfigManager, WorkflowConfig, AgentConfigInfo,
    ConditionalAgentConfigInfo, EventSequenceGroup, STOCK_WORKFLOW_NAME
)



class WorkflowTreeCtrl(wx.TreeCtrl):
    """Tree control for displaying workflows and their agents."""

    TYPE_WORKFLOW = "workflow"
    TYPE_AGENT = "agent"
    TYPE_EVENT = "event"
    TYPE_SEQUENCE_GROUP = "sequence_group" 

    def __init__(self, parent, config_mgr: WorkflowConfigManager):
        super().__init__(parent, style=wx.TR_DEFAULT_STYLE | wx.TR_HIDE_ROOT)
        self.config_mgr = config_mgr
        self.img_list = wx.ImageList(16, 16)
        self.img_workflow = self.img_list.Add(self._create_workflow_bitmap())
        self.img_agent = self.img_list.Add(self._create_agent_bitmap())
        self.img_event = self.img_list.Add(self._create_event_bitmap())
        self.img_group = self.img_list.Add(self._create_sequence_group_bitmap())
        self.SetImageList(self.img_list)
        self.root = self.AddRoot("Workflows")
        self.Bind(wx.EVT_TREE_ITEM_ACTIVATED, self._on_item_activated)
        self.Bind(wx.EVT_TREE_ITEM_RIGHT_CLICK, self._on_item_right_click)
        self.on_agent_double_click: Optional[Callable[[str], None]] = None
        self.on_sequence_group_double_click: Optional[Callable[[str, str, str], None]] = None
        self.on_agents_changed: Optional[Callable[[], None]] = None
        self.on_start_conversation: Optional[Callable[[str], None]] = None
        self._refresh_tree()

    def _create_workflow_bitmap(self) -> wx.Bitmap:
        bmp = wx.Bitmap(16, 16)
        dc = wx.MemoryDC(bmp)
        dc.SetBackground(wx.Brush(wx.Colour(240, 240, 240)))
        dc.Clear()
        dc.SetPen(wx.Pen(wx.Colour(80, 120, 200)))
        dc.SetBrush(wx.Brush(wx.Colour(130, 170, 230)))
        dc.DrawRoundedRectangle(1, 1, 14, 14, 3)
        dc.SelectObject(wx.NullBitmap)
        return bmp


    def _create_agent_bitmap(self) -> wx.Bitmap:
        bmp = wx.Bitmap(16, 16)
        dc = wx.MemoryDC(bmp)
        dc.SetBackground(wx.Brush(wx.Colour(240, 240, 240)))
        dc.Clear()
        dc.SetPen(wx.Pen(wx.Colour(60, 60, 60)))
        dc.SetBrush(wx.Brush(wx.Colour(180, 180, 180)))
        dc.DrawCircle(8, 5, 4)
        dc.DrawRoundedRectangle(4, 10, 8, 5, 2)
        dc.SelectObject(wx.NullBitmap)
        return bmp

    def _create_event_bitmap(self) -> wx.Bitmap:
        bmp = wx.Bitmap(16, 16)
        dc = wx.MemoryDC(bmp)
        dc.SetBackground(wx.Brush(wx.Colour(240, 240, 240)))
        dc.Clear()
        dc.SetPen(wx.Pen(wx.Colour(200, 120, 0)))
        dc.SetBrush(wx.Brush(wx.Colour(255, 165, 0)))
        dc.DrawRoundedRectangle(2, 2, 12, 12, 2)
        dc.SelectObject(wx.NullBitmap)
        return bmp

    def _create_sequence_group_bitmap(self) -> wx.Bitmap:
        bmp = wx.Bitmap(16, 16)
        dc = wx.MemoryDC(bmp)
        dc.SetBackground(wx.Brush(wx.Colour(240, 240, 240)))
        dc.Clear()
        dc.SetPen(wx.Pen(wx.Colour(0, 120, 60)))
        dc.SetBrush(wx.Brush(wx.Colour(80, 180, 120)))
        dc.DrawRoundedRectangle(2, 2, 12, 12, 2)
        dc.SelectObject(wx.NullBitmap)
        return bmp

    def _refresh_tree(self):
        self.DeleteChildren(self.root)
        for workflow in self.config_mgr.get_workflows():
            self._add_workflow_node(workflow)


    def _add_workflow_node(self, workflow: WorkflowConfig) -> wx.TreeItemId:
        label = workflow.name
        if workflow.is_stock and workflow.is_modified_stock():
            label += " (modified)"
        item = self.AppendItem(self.root, label, image=self.img_workflow)
        self.SetItemData(item, (self.TYPE_WORKFLOW, workflow.name))
        for agent_name in workflow.agents:
            self._add_agent_node(item, agent_name)
        for event_key, groups in workflow.events.items():
            self._add_event_node(item, event_key, groups)
        return item


    def _add_agent_node(self, parent: wx.TreeItemId, agent_name: str) -> wx.TreeItemId:
        agent = self.config_mgr.get_agent(agent_name)
        label = agent_name
        if agent and agent.is_stock and agent.is_overridden():
            label += " (overridden)"
        item = self.AppendItem(parent, label, image=self.img_agent)
        self.SetItemData(item, (self.TYPE_AGENT, agent_name))
        return item

    def _add_event_node(self, parent: wx.TreeItemId, event_key: str, groups) -> wx.TreeItemId:
        label = event_key
        item = self.AppendItem(parent, label, image=self.img_event)
        parent_data = self.GetItemData(parent)
        workflow_name = parent_data[1] if parent_data else ""
        self.SetItemData(item, (self.TYPE_EVENT, (workflow_name, event_key)))
        for group in groups:
            self._add_sequence_group_node(item, workflow_name, event_key, group)
        return item

    def _add_sequence_group_node(self, parent: wx.TreeItemId, workflow_name: str,
                                  event_key: str, group) -> wx.TreeItemId:
        label = f"{group.name} ({len(group.sequence)})"
        item = self.AppendItem(parent, label, image=self.img_group)
        self.SetItemData(item, (self.TYPE_SEQUENCE_GROUP, (workflow_name, event_key, group.name)))
        return item

    def refresh_tree(self):
        self._refresh_tree()


    def _on_item_activated(self, event: wx.TreeEvent):
        item = event.GetItem()
        if not item.IsOk():
            return
        data = self.GetItemData(item)
        if not data:
            return
        item_type, item_data = data
        if item_type == self.TYPE_AGENT and self.on_agent_double_click:
            self.on_agent_double_click(item_data)
        elif item_type == self.TYPE_SEQUENCE_GROUP and self.on_sequence_group_double_click:
            workflow_name, event_key, group_name = item_data
            self.on_sequence_group_double_click(workflow_name, event_key, group_name)


    def _on_item_right_click(self, event: wx.TreeEvent):
        item = event.GetItem()
        if not item.IsOk():
            return
        self.SelectItem(item)
        data = self.GetItemData(item)
        if not data:
            return
        item_type, item_data = data
        if item_type == self.TYPE_WORKFLOW:
            self._show_workflow_menu(item, item_data, event.GetPoint())
        elif item_type == self.TYPE_AGENT:
            self._show_agent_menu(item, item_data, event.GetPoint())
        elif item_type == self.TYPE_EVENT:
            self._show_event_menu(item, item_data, event.GetPoint())
        elif item_type == self.TYPE_SEQUENCE_GROUP:
            self._show_sequence_group_menu(item, item_data, event.GetPoint())

    def _show_workflow_menu(self, item: wx.TreeItemId, workflow_name: str, position: wx.Point):
        menu = wx.Menu()
        id_start = wx.NewIdRef()
        menu.Append(id_start, "Start Conversation")
        menu.Bind(wx.EVT_MENU, lambda e: self._on_start_conversation(workflow_name), id=id_start)
        menu.AppendSeparator()
        id_new_agent = wx.NewIdRef()
        menu.Append(id_new_agent, "New Agent...")
        menu.Bind(wx.EVT_MENU, lambda e: self._on_new_agent(workflow_name), id=id_new_agent)
        id_add_agent = wx.NewIdRef()
        menu.Append(id_add_agent, "Add Agent...")
        menu.Bind(wx.EVT_MENU, lambda e: self._on_add_agent(workflow_name), id=id_add_agent)
        menu.AppendSeparator()
        if self.config_mgr.is_stock_workflow(workflow_name):
            id_restore = wx.NewIdRef()
            menu.Append(id_restore, "Restore Stock Workflow")
            menu.Bind(wx.EVT_MENU, lambda e: self._on_restore_stock_workflow(), id=id_restore)
        else:
            id_delete = wx.NewIdRef()
            menu.Append(id_delete, "Delete Workflow")
            menu.Bind(wx.EVT_MENU, lambda e: self._on_delete_workflow(workflow_name, item), id=id_delete)
        self.PopupMenu(menu, position)
        menu.Destroy()

    def _show_agent_menu(self, item: wx.TreeItemId, agent_name: str, position: wx.Point):
        menu = wx.Menu()
        parent_item = self.GetItemParent(item)
        parent_data = self.GetItemData(parent_item)
        if not parent_data:
            return
        _, workflow_name = parent_data
        if self.config_mgr.is_stock_agent(agent_name):
            id_restore = wx.NewIdRef()
            menu.Append(id_restore, "Restore Stock Agent")
            if not self.config_mgr.is_agent_overridden(agent_name):
                menu.Enable(id_restore, False)
            menu.Bind(wx.EVT_MENU, lambda e: self._on_restore_stock_agent(agent_name), id=id_restore)
        else:
            id_remove = wx.NewIdRef()
            menu.Append(id_remove, "Remove Agent")
            menu.Bind(wx.EVT_MENU, lambda e: self._on_remove_agent(workflow_name, agent_name), id=id_remove)
        self.PopupMenu(menu, position)
        menu.Destroy()

    def _show_event_menu(self, item: wx.TreeItemId, event_data: tuple, position: wx.Point):
        menu = wx.Menu()
        workflow_name, event_key = event_data
        id_add_group = wx.NewIdRef()
        menu.Append(id_add_group, "Add Conditional Agent Group...")
        menu.Bind(wx.EVT_MENU, lambda e: self._on_add_sequence_group(workflow_name, event_key), id=id_add_group)
        menu.AppendSeparator()
        id_delete = wx.NewIdRef()
        menu.Append(id_delete, "Delete Event")
        menu.Bind(wx.EVT_MENU, lambda e: self._on_delete_event(workflow_name, event_key), id=id_delete)
        self.PopupMenu(menu, position)
        menu.Destroy()

    def _on_delete_event(self, workflow_name: str, event_key: str):
        dlg = wx.MessageDialog(
            self,
            f"Delete event '{event_key}' from workflow '{workflow_name}'?\n"
            "Conditional agent configs will NOT be deleted.",
            "Delete Event?",
            wx.YES_NO | wx.ICON_WARNING
        )
        if dlg.ShowModal() == wx.ID_YES:
            self.config_mgr.remove_event(workflow_name, event_key)
            self._refresh_tree()
        dlg.Destroy()

    def _on_new_agent(self, workflow_name: str):
        dlg = wx.TextEntryDialog(self, "Enter a name for the new agent:", "New Agent", "")
        if dlg.ShowModal() == wx.ID_OK:
            name = dlg.GetValue().strip()
            if name:
                try:
                    self.config_mgr.create_agent(name)
                    self.config_mgr.add_agent_to_workflow(workflow_name, name)
                    self._refresh_tree()
                    if self.on_agents_changed:
                        self.on_agents_changed()
                    if self.on_agent_double_click:
                        self.on_agent_double_click(name)
                except ValueError as e:
                    wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)
        dlg.Destroy()

    def _on_add_agent(self, workflow_name: str):
        workflow = self.config_mgr.get_workflow(workflow_name)
        if workflow is None:
            return
        all_agents = self.config_mgr.get_agents()
        available = [a.name for a in all_agents if a.name not in workflow.agents]
        if not available:
            wx.MessageBox("No available agents to add.", "No Agents", wx.OK | wx.ICON_INFORMATION)
            return
        dlg = wx.SingleChoiceDialog(self, "Select an agent to add:", "Add Agent", available)
        if dlg.ShowModal() == wx.ID_OK:
            agent_name = dlg.GetStringSelection()
            self.config_mgr.add_agent_to_workflow(workflow_name, agent_name)
            self._refresh_tree()
            if self.on_agents_changed:
                self.on_agents_changed()
        dlg.Destroy()

    def _on_delete_workflow(self, workflow_name: str, item: wx.TreeItemId):
        dlg = wx.MessageDialog(self, f"Delete workflow '{workflow_name}'?", "Delete Workflow?", wx.YES_NO | wx.ICON_WARNING)
        if dlg.ShowModal() == wx.ID_YES:
            self.config_mgr.delete_workflow(workflow_name)
            self._refresh_tree()
            if self.on_agents_changed:
                self.on_agents_changed()
        dlg.Destroy()

    def _on_start_conversation(self, workflow_name: str):
        """Handle Start Conversation menu item - notify callback."""
        if self.on_start_conversation:
            self.on_start_conversation(workflow_name)

    def _on_restore_stock_workflow(self):
        self.config_mgr.restore_stock_workflow()
        self._refresh_tree()
        if self.on_agents_changed:
            self.on_agents_changed()

    def _on_restore_stock_agent(self, agent_name: str):
        self.config_mgr.restore_stock_agent(agent_name)
        self._refresh_tree()
        if self.on_agents_changed:
            self.on_agents_changed()

    def _on_remove_agent(self, workflow_name: str, agent_name: str):
        self.config_mgr.remove_agent_from_workflow(workflow_name, agent_name)
        self._refresh_tree()
        if self.on_agents_changed:
            self.on_agents_changed()

    def _show_sequence_group_menu(self, item: wx.TreeItemId, group_data: tuple, position: wx.Point):
        menu = wx.Menu()
        workflow_name, event_key, group_name = group_data
        id_delete = wx.NewIdRef()
        menu.Append(id_delete, "Delete Group")
        menu.Bind(wx.EVT_MENU, lambda e: self._on_delete_sequence_group(
            workflow_name, event_key, group_name), id=id_delete)
        self.PopupMenu(menu, position)
        menu.Destroy()

    def _on_add_sequence_group(self, workflow_name: str, event_key: str):
        dlg = wx.TextEntryDialog(
            self, "Enter a name for the new conditional agent group:",
            "Add Group", "")
        if dlg.ShowModal() == wx.ID_OK:
            group_name = dlg.GetValue().strip()
            if group_name:
                try:
                    self.config_mgr.add_event_sequence_group(workflow_name, event_key, group_name)
                    self._refresh_tree()
                except ValueError as e:
                    wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)
        dlg.Destroy()

    def _on_delete_sequence_group(self, workflow_name: str, event_key: str, group_name: str):
        dlg = wx.MessageDialog(
            self,
            f"Delete conditional agent group '{group_name}' from event '{event_key}'?\n"
            "Conditional agent configs will NOT be deleted.",
            "Delete Group?",
            wx.YES_NO | wx.ICON_WARNING
        )
        if dlg.ShowModal() == wx.ID_YES:
            self.config_mgr.remove_event_sequence_group(workflow_name, event_key, group_name)
            self._refresh_tree()
        dlg.Destroy()


class WorkflowPanel(wx.Panel):
    """Left panel tab for managing workflows and agents."""

    def __init__(self, parent, config_mgr: WorkflowConfigManager):
        super().__init__(parent)
        self.config_mgr = config_mgr
        sizer = wx.BoxSizer(wx.VERTICAL)
        toolbar = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_add_workflow = wx.Button(self, label="Add Workflow")
        toolbar.Add(self.btn_add_workflow, 0, wx.RIGHT, 5)
        self.btn_add_workflow.Bind(wx.EVT_BUTTON, self._on_add_workflow)
        sizer.Add(toolbar, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)
        self.tree = WorkflowTreeCtrl(self, config_mgr)
        sizer.Add(self.tree, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(sizer)

    def _on_add_workflow(self, event: wx.Event):
        dlg = wx.TextEntryDialog(self, "Enter a name for the new workflow:", "Add Workflow", "")
        if dlg.ShowModal() == wx.ID_OK:
            name = dlg.GetValue().strip()
            if name:
                try:
                    self.config_mgr.create_workflow(name)
                    self.tree.refresh_tree()
                except ValueError as e:
                    wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)
        dlg.Destroy()

    def set_callbacks(self, on_agent_double_click=None, on_agents_changed=None,
                      on_sequence_group_double_click=None, on_start_conversation=None):
        self.tree.on_agent_double_click = on_agent_double_click
        self.tree.on_agents_changed = on_agents_changed
        self.tree.on_sequence_group_double_click = on_sequence_group_double_click
        self.tree.on_start_conversation = on_start_conversation
