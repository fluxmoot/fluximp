"""Agent tab - Panel for viewing and editing agent configuration."""

import wx
from typing import Optional, List

from fluximp_wxwidgets.workflow_config import WorkflowConfigManager, AgentConfigInfo


class AgentTab(wx.Panel):
    """Tab for viewing and editing an agent configuration."""

    def __init__(self, parent, agent_name: str, config_mgr: WorkflowConfigManager):
        super().__init__(parent)
        self.agent_name = agent_name
        self.config_mgr = config_mgr
        self._dirty = False

        sizer = wx.BoxSizer(wx.VERTICAL)

        # Name field
        name_sizer = wx.BoxSizer(wx.HORIZONTAL)
        name_sizer.Add(wx.StaticText(self, label="Name:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        self.name_ctrl = wx.TextCtrl(self, value=agent_name)
        agent = config_mgr.get_agent(agent_name)
        if agent and agent.is_stock:
            self.name_ctrl.SetEditable(False)
            self.name_ctrl.SetToolTip("Stock agents cannot be renamed")
        name_sizer.Add(self.name_ctrl, 1, wx.EXPAND)
        sizer.Add(name_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # Tools section: 4 groups inside an "Agent Tools" box
        tools_box = wx.StaticBox(self, label="Agent Tools")
        bold_font = tools_box.GetFont()
        bold_font.SetWeight(wx.FONTWEIGHT_BOLD)
        tools_sizer = wx.StaticBoxSizer(tools_box, wx.VERTICAL)

        note = wx.StaticText(tools_box, label=(
            "Tools with associated experiments will be unavailable to the agent "
            "even if checked here.  The tools checked here are the maximal set "
            "of tools that may be available to the agent."
        ))
        note.Wrap(560)
        tools_sizer.Add(note, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 10)

        self.tool_checks = {}
        seen = set()
        selected_tools = set(agent.tool_names) if agent else set()
        for group_title, group_tools in config_mgr.get_tool_groups():
            group_label = wx.StaticText(tools_box, label=group_title)
            group_label.SetFont(bold_font)
            tools_sizer.Add(group_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)
            grid = wx.GridSizer(cols=3, gap=(5, 5))
            for tool_name in group_tools:
                if tool_name in seen:
                    continue  # tool_checks keyed by name; no dup checkboxes
                seen.add(tool_name)
                cb = wx.CheckBox(tools_box, label=tool_name)
                cb.SetValue(tool_name in selected_tools)
                cb.Bind(wx.EVT_CHECKBOX, self._on_dirty)
                self.tool_checks[tool_name] = cb
                grid.Add(cb)
            tools_sizer.Add(grid, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        sizer.Add(tools_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # Agent description section
        desc_label = wx.StaticText(self, label="Agent Description:")
        desc_label.SetFont(bold_font)
        sizer.Add(desc_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)

        self.desc_ctrl = wx.TextCtrl(
            self, value=agent.agent_description if agent else "",
            style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER,
            size=(-1, 200)
        )
        self.desc_ctrl.Bind(wx.EVT_TEXT, self._on_dirty)
        sizer.Add(self.desc_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Save button
        self.btn_save = wx.Button(self, label="Save")
        self.btn_save.Bind(wx.EVT_BUTTON, self._on_save)
        self.btn_save.Enable(False)
        sizer.Add(self.btn_save, 0, wx.ALIGN_RIGHT | wx.ALL, 10)

        self.SetSizer(sizer)

    def _on_dirty(self, event=None):
        self._dirty = True
        self.btn_save.Enable(True)

    def _on_save(self, event=None):
        name = self.name_ctrl.GetValue().strip()
        if not name:
            wx.MessageBox("Agent name cannot be empty.", "Error", wx.OK | wx.ICON_ERROR)
            return
        tool_names = sorted(t for t, cb in self.tool_checks.items() if cb.IsChecked())
        agent_description = self.desc_ctrl.GetValue()
        try:
            self.config_mgr.save_agent(name, tool_names, agent_description)
            self._dirty = False
            self.btn_save.Enable(False)
            self.agent_name = name
        except ValueError as e:
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)

    def get_title(self) -> str:
        return f"Agent: {self.agent_name}"
