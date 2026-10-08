"""Event tab - Panel for editing conditional agent sequences for a workflow event."""

import wx
from typing import Optional, Callable, List, Dict
from copy import deepcopy

from fluximp_wxwidgets.workflow_config import (
    WorkflowConfigManager, ConditionalAgentConfigInfo, EventSequenceGroup
)


class EventTab(wx.Panel):
    """Tab for editing a conditional agent sequence for a specific event.

    All changes are in-memory only until the user presses Save.
    The sequence list shows agent names, and the editor section below
    shows the details of the currently selected agent.
    """

    def __init__(self, parent, workflow_name: str, event_key: str, group_name: str,
                 config_mgr: WorkflowConfigManager,
                 on_save_callback: Optional[Callable[[], None]] = None):
        super().__init__(parent)
        self.workflow_name = workflow_name
        self.event_key = event_key
        self.group_name = group_name
        self.config_mgr = config_mgr
        self.on_save_callback = on_save_callback
        self._dirty = False

        # Load current state from config manager into local in-memory copies
        group = config_mgr.get_event_sequence_group(workflow_name, event_key, group_name)
        self._agent_sequence: List[str] = list(group.sequence) if group else []
        self._agent_types: List[str] = list(group.agent_types) if group else []
        # Deep copy of conditional agent configs for in-memory editing
        self._agents: Dict[str, ConditionalAgentConfigInfo] = {}
        for name in self._agent_sequence:
            agent = config_mgr.get_conditional_agent(name)
            if agent:
                self._agents[name] = deepcopy(agent)
            else:
                # Agent referenced in sequence but no config exists yet
                self._agents[name] = ConditionalAgentConfigInfo(name=name)

        # Track original names for rename detection on save
        self._original_names: Dict[str, str] = {
            name: name for name in self._agent_sequence
        }

        # Track new agents not yet saved to disk
        self._new_agents: set = set()

        # Track which agent the editor is currently showing
        # (differs from list selection during selection transitions)
        self._current_agent_name: Optional[str] = None

        self._build_ui()
        self._populate_list()
        self._update_editor()

    def _build_ui(self):
        """Build the UI layout."""
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # --- Group Name ---
        name_sizer = wx.BoxSizer(wx.HORIZONTAL)
        name_label = wx.StaticText(self, label="Group Name:")
        self.group_name_ctrl = wx.TextCtrl(self, value=self.group_name)
        self.group_name_ctrl.Bind(wx.EVT_TEXT, self._on_field_changed)
        name_sizer.Add(name_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        name_sizer.Add(self.group_name_ctrl, 1, wx.EXPAND)
        main_sizer.Add(name_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 10)

        # --- Agent Types ---
        types_label = wx.StaticText(self, label="Agent Types:")
        main_sizer.Add(types_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)
        workflow = self.config_mgr.get_workflow(self.workflow_name)
        agent_type_names = list(workflow.agents) if workflow else []
        self.agent_types_checklist = wx.CheckListBox(self, choices=agent_type_names)
        for i, name in enumerate(agent_type_names):
            if name in self._agent_types:
                self.agent_types_checklist.Check(i)
        self.agent_types_checklist.Bind(wx.EVT_CHECKLISTBOX, self._on_field_changed)
        main_sizer.Add(self.agent_types_checklist, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)

        # --- Top section: Sequence list ---
        list_label = wx.StaticText(self, label="Agent Sequence:")
        font = list_label.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        list_label.SetFont(font)
        main_sizer.Add(list_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)

        list_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # List control
        self.agent_list = wx.ListCtrl(
            self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.LC_NO_HEADER
        )
        self.agent_list.InsertColumn(0, "Agent", width=250)
        self.agent_list.Bind(wx.EVT_LIST_ITEM_SELECTED, self._on_list_selection_changed)
        self.agent_list.Bind(wx.EVT_LIST_ITEM_DESELECTED, self._on_list_selection_changed)
        list_sizer.Add(self.agent_list, 1, wx.EXPAND | wx.RIGHT, 5)

        # Button column
        btn_sizer = wx.BoxSizer(wx.VERTICAL)
        self.btn_up = wx.Button(self, label="Up")
        self.btn_down = wx.Button(self, label="Down")
        self.btn_add = wx.Button(self, label="Add New")
        self.btn_add_existing = wx.Button(self, label="Add Existing")
        self.btn_remove = wx.Button(self, label="Remove")

        self.btn_up.Bind(wx.EVT_BUTTON, self._on_up)
        self.btn_down.Bind(wx.EVT_BUTTON, self._on_down)
        self.btn_add.Bind(wx.EVT_BUTTON, self._on_add)
        self.btn_add_existing.Bind(wx.EVT_BUTTON, self._on_add_existing)
        self.btn_remove.Bind(wx.EVT_BUTTON, self._on_remove)

        btn_sizer.Add(self.btn_up, 0, wx.EXPAND | wx.BOTTOM, 3)
        btn_sizer.Add(self.btn_down, 0, wx.EXPAND | wx.BOTTOM, 3)
        btn_sizer.Add(self.btn_add, 0, wx.EXPAND | wx.BOTTOM, 3)
        btn_sizer.Add(self.btn_add_existing, 0, wx.EXPAND | wx.BOTTOM, 3)
        btn_sizer.Add(self.btn_remove, 0, wx.EXPAND)
        list_sizer.Add(btn_sizer, 0, wx.ALIGN_TOP)

        main_sizer.Add(list_sizer, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)

        # --- Separator ---
        main_sizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.ALL, 10)

        # --- Bottom section: Agent editor ---
        editor_label = wx.StaticText(self, label="Agent Details:")
        editor_label.SetFont(font)
        main_sizer.Add(editor_label, 0, wx.LEFT | wx.RIGHT, 10)

        # Editor panel (shown when an agent is selected)
        self.editor_panel = wx.Panel(self)
        editor_sizer = wx.FlexGridSizer(cols=2, gap=(5, 10))
        editor_sizer.AddGrowableCol(1, 1)

        # Name
        editor_sizer.Add(wx.StaticText(self.editor_panel, label="Name:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.name_ctrl = wx.TextCtrl(self.editor_panel, style=wx.TE_PROCESS_ENTER)
        self.name_ctrl.Bind(wx.EVT_TEXT, self._on_name_changed)
        editor_sizer.Add(self.name_ctrl, 1, wx.EXPAND)

        # Tool Set
        editor_sizer.Add(wx.StaticText(self.editor_panel, label="Tool Set:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.tool_set_choice = wx.Choice(
            self.editor_panel,
            choices=["Report only", "Read-only filesystem"]
        )
        self.tool_set_choice.Bind(wx.EVT_CHOICE, self._on_field_changed)
        # Map display names to internal values
        self._tool_set_map = {
            0: "report_only",
            1: "readonly_filesystem",
        }
        self._tool_set_reverse = {v: k for k, v in self._tool_set_map.items()}
        editor_sizer.Add(self.tool_set_choice, 1, wx.EXPAND)

        # Depends on
        editor_sizer.Add(wx.StaticText(self.editor_panel, label="Depends on:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.depends_choice = wx.Choice(self.editor_panel, choices=["(None)"])
        self.depends_choice.Bind(wx.EVT_CHOICE, self._on_field_changed)
        editor_sizer.Add(self.depends_choice, 1, wx.EXPAND)

        # Max API Requests
        editor_sizer.Add(wx.StaticText(self.editor_panel, label="Max API Requests:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.max_requests_spin = wx.SpinCtrl(self.editor_panel, min=1, max=100, initial=5)
        self.max_requests_spin.Bind(wx.EVT_SPINCTRL, self._on_field_changed)
        editor_sizer.Add(self.max_requests_spin, 1, wx.EXPAND)

        # Retrigger
        editor_sizer.Add(wx.StaticText(self.editor_panel, label="Trigger until empty report:"), 0, wx.ALIGN_CENTER_VERTICAL)
        retrigger_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.retrigger_check = wx.CheckBox(self.editor_panel, label="")
        self.retrigger_check.Bind(wx.EVT_CHECKBOX, self._on_field_changed)
        retrigger_sizer.Add(self.retrigger_check, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        retrigger_sizer.Add(wx.StaticText(self.editor_panel, label="Max:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.max_retriggers_spin = wx.SpinCtrl(self.editor_panel, min=1, max=100, initial=3)
        self.max_retriggers_spin.Bind(wx.EVT_SPINCTRL, self._on_field_changed)
        retrigger_sizer.Add(self.max_retriggers_spin, 0, wx.ALIGN_CENTER_VERTICAL)
        editor_sizer.Add(retrigger_sizer, 1, wx.EXPAND)

        # Context Sources
        editor_sizer.Add(wx.StaticText(self.editor_panel, label="Context Sources:"), 0, wx.ALIGN_CENTER_VERTICAL)
        ctx_sizer = wx.BoxSizer(wx.VERTICAL)
        self.ctx_user_assistant = wx.CheckBox(self.editor_panel, label="User & Assistant Messages")
        self.ctx_user_assistant.Bind(wx.EVT_CHECKBOX, self._on_field_changed)
        ctx_sizer.Add(self.ctx_user_assistant, 0, wx.BOTTOM, 2)
        self.ctx_diff = wx.CheckBox(self.editor_panel, label="Diff")
        self.ctx_diff.Bind(wx.EVT_CHECKBOX, self._on_field_changed)
        ctx_sizer.Add(self.ctx_diff, 0, wx.BOTTOM, 2)
        self.ctx_prior_agent = wx.CheckBox(self.editor_panel, label="Prior Agent Message")
        self.ctx_prior_agent.Bind(wx.EVT_CHECKBOX, self._on_field_changed)
        ctx_sizer.Add(self.ctx_prior_agent, 0)
        editor_sizer.Add(ctx_sizer, 1, wx.EXPAND)

        self.editor_panel.SetSizer(editor_sizer)
        main_sizer.Add(self.editor_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)

        # Prompt
        prompt_label = wx.StaticText(self, label="Prompt:")
        prompt_label.SetFont(font)
        main_sizer.Add(prompt_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)
        self.prompt_ctrl = wx.TextCtrl(
            self, style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER,
            size=(-1, 150)
        )
        self.prompt_ctrl.Bind(wx.EVT_TEXT, self._on_field_changed)
        main_sizer.Add(self.prompt_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)

        # Save button
        self.btn_save = wx.Button(self, label="Save")
        self.btn_save.Bind(wx.EVT_BUTTON, self._on_save)
        self.btn_save.Enable(False)
        main_sizer.Add(self.btn_save, 0, wx.ALIGN_RIGHT | wx.ALL, 10)

        self.SetSizer(main_sizer)

    def _populate_list(self):
        """Populate the agent list from the in-memory sequence."""
        self.agent_list.DeleteAllItems()
        for i, name in enumerate(self._agent_sequence):
            self.agent_list.InsertItem(i, name)

    def _get_selected_index(self) -> int:
        """Get the index of the currently selected list item, or -1."""
        return self.agent_list.GetNextItem(-1, wx.LIST_NEXT_ALL, wx.LIST_STATE_SELECTED)

    def _get_selected_agent_name(self) -> Optional[str]:
        """Get the name of the currently selected agent, or None."""
        idx = self._get_selected_index()
        if idx >= 0 and idx < len(self._agent_sequence):
            return self._agent_sequence[idx]
        return None

    def _update_editor(self):
        """Update the editor section to reflect the currently selected agent."""
        name = self._get_selected_agent_name()
        if name is None:
            self._current_agent_name = None
            self.editor_panel.Disable()
            self.prompt_ctrl.Disable()
            self.name_ctrl.SetValue("")
            self.prompt_ctrl.SetValue("")
            return

        self._current_agent_name = name
        self.editor_panel.Enable()
        self.prompt_ctrl.Enable()

        agent = self._agents.get(name)
        if agent is None:
            agent = ConditionalAgentConfigInfo(name=name)
            self._agents[name] = agent

        # Temporarily unbind name change to avoid feedback loop
        self.name_ctrl.Unbind(wx.EVT_TEXT)
        self.name_ctrl.SetValue(agent.name)
        self.name_ctrl.Bind(wx.EVT_TEXT, self._on_name_changed)

        # Tool set
        tool_set_idx = self._tool_set_reverse.get(agent.tool_set, 0)
        self.tool_set_choice.SetSelection(tool_set_idx)

        # Depends on
        self._update_depends_dropdown(name, agent.depends_on)

        # Max API requests
        self.max_requests_spin.SetValue(agent.max_requests)

        # Retrigger
        self.retrigger_check.SetValue(agent.retrigger)
        self.max_retriggers_spin.SetValue(agent.max_retriggers)
        self.max_retriggers_spin.Enable(agent.retrigger)

        # Context sources
        sources = agent.context_sources or []
        self.ctx_user_assistant.SetValue("user_and_assistant_messages" in sources)
        self.ctx_diff.SetValue("diff" in sources)
        self.ctx_prior_agent.SetValue("prior_agent_message" in sources)

        # Prompt
        self.prompt_ctrl.Unbind(wx.EVT_TEXT)
        self.prompt_ctrl.SetValue(agent.prompt)
        self.prompt_ctrl.Bind(wx.EVT_TEXT, self._on_field_changed)

    def _update_depends_dropdown(self, current_name: str, current_depends: Optional[str] = None):
        """Update the depends_on dropdown with other agents in the sequence."""
        choices = ["(None)"]
        other_names = [n for n in self._agent_sequence if n != current_name]
        choices.extend(other_names)
        self.depends_choice.Clear()
        for choice in choices:
            self.depends_choice.Append(choice)

        if current_depends and current_depends in other_names:
            self.depends_choice.SetSelection(choices.index(current_depends))
        else:
            self.depends_choice.SetSelection(0)  # (None)

    def _save_current_agent(self):
        """Save the current editor values back to the in-memory agent config.
        
        Uses _current_agent_name (the agent the editor is displaying)
        rather than the list selection, because during selection transitions
        the list has already moved to the new item while the editor still
        shows the old agent's values.
        """
        name = self._current_agent_name
        if name is None:
            return

        agent = self._agents.get(name)
        if agent is None:
            return

        agent.tool_set = self._tool_set_map.get(self.tool_set_choice.GetSelection(), "report_only")

        dep_idx = self.depends_choice.GetSelection()
        dep_text = self.depends_choice.GetString(dep_idx) if dep_idx >= 0 else "(None)"
        agent.depends_on = None if dep_text == "(None)" else dep_text

        agent.retrigger = self.retrigger_check.IsChecked()
        agent.max_retriggers = self.max_retriggers_spin.GetValue()
        agent.max_requests = self.max_requests_spin.GetValue()

        sources = []
        if self.ctx_user_assistant.IsChecked():
            sources.append("user_and_assistant_messages")
        if self.ctx_diff.IsChecked():
            sources.append("diff")
        if self.ctx_prior_agent.IsChecked():
            sources.append("prior_agent_message")
        agent.context_sources = sources

        agent.prompt = self.prompt_ctrl.GetValue()

    def _get_checked_agent_types(self) -> List[str]:
        """Get the list of checked agent types from the CheckListBox."""
        checked = []
        for i in range(self.agent_types_checklist.GetCount()):
            if self.agent_types_checklist.IsChecked(i):
                checked.append(self.agent_types_checklist.GetString(i))
        return checked

    def _mark_dirty(self):
        """Mark the tab as having unsaved changes."""
        self._dirty = True
        self.btn_save.Enable(True)

    # --- Event handlers ---

    def _on_list_selection_changed(self, event=None):
        """Handle list selection change - save current agent and update editor."""
        self._save_current_agent()
        self._update_editor()

    def _on_name_changed(self, event=None):
        """Handle name text change - update the list item live."""
        idx = self._get_selected_index()
        if idx < 0:
            return

        new_name = self.name_ctrl.GetValue().strip()
        old_name = self._agent_sequence[idx]

        if new_name and new_name != old_name:
            # Prevent renaming to a name that already exists in the sequence
            # (would overwrite the other agent's config in _agents dict)
            if new_name in self._agents and new_name != old_name:
                return  # Don't allow duplicate names in the sequence

            # Update the in-memory agent
            agent = self._agents.get(old_name)
            if agent:
                agent.name = new_name
                # Move in dict
                del self._agents[old_name]
                self._agents[new_name] = agent

            # Update sequence
            self._agent_sequence[idx] = new_name

            # Update current agent name tracking
            self._current_agent_name = new_name

            # Update original names tracking
            if old_name in self._original_names:
                self._original_names[new_name] = self._original_names.pop(old_name)

            # Update depends_on references in other agents
            for other_name, other_agent in self._agents.items():
                if other_agent.depends_on == old_name:
                    other_agent.depends_on = new_name

            # Update the list item text
            self.agent_list.SetItemText(idx, new_name)

            # Update depends_on dropdown for other agents
            self._update_depends_dropdown(new_name, agent.depends_on if agent else None)

            self._mark_dirty()

    def _on_field_changed(self, event=None):
        """Handle any field change - save current agent and mark dirty."""
        self._save_current_agent()

        # Update max_retriggers enable state based on retrigger checkbox
        self.max_retriggers_spin.Enable(self.retrigger_check.IsChecked())

        self._mark_dirty()

    def _on_up(self, event=None):
        """Move the selected agent up in the sequence."""
        idx = self._get_selected_index()
        if idx <= 0:
            return

        self._save_current_agent()

        # Swap in sequence
        self._agent_sequence[idx], self._agent_sequence[idx - 1] =             self._agent_sequence[idx - 1], self._agent_sequence[idx]

        self._populate_list()
        self.agent_list.Select(idx - 1)
        self._update_editor()
        self._mark_dirty()

    def _on_down(self, event=None):
        """Move the selected agent down in the sequence."""
        idx = self._get_selected_index()
        if idx < 0 or idx >= len(self._agent_sequence) - 1:
            return

        self._save_current_agent()

        # Swap in sequence
        self._agent_sequence[idx], self._agent_sequence[idx + 1] =             self._agent_sequence[idx + 1], self._agent_sequence[idx]

        self._populate_list()
        self.agent_list.Select(idx + 1)
        self._update_editor()
        self._mark_dirty()

    def _on_add(self, event=None):
        """Add a new conditional agent to the sequence with a default name."""
        self._save_current_agent()

        # Generate a unique default name
        counter = 1
        while True:
            name = f"New Step {counter}"
            # Check against in-memory agents and disk
            if name not in self._agents and not self.config_mgr.conditional_agent_exists(name):
                break
            counter += 1

        # Create in-memory agent config
        agent = ConditionalAgentConfigInfo(name=name)
        self._agents[name] = agent
        self._agent_sequence.append(name)
        self._original_names[name] = name
        self._new_agents.add(name)

        self._populate_list()
        self.agent_list.Select(len(self._agent_sequence) - 1)
        self._update_editor()
        self._mark_dirty()

    def _on_add_existing(self, event=None):
        """Add an existing conditional agent from disk to the sequence."""
        self._save_current_agent()

        # Get agents on disk that are not already in the sequence
        all_agents = self.config_mgr.get_conditional_agents()
        available = [a.name for a in all_agents if a.name not in self._agents]

        if not available:
            wx.MessageBox(
                "No existing conditional agents available to add.",
                "No Agents Available",
                wx.OK | wx.ICON_INFORMATION
            )
            return

        dlg = wx.SingleChoiceDialog(
            self,
            "Select a conditional agent to add:",
            "Add Existing Agent",
            available
        )
        if dlg.ShowModal() == wx.ID_OK:
            name = dlg.GetStringSelection()
            # Load from disk
            agent = self.config_mgr.get_conditional_agent(name)
            if agent:
                self._agents[name] = deepcopy(agent)
            else:
                self._agents[name] = ConditionalAgentConfigInfo(name=name)
            self._agent_sequence.append(name)
            self._original_names[name] = name

            self._populate_list()
            self.agent_list.Select(len(self._agent_sequence) - 1)
            self._update_editor()
            self._mark_dirty()
        dlg.Destroy()

    def _on_remove(self, event=None):
        """Remove the selected agent from the sequence (does NOT delete the config file)."""
        idx = self._get_selected_index()
        if idx < 0:
            return

        name = self._agent_sequence[idx]

        self._save_current_agent()

        # Remove from sequence
        self._agent_sequence.pop(idx)

        # Remove from in-memory agents
        if name in self._agents:
            del self._agents[name]

        # Remove from original names tracking
        if name in self._original_names:
            del self._original_names[name]

        # Remove from new agents tracking
        self._new_agents.discard(name)

        # Clear depends_on references in other agents
        for other_name, other_agent in self._agents.items():
            if other_agent.depends_on == name:
                other_agent.depends_on = None

        self._populate_list()

        # Select a new item if available
        if self._agent_sequence:
            new_idx = min(idx, len(self._agent_sequence) - 1)
            self.agent_list.Select(new_idx)
        self._update_editor()
        self._mark_dirty()

    def _on_save(self, event=None):
        """Save all changes to disk - workflow events and conditional agent configs."""
        # Save current editor state
        self._save_current_agent()

        # Validate names - check for duplicates and conflicts
        for name in self._agent_sequence:
            if not name.strip():
                wx.MessageBox("Agent name cannot be empty.", "Error", wx.OK | wx.ICON_ERROR)
                return

        # Check for name conflicts with existing agents on disk
        for name in self._agent_sequence:
            original_name = self._original_names.get(name, name)
            is_rename = (original_name != name)
            # Check if the new name conflicts with an existing agent on disk
            # (that isn't the one being renamed)
            if is_rename and self.config_mgr.conditional_agent_exists(name):
                wx.MessageBox(
                    f"Agent name '{name}' conflicts with an existing conditional agent.",
                    "Error", wx.OK | wx.ICON_ERROR
                )
                return

        # Handle renames first
        for name in self._agent_sequence:
            original_name = self._original_names.get(name, name)
            if original_name != name:
                # Rename the agent on disk (updates all workflow references)
                try:
                    self.config_mgr.rename_conditional_agent(original_name, name)
                except ValueError as e:
                    wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)
                    return

        # Save all conditional agent configs
        for name in self._agent_sequence:
            agent = self._agents.get(name)
            if agent:
                self.config_mgr.save_conditional_agent(agent)

        # Update the workflow's event sequence group
        workflow = self.config_mgr.get_workflow(self.workflow_name)
        if workflow:
            group = self.config_mgr.get_event_sequence_group(
                self.workflow_name, self.event_key, self.group_name)
            if group:
                new_name = self.group_name_ctrl.GetValue().strip()
                group.name = new_name
                group.agent_types = self._get_checked_agent_types()
                group.sequence = list(self._agent_sequence)
                self.group_name = new_name
            self.config_mgr._save_workflow_config(workflow)

        self._dirty = False
        self.btn_save.Enable(False)

        # Update original names tracking
        self._original_names = {name: name for name in self._agent_sequence}
        self._new_agents.clear()

        # Notify parent to refresh tree
        if self.on_save_callback:
            self.on_save_callback()
