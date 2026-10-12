"""Right panel - Fork/turn grid for navigating dev workflow history."""

import wx
from fluximp_agent import fs_agents
from fluximp_wxwidgets.workflow_config import WorkflowConfigManager
import wx.grid
from typing import Optional, Callable, List

from fluximp_agent.model_config import get_provider_names, create_provider
from fluximp_agent.curated_models import filter_provider_names, filter_model_names
from fluximp_wxwidgets.app_settings import AppSettings
from fluximp_wxwidgets.conversation_manager import Conversation, ForkTurnInfo


class ForkTurnGrid(wx.grid.Grid):
    """Grid displaying fork/turn status for the current conversation."""

    # Column indices
    COL_FORK = 0
    COL_TURN = 1
    COL_PROMPT = 2
    COL_CACHED = 3
    COL_COMPLETION = 4
    COL_REASONING = 5
    COL_TIME = 6

    def __init__(self, parent):
        super().__init__(parent)

        # Initialize with single row, will be updated dynamically
        self.CreateGrid(1, 7)

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


class ForkTurnPanel(wx.Panel):
    """Panel containing the fork/turn grid."""

    def __init__(self, parent):
        super().__init__(parent)

        sizer = wx.BoxSizer(wx.VERTICAL)

        self.info_label = wx.StaticText(self, label="No workflow selected")
        sizer.Add(self.info_label, 0, wx.EXPAND | wx.ALL, 5)

        self.grid = ForkTurnGrid(self)
        sizer.Add(self.grid, 1, wx.EXPAND)

        self.SetSizer(sizer)

        self.conversation: Optional[Conversation] = None
        self.on_turn_selected: Optional[Callable[[int, int], None]] = None

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
        self._update_info()

    def _update_grid(self):
        """Update the grid in the main thread."""
        self.grid.update_from_conversation(self.conversation)
        self._update_info()

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
            self.agent_types = sorted(config_mgr.get_workflow_agent_names('Dev Workflow'))
        else:
            self.agent_types = sorted(fs_agents.AGENT_DEFINITIONS.keys())
        # Assert scrutinator is not in the GUI agent types
        assert 'scrutinator' not in self.agent_types, "scrutinator is for internal use only"

        sizer = wx.BoxSizer(wx.VERTICAL)
        self.fork_turn_panel = ForkTurnPanel(self)
        sizer.Add(self.fork_turn_panel, 1, wx.EXPAND)

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
        self.log_path_label.SetForegroundColour(wx.Colour(80, 80, 80))
        self.log_copy_button = wx.Button(self, label="Copy", size=(70, -1))
        self.log_copy_button.Bind(wx.EVT_BUTTON, self._on_copy_log_path)
        self.log_copy_button.Enable(False)
        log_sizer.Add(self.log_path_label, 1, wx.ALIGN_CENTER_VERTICAL)
        log_sizer.Add(self.log_copy_button, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 5)
        sizer.Add(log_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)
        sizer.Add(config_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Status indicator label
        self.status_label = wx.StaticText(self, label="")
        self.status_label.SetForegroundColour(wx.Colour(255, 140, 0))  # Orange color
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
            self.status_label.SetForegroundColour(wx.Colour(0, 0, 0))  # Black
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
            self.status_label.SetForegroundColour(wx.Colour(0, 0, 0))  # Black
            self.status_label.SetLabel("✓ Agent running (unattached)")
            return
        self.status_label.SetForegroundColour(wx.Colour(255, 140, 0))  # Orange
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
        input/Send lock — applied at tab sync (set_conversation) and
        forwarded live by the displayed tab (apply_displayed_tab_state,
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

    def apply_displayed_tab_state(self, send_enabled: bool):
        """Forward from the displayed tab: mirror its input/Send lock.

        Replaces the old per-conversation on_agent_active_changed binding,
        which carried no tab identity — a background tab's transition
        disabled these combos for whichever tab was displayed, and only a
        switch away/back resynced. The forwarding tab's recompute is the
        sole traffic now; the status refresh keeps the old handler's
        live-display behavior.
        """
        self.enable_config_controls(send_enabled)
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
            self.status_label.SetForegroundColour(wx.Colour(0, 0, 0))
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
            self.status_label.SetForegroundColour(wx.Colour(0, 0, 0))  # Black
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

            # Combos mirror the displayed tab's input/Send pair (§2.2): ask
            # the tab — its Send state carries the same gates the pair uses
            # (agent active, §7.7 sequence lock, non-resumable), while the
            # conversation's is_agent_active omits them and is stale during
            # live conditional runs.
            self.enable_config_controls(
                self.current_tab.get_send_enabled()
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
            workflow_name = 'Dev Workflow'
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
