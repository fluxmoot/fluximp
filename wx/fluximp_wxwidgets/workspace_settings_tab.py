"""Workspace settings tab - per-workspace settings shown in the center notebook."""

import wx
from typing import Optional

from fluximp_wxwidgets.workspace import Workspace
from fluximp_wxwidgets.workflow_config import WorkflowConfigManager, STOCK_WORKFLOW_NAME


class WorkspaceSettingsTab(wx.Panel):
    """Per-workspace settings panel shown as a center-notebook tab.

    Currently exposes:
      * the workspace's default workflow - the workflow created by the
        'New <workflow>' item on the workspace tree's context menu. The
        choice applies immediately when the combo selection changes
        (no OK button) and is persisted in the workspace's config.json.
      * the workspace's test-runner plugin directory - its raw text is
        processed exactly as it will be passed to agents
        (Workspace.resolve_plugin_dir) and verified to be an existing
        directory by the 'Check and Save' button before being persisted
        to the workspace's config.json. Agents are passed
        --test-runner-plugin-dir at their next launch, which enables the
        run_tests tool.
    """

    def __init__(self, parent, workspace: Workspace,
                 config_mgr: Optional[WorkflowConfigManager] = None):
        super().__init__(parent)
        self.workspace = workspace

        sizer = wx.BoxSizer(wx.VERTICAL)

        # Read-only workspace info
        sizer.Add(wx.StaticText(self, label=f"Workspace: {workspace.name}"),
                  0, wx.ALL, 10)
        sizer.Add(wx.StaticText(self, label=f"Path: {workspace.path}"),
                  0, wx.LEFT | wx.RIGHT, 10)

        sizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.ALL, 10)

        # Default workflow selection
        row_sizer = wx.BoxSizer(wx.HORIZONTAL)
        row_sizer.Add(wx.StaticText(self, label="Default workflow:"),
                      0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)

        # Populate from the Workflows tab's workflow list
        names = [wf.name for wf in config_mgr.get_workflows()] if config_mgr else []

        # If the stored default is no longer defined (e.g. the workflow was
        # deleted from the Workflows tab), show it rather than silently
        # hiding it, so the user can see and fix the setting.
        current = workspace.get_default_workflow()
        if current not in names:
            names.insert(0, current)

        self.workflow_combo = wx.ComboBox(self, choices=names, style=wx.CB_READONLY)
        if names:
            self.workflow_combo.SetSelection(names.index(current))
        else:
            self.workflow_combo.Enable(False)
        self.workflow_combo.Bind(wx.EVT_COMBOBOX, self._on_default_workflow_changed)
        row_sizer.Add(self.workflow_combo, 1, wx.EXPAND)
        sizer.Add(row_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)

        sizer.Add(wx.StaticText(
            self,
            label=f"Created by the 'New {current}' item on the "
                  f"workspace context menu. Stock workflow: {STOCK_WORKFLOW_NAME}."
        ), 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)

        sizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.ALL, 10)

        # Test runner plugin directory (a workspace-level setting: passed
        # to agents as --test-runner-plugin-dir at their next launch,
        # which enables the agent's run_tests tool).
        plugin_row = wx.BoxSizer(wx.HORIZONTAL)
        plugin_row.Add(wx.StaticText(self, label="Test runner plugin directory:"),
                       0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        self.plugin_dir_text = wx.TextCtrl(self)
        self.plugin_dir_text.Bind(wx.EVT_TEXT, self._on_plugin_dir_text_changed)
        # Note: no ALIGN_CENTER_VERTICAL here - wxEXPAND overrides alignment
        # flags in box sizers and debug wx builds assert on the combination.
        plugin_row.Add(self.plugin_dir_text, 1, wx.EXPAND)
        self.browse_btn = wx.Button(self, label="Browse...")
        self.browse_btn.Bind(wx.EVT_BUTTON, self._on_browse_plugin_dir)
        plugin_row.Add(self.browse_btn, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.check_save_btn = wx.Button(self, label="Check and Save")
        self.check_save_btn.Bind(wx.EVT_BUTTON, self._on_check_save_plugin_dir)
        plugin_row.Add(self.check_save_btn, 0, wx.ALIGN_CENTER_VERTICAL)
        sizer.Add(plugin_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 10)

        # Checked/saved status line.
        self.plugin_dir_status = wx.StaticText(self, label="")
        self._default_status_colour = self.plugin_dir_status.GetForegroundColour()
        self._fail_colour = wx.Colour(190, 0, 0)
        self._unchecked_colour = wx.Colour(170, 105, 0)
        self._saved_colour = wx.Colour(0, 128, 64)
        sizer.Add(self.plugin_dir_status, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)

        sizer.Add(wx.StaticText(
            self,
            label="Passed to agents as --test-runner-plugin-dir; enables the "
                  "agent's run_tests tool. Applies to every conversation in "
                  "this workspace at its next agent launch. Leave blank to "
                  "disable."
        ), 0, wx.LEFT | wx.RIGHT, 10)

        # The agents the GUI launches run under this same interpreter, so
        # this mirrors the agent-side SDK check (main.py): without the SDK,
        # run_tests stays disabled even with a plugin directory set.
        try:
            from fluximp_agent.test_runner_plugin_manager import SDK_AVAILABLE
        except Exception:
            SDK_AVAILABLE = False
        if not SDK_AVAILABLE:
            sdk_warn = wx.StaticText(
                self,
                label="Warning: fluximp-agent-sdk is not installed; run_tests "
                      "stays disabled for agents even with a plugin directory set."
            )
            sdk_warn.SetForegroundColour(self._fail_colour)
            sizer.Add(sdk_warn, 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)

        # Populate the box with the saved raw value and show its check
        # state right away (opening the tab needs no Check and Save).
        # ChangeValue (not SetValue) avoids firing EVT_TEXT.
        self._saved_plugin_dir_raw: Optional[str] = self.workspace.get_test_runner_plugin_dir()
        if self._saved_plugin_dir_raw:
            self.plugin_dir_text.ChangeValue(self._saved_plugin_dir_raw)
        self._show_saved_state_status()
        self.SetSizer(sizer)

    def _on_default_workflow_changed(self, event: wx.CommandEvent):
        """Persist the chosen default workflow immediately."""
        name = event.GetString()
        if name:
            self.workspace.set_default_workflow(name)

    def _set_plugin_dir_status(self, text: str, colour: Optional[wx.Colour] = None) -> None:
        """Set the plugin-dir status line; None colour = default foreground."""
        self.plugin_dir_status.SetLabel(text)
        self.plugin_dir_status.SetForegroundColour(
            colour if colour is not None else self._default_status_colour)
        self.plugin_dir_status.Refresh()

    def _check_plugin_dir(self):
        """Process the box's content exactly as agent launch would, then disk-check it.

        Returns (resolved, ok): resolved is the Path the agent would be
        given (None for a blank box); ok is True for the blank/disabled
        state or when resolved is an existing directory.
        """
        resolved = self.workspace.resolve_plugin_dir(self.plugin_dir_text.GetValue())
        return resolved, resolved is None or resolved.is_dir()

    def _show_saved_state_status(self):
        """Status text/colour for 'current box content == persisted value'.

        Re-runs the same processing + directory check used by Check and
        Save so a path whose directory vanished after it was saved shows
        the red failed state while the stored config is left unchanged
        until the user fixes or clears it.
        """
        if not (self._saved_plugin_dir_raw or "").strip():
            self._set_plugin_dir_status("No plugin directory set (run_tests is disabled).")
            return
        resolved, ok = self._check_plugin_dir()
        if ok:
            self._set_plugin_dir_status(f"Checked and saved: '{resolved}'.", self._saved_colour)
        else:
            self._set_plugin_dir_status(
                f"Checked: not an existing directory: '{resolved}'. The saved value is unchanged.",
                self._fail_colour)

    def _on_plugin_dir_text_changed(self, event: wx.CommandEvent):
        """Box edited: flip the status to unchecked until Check and Save runs.

        Re-typing content identical to the checked-and-saved value restores
        its status instead.
        """
        if self.plugin_dir_text.GetValue() == (self._saved_plugin_dir_raw or ""):
            self._show_saved_state_status()
        else:
            self._set_plugin_dir_status("Not checked or saved yet.", self._unchecked_colour)
        event.Skip()

    def _on_browse_plugin_dir(self, event: wx.CommandEvent):
        """Choose a directory via dialog; fills the box (still unchecked)."""
        current = self.workspace.resolve_plugin_dir()
        default_path = (str(current) if current is not None and current.is_dir()
                        else str(self.workspace.path))
        with wx.DirDialog(self, "Choose test runner plugin directory",
                          defaultPath=default_path,
                          style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            # SetValue (not ChangeValue) fires EVT_TEXT, which flips the
            # status to 'Not checked or saved yet.' - Browse picks the
            # directory, Check and Save commits it.
            self.plugin_dir_text.SetValue(dlg.GetPath())

    def _on_check_save_plugin_dir(self, event: wx.CommandEvent):
        """Check the box's content, then save it - only if the check passes.

        Processing is done by Workspace.resolve_plugin_dir(), the exact
        same function agent launch uses, so what was checked is exactly
        what gets passed to the agent as --test-runner-plugin-dir.
        """
        resolved, ok = self._check_plugin_dir()
        if not ok:
            self._set_plugin_dir_status(
                f"Checked: not an existing directory: '{resolved}'. Not saved.",
                self._fail_colour)
            return
        # Persist exactly what the user entered (raw); blank -> None.
        raw = self.plugin_dir_text.GetValue()
        self.workspace.set_test_runner_plugin_dir(raw if raw.strip() else None)
        self._saved_plugin_dir_raw = self.workspace.get_test_runner_plugin_dir()
        if resolved is None:
            self._set_plugin_dir_status(
                "Checked and saved - no plugin directory (run_tests is disabled).")
        else:
            self._set_plugin_dir_status(f"Checked and saved: '{resolved}'.", self._saved_colour)

    def get_title(self) -> str:
        return f"Settings: {self.workspace.name}"
