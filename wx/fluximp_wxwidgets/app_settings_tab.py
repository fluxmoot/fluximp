"""App settings tab - application-level (global) settings in the center notebook."""

import wx

from fluximp_wxwidgets.app_settings import AppSettings


class NoCuratedProvidersDialog(wx.Dialog):
    """Modal shown when the curated filter leaves no provider with an API key.

    The GUI can do nothing useful without a provider, so this mirrors the
    no-API-key startup gate (NoApiKeyDialog in main.py): the user either
    turns the curated filter off (wx.ID_YES) or exits the GUI (wx.ID_CANCEL).
    """

    def __init__(self, parent):
        super().__init__(parent, title="No Providers After Curated Filtering")

        sizer = wx.BoxSizer(wx.VERTICAL)

        msg = wx.StaticText(
            self,
            label="Curated filtering removed every provider that has an API "
                  "key set, so no provider is available.\n\n"
                  "Turn off curated filtering to use all providers with an "
                  "API key, or exit the application.",
        )
        msg.Wrap(420)
        sizer.Add(msg, 0, wx.ALL, 12)

        btns = wx.StdDialogButtonSizer()
        turn_off_btn = wx.Button(self, wx.ID_YES, "Turn Off Curated Filtering")
        turn_off_btn.SetDefault()
        btns.AddButton(turn_off_btn)
        btns.AddButton(wx.Button(self, wx.ID_CANCEL, "Exit"))
        btns.Realize()
        sizer.Add(btns, 0, wx.ALIGN_RIGHT | wx.ALL, 12)

        self.SetSizerAndFit(sizer)
        self.CentreOnScreen()


class AppSettingsTab(wx.Panel):
    """Application-level settings panel shown as a center-notebook tab.

    Currently exposes a single setting: whether the Provider: and Model:
    dropdowns are limited to the curated set (curated_models.py).  The
    change applies immediately (no OK button) and is persisted in
    base_dir/settings.json.
    """

    def __init__(self, parent, app_settings: AppSettings,
                 on_limit_to_curated_changed):
        super().__init__(parent)
        self.app_settings = app_settings
        # Callback returns True when the requested value was applied, False
        # when it was rejected (it would leave no usable provider).
        self.on_limit_to_curated_changed = on_limit_to_curated_changed

        sizer = wx.BoxSizer(wx.VERTICAL)

        self.limit_checkbox = wx.CheckBox(
            self, label="Limit providers and models to the curated set")
        self.limit_checkbox.SetValue(
            app_settings.limit_providers_and_models_to_curated)
        self.limit_checkbox.Bind(wx.EVT_CHECKBOX, self._on_limit_changed)
        sizer.Add(self.limit_checkbox, 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)

        sizer.Add(wx.StaticText(
            self,
            label="When enabled (the default), the Provider: and Model: "
                  "dropdowns show only the providers and models listed in "
                  "curated_models.py. Uncheck to show every provider that has "
                  "an API key set and all of that provider's models."
        ), 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)

        self.SetSizer(sizer)

    def _on_limit_changed(self, event: wx.CommandEvent):
        """Apply the checkbox change immediately, reverting if rejected."""
        desired = self.limit_checkbox.GetValue()
        applied = self.on_limit_to_curated_changed(desired)
        if applied is False:
            # Rejected (would leave no provider): show the persisted value.
            self.limit_checkbox.SetValue(
                self.app_settings.limit_providers_and_models_to_curated)

    def get_title(self) -> str:
        return "App Settings"
