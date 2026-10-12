"""Main application entry point for Fluximp GUI."""

import wx
import wx.adv
import sys
import os
import argparse
import logging
import traceback
import time
from datetime import datetime, timezone
from pathlib import Path

import keyring
from keyring.errors import KeyringError

from fluximp_agent.model_config import get_provider_names
from fluximp_agent.curated_models import filter_provider_names
from fluximp_wxwidgets.main_frame import MainFrame
from fluximp_wxwidgets.app_settings import AppSettings
from fluximp_wxwidgets.app_settings_tab import NoCuratedProvidersDialog


# Keyring identifier for a provider's API key. Not a real username: every
# provider stores its key under this fixed username with the provider's name
# (e.g. "deepinfra.com") as the keyring service name.
API_KEY_USERNAME = "api_key"

NO_API_KEY_URL = "https://github.com/fluxmoot/fluximp#quick-start"


def scan_providers_with_api_keys(provider_names: list) -> list:
    """Return the subset of provider_names that have an API key set.

    keyring has no existence check, so each key is fetched and immediately
    discarded; only None means "not set" (an empty string still counts as
    set). A broken/unavailable backend must not crash startup, so keyring
    errors are treated as "no key" for that provider.
    """
    providers_with_keys = []
    for name in provider_names:
        try:
            api_key = keyring.get_password(name, API_KEY_USERNAME)
        except KeyringError:
            api_key = None
        if api_key is not None:
            providers_with_keys.append(name)
    return providers_with_keys


class NoApiKeyDialog(wx.Dialog):
    """Modal shown when no LLM inference provider has an API key set."""

    def __init__(self, parent):
        super().__init__(parent, title="No API Key Set")

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(
            wx.StaticText(
                self,
                label="No API key is set for any LLM inference provider.  See",
            ),
            0, wx.ALL, 12,
        )
        # The default EVT_HYPERLINK handling opens the URL in the browser.
        link = wx.adv.HyperlinkCtrl(self, label=NO_API_KEY_URL, url=NO_API_KEY_URL)
        sizer.Add(link, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

        btns = wx.StdDialogButtonSizer()
        ok_btn = wx.Button(self, wx.ID_OK, "Okay")
        ok_btn.SetDefault()
        btns.AddButton(ok_btn)
        btns.Realize()
        sizer.Add(btns, 0, wx.ALIGN_RIGHT | wx.ALL, 12)

        self.SetSizerAndFit(sizer)
        self.CentreOnScreen()


class FluximpGuiApp(wx.App):
    """Main wxPython application."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        # Setup application-wide logging
        self._setup_logging()
        # Install global exception hook to catch crashes
        self._original_excepthook = sys.excepthook
        sys.excepthook = self._global_excepthook
        super().__init__()
    
    def _setup_logging(self):
        """Setup application-wide logging to a uniquely named per-process file."""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        log_file = log_dir / f"app_{timestamp}_{pid}.log"
        
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        formatter.converter = time.gmtime
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    def _global_excepthook(self, exc_type, exc_value, exc_traceback):
        """Global exception hook to log crashes instead of silently dying."""
        tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logging.error(f"Uncaught exception: {tb_str}")
        # Also call the original hook
        self._original_excepthook(exc_type, exc_value, exc_traceback)
    
    def OnInit(self) -> bool:
        """Initialize the application."""
        # Enable high DPI support on Windows
        if sys.platform == 'win32':
            try:
                from ctypes import OleDLL
                OleDLL('shcore').SetProcessDpiAwareness(1)
            except (ImportError, OSError):
                pass

        # Gate startup on at least one LLM inference provider having an API
        # key set in keyring. Done before the frame exists so a missing key
        # aborts startup without ever flashing the main window.
        providers_with_keys = scan_providers_with_api_keys(get_provider_names())
        if not providers_with_keys:
            dlg = NoApiKeyDialog(None)
            dlg.ShowModal()
            dlg.Destroy()
            # OnInit returning False exits the app without entering the main loop.
            return False

        # Application-level settings (base_dir/settings.json). Created here so
        # the curated-provider gate below can read and persist them.
        app_settings = AppSettings(self.base_dir)

        # Gate startup on the curated filter leaving at least one provider with
        # an API key set. Without a provider the GUI can do nothing useful, so
        # offer to turn the filter off or exit - mirroring the no-API-key gate
        # above (also runs before the frame exists).
        if (app_settings.limit_providers_and_models_to_curated
                and not filter_provider_names(providers_with_keys, True)):
            dlg = NoCuratedProvidersDialog(None)
            result = dlg.ShowModal()
            dlg.Destroy()
            if result != wx.ID_YES:
                # Exit chosen (or the dialog was dismissed).
                return False
            app_settings.limit_providers_and_models_to_curated = False

        logging.info(f"Creating MainFrame")

        # Create main frame with base directory
        frame = MainFrame(title="Fluximp", base_dir=self.base_dir,
                          app_settings=app_settings)
        # Restrict the Provider combo to providers that have an API key set
        # (and, when enabled, the curated set).
        frame.right_panel.set_providers_with_keys(providers_with_keys)
        frame.Show()
        self.SetTopWindow(frame)

        return True


def main():
    """Entry point for the GUI application."""
    default_base_dir = Path.home() / ".fluximp"

    # Platform-appropriate display for help text
    if sys.platform == "win32":
        default_display = r"%%USERPROFILE%%\.fluximp"
    else:
        default_display = "~/.fluximp"

    parser = argparse.ArgumentParser(description="Fluximp GUI")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=default_base_dir,
        help=f"Base directory for workspaces (default: {default_display})"
    )
    args = parser.parse_args()

    # Single-instance guard (process_lock.py): exactly one GUI process per
    # --base-dir. The lock — a flock'd .fluximp-gui.lock file under the
    # base dir on POSIX, a named kernel mutex keyed by the base-dir hash
    # on Windows — is created once and never deleted; held-ness is the
    # exclusivity signal. A previous GUI's crash frees it instantly
    # (kernel-held state dies with the process) and a stale lock file can
    # never block a restart, so there is no cleanup path to get wrong.
    from fluximp_agent.process_lock import DirLock, probe
    base_dir = args.base_dir.resolve()
    message = (
        f"A Fluximp GUI is already running for base directory "
        f"'{base_dir}'. Exiting."
    )
    if probe(base_dir, 'gui') == 'held':
        print(message)
        sys.exit(1)
    gui_lock = DirLock.acquire(base_dir, 'gui', wait_timeout=2.0)
    if gui_lock is None:
        # The probe said free but ownership was taken before the acquire —
        # same verdict, same exit; the bounded wait covered the race.
        print(message)
        sys.exit(1)

    app = FluximpGuiApp(base_dir=base_dir)
    # Keep the single-instance lock referenced for the process lifetime
    # (the OS releases it at process exit, crash included).
    app._single_instance_lock = gui_lock
    app.MainLoop()


if __name__ == "__main__":
    main()
