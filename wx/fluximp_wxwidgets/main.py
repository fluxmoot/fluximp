"""Main application entry point for Fluximp GUI."""

import wx
import sys
import os
import argparse
import logging
import traceback
import time
from datetime import datetime, timezone
from pathlib import Path

from fluximp_wxwidgets.main_frame import MainFrame


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
        
        logging.info(f"Creating MainFrame")

        # Create main frame with base directory
        frame = MainFrame(title="Fluximp", base_dir=self.base_dir)
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
    
    app = FluximpGuiApp(base_dir=args.base_dir)
    app.MainLoop()


if __name__ == "__main__":
    main()
