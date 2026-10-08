"""GUI entrypoint — invoke directly by path; packages need not be installed."""
import sys
from pathlib import Path

for d in ("agent", "protocol", "sdk", "wx"):
    sys.path.insert(0, str(Path(__file__).resolve().parent / d))

from fluximp_wxwidgets.main import main

if __name__ == "__main__":
    raise SystemExit(main())
