# fluximp-wxwidgets

The FluxImp wxWidgets package is a desktop GUI providing multiple AI agent workflows for coding and web research.

FluxImp has been heavily used on Windows 11 with this Python 3.10 interpreter:  https://github.com/conda-forge/miniforge/releases/tag/24.3.0-0

It has been lightly tested on Windows 11 with Python 3.11 and Python 3.12 versions of Miniforge and Python 3.12 on Windows 10 but not heavily used in those environments.  
Python 3.13+ is not supported by https://pypi.org/project/python-ripgrep/ which is required.

While wxWidgets is cross-platform FluxImp has not yet been tested on an OS other than Windows.

## Install, Configure, and Run

```console
python -m pip install fluximp-wxwidgets
python -m keyring set deepinfra.com api_key
python -m fluximp_wxwidgets --base-dir <e.g. %USERPROFILE%\fluximp or ~/fluximp>
```

## License

MPL-2.0 — see [LICENSE](http://mozilla.org/MPL/2.0/).
