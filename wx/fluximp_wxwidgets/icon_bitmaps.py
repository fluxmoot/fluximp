"""DPI-aware, transparent icon bitmaps for the tree controls.

The tree icons in ``left_panel.py`` and ``workflow_panel.py`` are built here
rather than inline because two things have to be right, and both are easy to
get subtly wrong:

Transparency
    A bitmap created with ``wx.Bitmap(w, h)`` has no alpha channel, so it
    cannot be transparent no matter what is drawn on it.  That is why the tree
    icons used to carry an opaque light square behind them while the tab
    status icons (``tab_status_icons.py``) were already transparent: the tab
    icons come from ``wx.Bitmap.FromRGBA(...)`` and the tree icons did not.
    Every bitmap here is created the same way, with alpha 0, and drawn with a
    ``wx.GraphicsContext`` so the shape is antialiased and blends with
    whatever is behind it.  No background is ever cleared.

DPI
    ``wxImageList`` is documented as unsuitable for new code because it
    cannot show DPI-dependent bitmaps -- "use ``wxWithImages::SetImages()``
    instead" -- so each tree control is handed a ``wx.BitmapBundle`` holding a
    1x/2x/3x rendering of the same shape and wx picks the one that matches the
    display.  Without that the icons are a fixed 16 physical pixels and simply
    look small on a HiDPI screen.

Callers pass a ``draw`` callback that paints one icon into a ``size`` x
``size`` logical box; the scale is applied here, so the drawing code is
identical at every resolution.
"""

import wx
from typing import Callable

# Logical size of every tree icon.  The ``draw`` callbacks work in this
# coordinate space and the bundle renders it at each scale below.
ICON_SIZE = 16

# Scales rendered into each bundle, smallest first: wxBitmapBundle infers the
# scale factor of each bitmap from its size relative to the first one.
_SCALES = (1.0, 2.0, 3.0)


def icon_bundle(draw: Callable[[wx.GraphicsContext], None],
                size: int = ICON_SIZE) -> wx.BitmapBundle:
    """Return a transparent, DPI-aware bundle for one icon.

    *draw* receives a ``wx.GraphicsContext`` scaled so the icon occupies a
    *size* x *size* logical box.  It should paint the shape and nothing else:
    there is no background to clear, and whatever it does not paint stays
    transparent.
    """
    bitmaps = []
    for scale in _SCALES:
        pixels = int(round(size * scale))
        # FromRGBA (not wx.Bitmap) is what makes the alpha channel exist at
        # all; alpha 0 means "fully transparent" and the shape is drawn over
        # it, so the untouched area keeps picking up whatever is behind it.
        bmp = wx.Bitmap.FromRGBA(pixels, pixels, 0, 0, 0, 0)
        dc = wx.MemoryDC(bmp)
        gc = wx.GraphicsContext.Create(dc)
        if gc is not None:
            gc.SetAntialiasMode(wx.ANTIALIAS_DEFAULT)
            # Draw in logical coordinates and let the scale transform do the
            # rest, so a 1x and a 3x rendering are the same code and the
            # stroke widths scale with them.
            gc.Scale(scale, scale)
            draw(gc)
            gc.Flush()
            del gc
        dc.SelectObject(wx.NullBitmap)
        bitmaps.append(bmp)
    return wx.BitmapBundle.FromBitmaps(bitmaps)
