"""Tab-header status icons (shape + colour).

Each workflow tab's header shows a small icon whose *shape* encodes the
conversation's status, with colour as a redundant secondary cue, so the
status stays legible for colour-blind users and in greyscale (the shapes
are mutually distinguishable by silhouette alone).

Consumed by ``MainFrame._apply_tab_status`` through the notebook's
per-page bitmap (``AuiNotebook.AddPage`` / ``SetPageBitmap``), driven by
``ConversationTab.get_status``.
"""

import wx
from typing import Dict, Tuple


# Status vocabulary (single source of truth, shared with ConversationTab).
STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_WAITING = "waiting"
STATUS_APPROVAL = "approval"
STATUS_PAUSED = "paused"
STATUS_STOPPED = "stopped"
STATUS_ERROR = "error"
STATUS_COMPLETED = "completed"

_ICON_SIZE = 16

# status -> (shape, colour).  Shapes never collapse into each other when
# colour is removed (hollow square, play triangle, diamond, octagon-with-
# exclamation, twin pause bars, crossed square, cross, check), so colour is
# never the sole cue.
_STATUS_SPECS: Dict[str, Tuple[str, wx.Colour]] = {
    STATUS_IDLE:      ("square",       wx.Colour(130, 130, 130)),
    STATUS_RUNNING:   ("triangle",     wx.Colour(40, 160, 60)),
    STATUS_WAITING:   ("diamond",      wx.Colour(230, 150, 0)),
    STATUS_APPROVAL:  ("octagon_bang", wx.Colour(230, 150, 0)),
    STATUS_PAUSED:    ("bars",         wx.Colour(60, 110, 190)),
    STATUS_STOPPED:   ("square_slash", wx.Colour(110, 110, 120)),
    STATUS_ERROR:     ("cross",        wx.Colour(200, 45, 45)),
    STATUS_COMPLETED: ("check",        wx.Colour(0, 140, 140)),
}

_icon_cache: Dict[str, wx.Bitmap] = {}


def _draw_shape(gc, shape: str, colour: wx.Colour) -> None:
    """Draw *shape* in *colour* into a 16x16 graphics context."""
    pen = wx.Pen(colour, 2)
    pen.SetCap(wx.CAP_ROUND)
    pen.SetJoin(wx.JOIN_ROUND)
    gc.SetPen(pen)
    gc.SetBrush(wx.Brush(colour))

    if shape == "triangle":
        path = gc.CreatePath()
        path.MoveToPoint(4.5, 3.0)
        path.AddLineToPoint(4.5, 13.0)
        path.AddLineToPoint(13.0, 8.0)
        path.CloseSubpath()
        gc.FillPath(path)
    elif shape == "diamond":
        path = gc.CreatePath()
        path.MoveToPoint(8.0, 2.0)
        path.AddLineToPoint(13.5, 8.0)
        path.AddLineToPoint(8.0, 14.0)
        path.AddLineToPoint(2.5, 8.0)
        path.CloseSubpath()
        gc.FillPath(path)
    elif shape == "octagon_bang":
        # Octagon outline (stop-sign silhouette) with a solid exclamation
        # inside: "the agent is blocked waiting on your approval".
        #
        # The outline is thinner than the other shapes' 2px pen.  Its path
        # is expanded outward by the difference in half-widths (0.5) so the
        # *outer* perimeter lands where it did at 2px while the *inner*
        # perimeter moves outward — that un-rounds the edge and gives the
        # exclamation room so the dot no longer crowds it.
        outline_pen = wx.Pen(colour, 1)
        outline_pen.SetJoin(wx.JOIN_MITER)
        gc.SetPen(outline_pen)
        path = gc.CreatePath()
        path.MoveToPoint(4.5, 2.0)
        path.AddLineToPoint(11.5, 2.0)
        path.AddLineToPoint(14.0, 4.5)
        path.AddLineToPoint(14.0, 11.5)
        path.AddLineToPoint(11.5, 14.0)
        path.AddLineToPoint(4.5, 14.0)
        path.AddLineToPoint(2.0, 11.5)
        path.AddLineToPoint(2.0, 4.5)
        path.CloseSubpath()
        gc.SetBrush(wx.TRANSPARENT_BRUSH)
        gc.StrokePath(path)
        # Exclamation (bar + dot) as a filled path so it antialiases on the
        # same float grid as the octagon (DrawRectangle/DrawEllipse snapped
        # to integer pixels on MSW).  The mark is nudged right of the
        # geometric centre (7.5, i.e. centre x=8.4, not 8.0): the octagon's
        # stroked outline and the filled mark rasterise on slightly
        # different subpixel grids, so at 8.0 the mark read a touch left of
        # the octagon's visual centre.  This optical correction re-centres it.
        bang = gc.CreatePath()
        bang.AddRectangle(7.5, 5.4, 1.8, 4.2)
        bang.AddEllipse(7.5, 10.4, 1.8, 1.8)
        gc.SetPen(wx.TRANSPARENT_PEN)
        gc.SetBrush(wx.Brush(colour))
        gc.FillPath(bang)
    elif shape == "bars":
        gc.DrawRectangle(4.0, 3.0, 3.0, 10.0)
        gc.DrawRectangle(9.0, 3.0, 3.0, 10.0)
    elif shape == "cross":
        gc.StrokeLine(4.0, 4.0, 12.0, 12.0)
        gc.StrokeLine(12.0, 4.0, 4.0, 12.0)
    elif shape == "check":
        path = gc.CreatePath()
        path.MoveToPoint(3.5, 8.5)
        path.AddLineToPoint(6.5, 11.5)
        path.AddLineToPoint(12.5, 4.5)
        gc.StrokePath(path)
    elif shape == "square":
        gc.SetBrush(wx.TRANSPARENT_BRUSH)
        gc.DrawRectangle(3.5, 3.5, 9.0, 9.0)
    elif shape == "square_slash":
        gc.SetBrush(wx.TRANSPARENT_BRUSH)
        gc.DrawRectangle(3.5, 3.5, 9.0, 9.0)
        gc.StrokeLine(3.0, 13.0, 13.0, 3.0)


def _create_status_bitmap(shape: str, colour: wx.Colour) -> wx.Bitmap:
    """Render a transparent 16x16 icon for the given shape/colour."""
    bmp = wx.Bitmap.FromRGBA(_ICON_SIZE, _ICON_SIZE, 0, 0, 0, 0)
    dc = wx.MemoryDC(bmp)
    gc = wx.GraphicsContext.Create(dc)
    if gc is not None:
        gc.SetAntialiasMode(wx.ANTIALIAS_DEFAULT)
        _draw_shape(gc, shape, colour)
        gc.Flush()
        del gc
    dc.SelectObject(wx.NullBitmap)
    return bmp


def get_status_icon(status: str) -> wx.Bitmap:
    """Return the cached status icon bitmap for *status*.

    Unknown statuses fall back to the idle icon.  Bitmaps are built once
    (lazily, on the GUI thread -- requires a running ``wx.App``) and cached
    for the life of the process.
    """
    cached = _icon_cache.get(status)
    if cached is not None:
        return cached
    shape, colour = _STATUS_SPECS.get(status, _STATUS_SPECS[STATUS_IDLE])
    bmp = _create_status_bitmap(shape, colour)
    _icon_cache[status] = bmp
    return bmp
