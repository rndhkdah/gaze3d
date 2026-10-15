"""Grid lookup and the dwell timer (pure, no pygame)."""
from __future__ import annotations


def cell_at(x: float, y: float, w: int, h: int, cols: int, rows: int):
    """(col, row) of the grid cell under a screen point, or None when off screen."""
    if not (0 <= x < w and 0 <= y < h):
        return None
    return int(x * cols / w), int(y * rows / h)


def clamp_to_screen(x: float, y: float, w: int, h: int, margin: float = 0.25):
    """A point up to `margin` x the screen size past an edge is moved onto that edge (the gaze aimed at an edge
    tile but mapped just outside); farther away it is returned unchanged, so cell_at still says off screen.
    margin <= 0 disables."""
    if margin <= 0:
        return x, y
    if -margin * w <= x < w * (1 + margin) and -margin * h <= y < h * (1 + margin):
        return min(max(x, 0.0), w - 1.0), min(max(y, 0.0), h - 1.0)
    return x, y


class DwellTimer:
    """Fills from 0 to 1 while the same cell is held; fires once when full; resets on a change."""

    def __init__(self, dwell_s: float):
        self.dwell_s = float(dwell_s)
        self._cell, self._t0, self._fired = None, None, False

    def update(self, cell, t: float):
        if cell is None or cell != self._cell:
            self._cell, self._t0, self._fired = cell, (t if cell is not None else None), False
            return 0.0, False
        progress = min(1.0, (t - self._t0) / self.dwell_s)
        fired = progress >= 1.0 and not self._fired
        if fired:
            self._fired = True
        return progress, fired
