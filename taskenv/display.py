"""Picking and opening the screen the camera is mounted under (the 4K TV, before that the 24" monitor),
plus the two drawing helpers every screen shares. It is not the laptop panel, and SDL's display order need
not match xrandr's, so `--display auto` picks it by resolution (`--screen-px`, SCREEN_PX in _env.sh), falling
back to the largest display after the panel."""
from __future__ import annotations

import os


def pick_display(want_index, sizes, want=(3840, 2160)) -> int:
    """`want_index`: "auto" or an explicit index. auto = the first display whose size is `want`; without one,
    the largest display that is not index 0 (SDL lists the laptop panel first), or 0 if it is the only one."""
    if isinstance(want_index, str) and want_index.lower() == "auto":
        for i, size in enumerate(sizes):
            if tuple(size) == tuple(want):
                return i
        others = [(size[0] * size[1], -i) for i, size in enumerate(sizes) if i > 0]
        return -max(others)[1] if others else 0
    i = int(want_index)
    if not 0 <= i < len(sizes):
        raise ValueError(f"display {i} out of range: {len(sizes)} display(s) {list(sizes)}")
    return i


def open_fullscreen(display="auto", want=(3840, 2160), caption="gaze", headless_size=None):
    """Fullscreen pygame surface on the chosen display. Returns (screen, display_index).
    headless_size=(w, h): no window at all (SDL dummy driver), for snapshot runs and tests."""
    if headless_size is not None:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    # Software window surface: with SDL's default GL-accelerated framebuffer, every window on this hybrid
    # amdgpu + nvidia laptop dies in X_GLXCreateContext (BadValue). Must be set before pygame initialises.
    os.environ.setdefault("SDL_FRAMEBUFFER_ACCELERATION", "0")
    import pygame

    pygame.init()
    if headless_size is not None:
        return pygame.display.set_mode(tuple(headless_size)), -1
    sizes = pygame.display.get_desktop_sizes()
    i = pick_display(display, sizes, want)
    if tuple(sizes[i]) != tuple(want):
        print(f"[display] WARNING: no {want[0]}x{want[1]} display (is the TV connected? xrandr); going fullscreen on "
              f"display {i} {sizes[i][0]}x{sizes[i][1]} instead", flush=True)
    screen = pygame.display.set_mode(sizes[i], pygame.FULLSCREEN, display=i)
    pygame.display.set_caption(caption)
    pygame.mouse.set_visible(False)
    keep_composited(pygame)
    print(f"[display] {i}: {sizes[i][0]}x{sizes[i][1]} (displays: {list(sizes)})", flush=True)
    return screen, i


def keep_composited(pygame):
    """Keep the fullscreen window inside the compositor so screen recorders see it. On X11 + GNOME (Mutter), an
    opaque window that redraws its whole area for 100 frames is "unredirected" (handed straight to the X server),
    and from then on GNOME's screencast and OBS record it as black; the task flips every frame, so that happens
    about 3 s in. _NET_WM_BYPASS_COMPOSITOR = 2 means "never bypass", which Mutter honours before that rule (SDL
    itself sets 1 = "please bypass"). No-op without an X11 window id or without xprop (Wayland, headless)."""
    import subprocess
    try:
        wid = pygame.display.get_wm_info().get("window")
    except Exception:
        wid = None
    if not wid:
        return False
    try:
        r = subprocess.run(["xprop", "-id", str(wid), "-f", "_NET_WM_BYPASS_COMPOSITOR", "32c",
                            "-set", "_NET_WM_BYPASS_COMPOSITOR", "2"], check=False, capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def text_box(screen, font, lines, x, y, center=False, color=(240, 240, 240), alpha=130, pad=12):
    """Lines of text on a translucent dark box, readable on any tile. (x, y) is the top-left corner, or
    the centre when `center`. Returns the box rect."""
    import pygame
    surfs = [font.render(t, True, color) for t in lines]
    lh = font.get_height() + 4
    bw, bh = max(s.get_width() for s in surfs) + 2 * pad, lh * len(surfs) + 2 * pad
    if center:
        x, y = x - bw / 2, y - bh / 2
    x, y = int(x), int(y)
    box = pygame.Surface((bw, bh), pygame.SRCALPHA)
    box.fill((0, 0, 0, alpha))
    screen.blit(box, (x, y))
    for i, s in enumerate(surfs):
        sx = x + pad + ((bw - 2 * pad - s.get_width()) // 2 if center else 0)
        screen.blit(s, (sx, y + pad + i * lh))
    return pygame.Rect(x, y, bw, bh)


def soft_dot(screen, x, y, color, radius=28):
    import pygame
    surf = pygame.Surface((radius * 2 + 2, radius * 2 + 2), pygame.SRCALPHA)
    for r, a in ((radius, 50), (int(radius * 0.66), 110), (int(radius * 0.33), 220)):
        pygame.draw.circle(surf, (*color, a), (radius + 1, radius + 1), r)
    screen.blit(surf, (int(x) - radius - 1, int(y) - radius - 1))


def parse_size(text) -> tuple:
    """'1920x1080' -> (1920, 1080)."""
    try:
        w, h = (int(v) for v in str(text).lower().split("x"))
    except ValueError:
        raise ValueError(f"size must look like 1920x1080, got {text!r}")
    return w, h


class ScreenRecorder:
    """The task window to a video file (cv2.VideoWriter; .avi = MJPG, anything else = mp4v) at most `fps` frames per
    second of wall-clock time, plus a sidecar <path>.times.csv (frame, t on the time.time() clock) so the video lines
    up with the camera recording (--record) and the per-sample CSV. write(surface, t) returns True when a frame
    was written."""

    def __init__(self, path, size, fps=30.0):
        import cv2
        fourcc = "MJPG" if str(path).lower().endswith(".avi") else "mp4v"
        self.path, self.size, self.fps = str(path), (int(size[0]), int(size[1])), float(fps)
        self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*fourcc), self.fps, self.size)
        if not self.writer.isOpened():
            raise SystemExit(f"cannot open {self.path} for writing (codec {fourcc}); try an .avi name")
        self.period, self.frames, self._last = 1.0 / self.fps, 0, None
        self.times_path = self.path + ".times.csv"
        self._times = open(self.times_path, "w")
        self._times.write("frame,t\n")

    def write(self, surface, t) -> bool:
        if self._last is not None and t - self._last < self.period:
            return False
        import numpy as np
        import pygame
        arr = pygame.surfarray.array3d(surface)                    # (W, H, RGB)
        self.write_array(np.ascontiguousarray(arr.transpose(1, 0, 2)[:, :, ::-1]), t)
        return True

    def write_array(self, bgr, t):
        """(H, W, 3) uint8 BGR of the recorder's size."""
        self.writer.write(bgr)
        self.frames += 1
        self._last = t
        self._times.write(f"{self.frames},{t:.3f}\n")
        self._times.flush()                                     # survives a killed task

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self._times.close()
