import pytest

from taskenv.display import pick_display

SIZES = [(2560, 1600), (3840, 2160)]      # SDL order may differ from xrandr order


def test_auto_picks_the_display_with_the_wanted_resolution():
    assert pick_display("auto", SIZES, want=(3840, 2160)) == 1
    assert pick_display("auto", [(2560, 1600), (1920, 1080)], want=(1920, 1080)) == 1


def test_explicit_index_is_used_as_is():
    assert pick_display("0", SIZES, want=(3840, 2160)) == 0
    assert pick_display(1, SIZES, want=(3840, 2160)) == 1


def test_auto_without_a_match_falls_back_to_the_largest_display_after_the_panel():
    assert pick_display("auto", [(2560, 1600)], want=(3840, 2160)) == 0
    assert pick_display("auto", [(2560, 1600), (1920, 1080), (2560, 1440)], want=(3840, 2160)) == 2


def test_explicit_index_out_of_range_raises():
    with pytest.raises(ValueError):
        pick_display("5", SIZES, want=(3840, 2160))


def test_open_fullscreen_disables_sdl_gl_framebuffer_acceleration(monkeypatch):
    """On this hybrid-GPU laptop every pygame window dies with an X GLX BadValue error unless SDL uses a
    software window surface; open_fullscreen must set the hint before pygame initialises."""
    import os
    import pygame
    from taskenv.display import open_fullscreen
    monkeypatch.delenv("SDL_FRAMEBUFFER_ACCELERATION", raising=False)
    open_fullscreen(headless_size=(64, 48))
    try:
        assert os.environ.get("SDL_FRAMEBUFFER_ACCELERATION") == "0"
    finally:
        pygame.quit()


def test_parse_size():
    from taskenv.display import parse_size
    assert parse_size("1920x1080") == (1920, 1080) and parse_size("3840X2160") == (3840, 2160)
    with pytest.raises(ValueError):
        parse_size("wide")


def test_screen_recorder_writes_frames_at_most_at_fps(tmp_path):
    cv2 = pytest.importorskip("cv2")
    pygame = pytest.importorskip("pygame")
    import numpy as np
    from taskenv.display import ScreenRecorder
    path = str(tmp_path / "task.avi")
    rec = ScreenRecorder(path, (64, 48), fps=10.0)
    surf = pygame.Surface((64, 48))
    surf.fill((255, 0, 0))                                        # red in RGB
    assert rec.write(surf, 100.0) is True
    assert rec.write(surf, 100.05) is False                        # inside the 0.1 s period
    assert rec.write(surf, 100.2) is True
    rec.write_array(np.zeros((48, 64, 3), np.uint8), 100.4)
    rec.close()
    assert rec.frames == 3
    cap = cv2.VideoCapture(path)
    n, first = 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        first = frame if first is None else first
        n += 1
    cap.release()
    assert n == 3 and first.shape == (48, 64, 3) and first[10, 10, 2] > 200 and first[10, 10, 0] < 60   # BGR: red last
    lines = open(path + ".times.csv").read().splitlines()
    assert lines == ["frame,t", "1,100.000", "2,100.200", "3,100.400"]
