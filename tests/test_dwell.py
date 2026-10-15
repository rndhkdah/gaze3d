from taskenv.dwell import DwellTimer, cell_at, clamp_to_screen


def test_cell_at_maps_pixels_to_grid_cells():
    assert cell_at(0, 0, 1920, 1080, 6, 4) == (0, 0)
    assert cell_at(1919, 1079, 1920, 1080, 6, 4) == (5, 3)
    assert cell_at(960, 540, 1920, 1080, 6, 4) == (3, 2)


def test_cell_at_outside_the_screen_is_none():
    assert cell_at(-1, 10, 1920, 1080, 6, 4) is None
    assert cell_at(10, 1080, 1920, 1080, 6, 4) is None


def test_dwell_fills_while_the_cell_is_held_and_fires_once():
    d = DwellTimer(dwell_s=1.0)
    assert d.update((1, 1), 0.0) == (0.0, False)
    assert d.update((1, 1), 0.5) == (0.5, False)
    progress, fired = d.update((1, 1), 1.0)
    assert progress == 1.0 and fired is True
    assert d.update((1, 1), 1.5) == (1.0, False)        # stays full, fires only once


def test_dwell_resets_when_the_cell_changes_or_gaze_is_lost():
    d = DwellTimer(dwell_s=1.0)
    d.update((1, 1), 0.0)
    d.update((1, 1), 0.6)
    assert d.update((2, 1), 0.7) == (0.0, False)
    d.update((2, 1), 1.2)
    assert d.update(None, 1.3) == (0.0, False)
    assert d.update((2, 1), 1.4) == (0.0, False)         # restarts from zero


def test_clamp_to_screen_pulls_near_misses_onto_the_edge_only():
    assert clamp_to_screen(100.0, 200.0, 1920, 1080) == (100.0, 200.0)             # inside: untouched
    assert clamp_to_screen(100.0, 1150.0, 1920, 1080) == (100.0, 1079.0)           # just below: bottom edge
    assert clamp_to_screen(-50.0, -20.0, 1920, 1080) == (0.0, 0.0)                 # top-left corner
    assert clamp_to_screen(2000.0, 500.0, 1920, 1080, margin=0.25) == (1919.0, 500.0)
    assert clamp_to_screen(100.0, 1500.0, 1920, 1080, margin=0.25) == (100.0, 1500.0)   # beyond the margin: off screen
    assert clamp_to_screen(100.0, 1150.0, 1920, 1080, margin=0.0) == (100.0, 1150.0)    # disabled
    x, y = clamp_to_screen(100.0, 1150.0, 1920, 1080)
    assert cell_at(x, y, 1920, 1080, 3, 2) == (0, 1) and cell_at(100.0, 1150.0, 1920, 1080, 3, 2) is None
