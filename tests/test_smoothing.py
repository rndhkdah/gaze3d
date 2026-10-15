import numpy as np

from taskenv.smoothing import GazeSmoother, OneEuro

DT = 1.0 / 30.0


def test_one_euro_constant_input_is_unchanged():
    f = OneEuro(min_cutoff=1.0, beta=0.02)
    out = [f(100.0, i * DT) for i in range(10)]
    assert out == [100.0] * 10


def test_one_euro_attenuates_jitter():
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 10.0, 300)
    f = OneEuro(min_cutoff=1.0, beta=0.0)
    out = np.array([f(500.0 + n, i * DT) for i, n in enumerate(noise)])
    assert out[30:].std() < 0.5 * noise[30:].std()


def test_one_euro_follows_a_fast_step():
    f = OneEuro(min_cutoff=1.0, beta=0.02)
    for i in range(10):
        f(0.0, i * DT)
    out = [f(500.0, (10 + i) * DT) for i in range(3)]
    assert out[-1] > 490.0


def test_smoother_returns_none_before_any_valid_point():
    sm = GazeSmoother()
    assert sm.update(None, 0.0) is None


def test_smoother_holds_last_point_while_invalid_within_window():
    sm = GazeSmoother(lost_after=0.3)
    sm.update((100.0, 200.0), 0.0)
    r = sm.update(None, 0.1)
    assert (r.x, r.y) == (100.0, 200.0)
    assert r.valid is False and r.lost is False
    assert abs(r.hold_age - 0.1) < 1e-9


def test_smoother_marks_lost_after_window():
    sm = GazeSmoother(lost_after=0.3)
    sm.update((100.0, 200.0), 0.0)
    r = sm.update(None, 0.4)
    assert r.lost is True and r.valid is False
    assert (r.x, r.y) == (100.0, 200.0)


def test_smoother_resets_filter_after_loss():
    sm = GazeSmoother(lost_after=0.3)
    sm.update((100.0, 200.0), 0.0)
    sm.update(None, 0.4)
    r = sm.update((900.0, 900.0), 0.5)
    assert (r.x, r.y) == (900.0, 900.0)
    assert r.valid is True and r.lost is False and r.hold_age == 0.0


def test_smoother_smooths_while_valid():
    sm = GazeSmoother(min_cutoff=1.0, beta=0.0)
    sm.update((100.0, 100.0), 0.0)
    r = sm.update((110.0, 100.0), DT)
    assert 100.0 < r.x < 110.0 and r.y == 100.0 and r.valid is True
