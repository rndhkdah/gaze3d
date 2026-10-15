import time

import numpy as np

from gaze3d.capture import Frame
from gaze3d.tracker import GazeTracker, eye_box

EYE_POINTS = np.array([(400, 308), (415, 300), (445, 300), (460, 308), (540, 308), (555, 300), (585, 300), (600, 308)])


class FakeSample:
    def __init__(self, t, eye_points=None):
        self.valid, self.face_found, self.blink, self.timestamp, self.eye_points = True, True, False, t, eye_points

    def as_vector(self):
        return np.array([1.0, 2.0])


class FakeCapture:
    def __init__(self, drop_every=0, size=(2, 2), frames=0):
        self.seq, self.started, self.drop_every, self.size, self.frames = 0, False, drop_every, tuple(size), frames
        self.ended = False

    def start(self):
        self.started = True
        return self

    def stop(self):
        self.started = False

    def read(self, timeout_ms=1000):
        time.sleep(0.004)
        self.seq += 1
        if self.frames and self.seq > self.frames:
            self.ended = True
            return None
        if self.drop_every and self.seq % self.drop_every == 0:
            return None                                   # a timeout
        now = time.time()
        h, w = self.size
        return Frame(np.zeros((h, w, 3), np.uint8), now, self.seq, now)


class FakeExtractor:
    closed = 0

    def process(self, frame):
        return FakeSample(frame.timestamp)

    def close(self):
        self.closed += 1


class LandmarkExtractor(FakeExtractor):
    def process(self, frame):
        return FakeSample(frame.timestamp, EYE_POINTS)


class FakeMapper:
    def predict(self, vec):
        return (100.0, 200.0)


def _wait_for(tracker, n, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = tracker.latest()
        if s is not None and s.seq >= n:
            return s
        time.sleep(0.005)
    raise AssertionError("tracker produced no samples")


def test_eye_box_covers_the_points_and_clamps():
    x0, y0, x1, y1 = eye_box(EYE_POINTS, 640, 480)
    assert x0 < 400 and x1 >= 600 and y0 < 300 and y1 > 308                      # covers both eyes, clamped at 640
    x0, y0, x1, y1 = eye_box(EYE_POINTS, 2000, 2000)
    assert abs((x1 - x0) / (y1 - y0) - 8 / 3) < 0.05                               # the preview aspect when unclamped
    assert eye_box([(0, 0), (1, 1)], 10, 10)[:2] == (0, 0) and eye_box([(600, 400), (640, 480)], 640, 480)[2:] == (640, 480)


def test_tracker_delivers_samples_with_screen_points_and_stops():
    cap, ex = FakeCapture(), FakeExtractor()
    tr = GazeTracker(cap, ex, mapper=FakeMapper()).start()
    try:
        assert cap.started
        s = _wait_for(tr, 5)
        assert s.vec.shape == (2,) and s.screen == (100.0, 200.0) and s.t_done >= s.t and tr.fps > 0
    finally:
        tr.stop()
    assert not cap.started and ex.closed == 1
    seq = tr.latest().seq
    time.sleep(0.05)
    assert tr.latest().seq == seq            # the thread really stopped


def test_tracker_without_mapper_then_with_one():
    tr = GazeTracker(FakeCapture(), FakeExtractor()).start()
    try:
        s = _wait_for(tr, 3)
        assert s.screen is None and s.vec is not None
        tr.set_mapper(FakeMapper())
        assert _wait_for(tr, s.seq + 3).screen == (100.0, 200.0)
    finally:
        tr.stop()


def test_tracker_survives_timeouts_and_ends_with_the_capture():
    tr = GazeTracker(FakeCapture(drop_every=2, frames=12), FakeExtractor()).start()
    try:
        s = _wait_for(tr, 6)
        assert s.seq >= 6
        time.sleep(0.2)
        assert tr.latest().seq == 6 and tr.error is None     # 12 reads, every second one dropped, then ended
    finally:
        tr.stop()


def test_eye_crop_or_small_frame():
    tr = GazeTracker(FakeCapture(size=(480, 640)), LandmarkExtractor()).start()
    try:
        s = _wait_for(tr, 3)
        crop, x0, y0 = s.eye_crop
        assert crop.shape[2] == 3 and x0 < 400 and x0 + crop.shape[1] > 600 and s.frame_small is None
    finally:
        tr.stop()
    tr = GazeTracker(FakeCapture(size=(480, 640)), FakeExtractor()).start()
    try:
        s = _wait_for(tr, 3)
        assert s.eye_crop is None and s.frame_small.shape == (240, 320, 3)
    finally:
        tr.stop()


def test_tracker_keeps_the_error_that_stopped_it():
    class Broken(FakeExtractor):
        def process(self, frame):
            raise RuntimeError("boom")

    tr = GazeTracker(FakeCapture(), Broken()).start()
    try:
        deadline = time.time() + 2.0
        while tr.error is None and time.time() < deadline:
            time.sleep(0.01)
        assert isinstance(tr.error, RuntimeError)
    finally:
        tr.stop()
