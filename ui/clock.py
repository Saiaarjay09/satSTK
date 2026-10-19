"""One shared simulation clock: every view listens to it, so globe, map, timeline and telemetry always agree."""
from __future__ import annotations

import time

from PySide6.QtCore import QObject, QTimer, Signal

SPEEDS = [("1x", 1.0), ("10x", 10.0), ("60x (1 min/s)", 60.0), ("300x", 300.0), ("600x (10 min/s)", 600.0),
          ("1800x", 1800.0), ("3600x (1 h/s)", 3600.0)]


class PlaybackClock(QObject):
    """Holds the current simulation time (seconds since scenario start) and advances it while playing."""
    time_changed = Signal(float)
    state_changed = Signal(bool)       # True = playing

    def __init__(self, parent=None):
        super().__init__(parent)
        self.t = 0.0
        self.t_end = 0.0
        self.step_s = 10.0
        self.speed = 60.0
        self.loop = True
        self.playing = False
        self._last = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(33)       # about 30 frames per second
        self._timer.timeout.connect(self._tick)

    def configure(self, t_end: float, step_s: float) -> None:
        self.t_end, self.step_s = float(t_end), float(step_s)
        self.pause()
        self.seek(0.0)

    def seek(self, t: float) -> None:
        self.t = min(max(float(t), 0.0), self.t_end)
        self.time_changed.emit(self.t)

    def play(self) -> None:
        if self.t_end <= 0:
            return
        if self.t >= self.t_end:
            self.seek(0.0)
        self._last = time.monotonic()
        self.playing = True
        self._timer.start()
        self.state_changed.emit(True)

    def pause(self) -> None:
        self.playing = False
        self._timer.stop()
        self.state_changed.emit(False)

    def toggle(self) -> None:
        self.pause() if self.playing else self.play()

    def step(self, direction: int = 1) -> None:
        """Move one scenario time step forward (+1) or back (-1). Pauses playback."""
        self.pause()
        self.seek(self.t + direction * self.step_s)

    def set_speed(self, speed: float) -> None:
        self.speed = float(speed)

    def _tick(self) -> None:
        now = time.monotonic()
        dt = min(now - self._last, 0.25) * self.speed
        self._last = now
        nt = self.t + dt
        if nt >= self.t_end:
            if self.loop:
                nt -= self.t_end
            else:
                self.seek(self.t_end)
                self.pause()
                return
        self.seek(nt)
