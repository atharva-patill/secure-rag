import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text


@dataclass
class _HeaderView:
    header: "SecureRagHeader"
    status: str
    active: bool = False
    title_length: Optional[int] = None
    start_time: float = 0.0

    def __rich__(self):
        elapsed = time.monotonic() - self.start_time
        frame_index = int(elapsed * self.header.frames_per_second)
        spinner = self.header.spinner_frames[frame_index % len(self.header.spinner_frames)]
        divider_phase = frame_index // 2
        return self.header.render(
            status=self.status,
            active=self.active,
            spinner=spinner,
            divider_phase=divider_phase,
            title_length=self.title_length,
        )


class SecureRagHeader:
    spinner_frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    frames_per_second = 10

    def __init__(self, console: Console, title: str = "Secure RAG"):
        self.console = console
        self.title = title
        self.status = "Initializing"

    def render(
        self,
        status: Optional[str] = None,
        *,
        active: bool = False,
        spinner: Optional[str] = None,
        divider_phase: int = 0,
        title_length: Optional[int] = None,
    ) -> Group:
        status = status or self.status
        title = self.title if title_length is None else self.title[:title_length]
        prefix = f"{spinner} " if active and spinner else ""
        left = f"{prefix}{title}".rstrip()
        width = max(20, self.console.width)
        line = self._header_line(left, status, width)
        divider = self._divider(width, divider_phase if active else None)
        return Group(Text(line), Text(divider))

    def startup(self, duration: float = 0.4) -> None:
        steps = max(1, int(duration * self.frames_per_second))
        view = _HeaderView(self, self.status, active=True, title_length=0, start_time=time.monotonic())

        with Live(
            view,
            console=self.console,
            refresh_per_second=self.frames_per_second,
            transient=True,
        ) as live:
            for step in range(steps + 1):
                visible = round(len(self.title) * step / steps)
                view.title_length = visible
                live.update(view, refresh=True)
                time.sleep(duration / steps)

            self.status = "Ready"
            view.status = self.status
            view.active = False
            view.title_length = None
            live.update(view, refresh=True)

    @contextmanager
    def working(self, status: str):
        self.status = status
        view = _HeaderView(self, status, active=True, start_time=time.monotonic())

        with Live(view, console=self.console, refresh_per_second=self.frames_per_second) as live:
            try:
                yield
            finally:
                self.status = "Ready"
                view.status = self.status
                view.active = False
                live.update(view, refresh=True)

    def set_status(self, status: str) -> None:
        self.status = status
        self.console.print(self.render(status=status, active=False))

    def _header_line(self, left: str, status: str, width: int) -> str:
        if not status:
            return left[:width]

        min_gap = 2
        available_left = max(0, width - len(status) - min_gap)
        if len(left) > available_left:
            left = left[:available_left].rstrip()

        gap = max(min_gap, width - len(left) - len(status))
        return f"{left}{' ' * gap}{status}"[:width]

    def _divider(self, width: int, phase: Optional[int]) -> str:
        if phase is None or width < 12:
            return "─" * width

        sweep_width = 3
        span = width + sweep_width
        start = phase % span - sweep_width
        chars = ["─"] * width
        for offset in range(sweep_width):
            index = start + offset
            if 0 <= index < width:
                chars[index] = "━"
        return "".join(chars)
