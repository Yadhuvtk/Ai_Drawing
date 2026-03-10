from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Stopwatch:
    start: float = 0.0

    def __post_init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start


@contextmanager
def timer():
    start = time.perf_counter()
    try:
        yield
    finally:
        _ = time.perf_counter() - start
