"""Thread-safe token bucket; rate units are requests per second."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable


class TokenBucket:
    """Limit request starts; capacity=1 prevents initial SEC request bursts."""

    def __init__(
        self,
        rate: float = 8,
        capacity: float = 1,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not math.isfinite(rate) or not math.isfinite(capacity) or rate <= 0 or capacity < 1:
            raise ValueError("rate must be positive and capacity at least one")
        self.rate, self.capacity = rate, capacity
        self.clock, self.sleep = clock, sleep
        self.tokens, self.updated = capacity, clock()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        """Block until one token is available; monotonic elapsed time is in seconds."""
        while True:
            with self.lock:
                now = self.clock()
                self.tokens = min(
                    self.capacity, self.tokens + max(0, now - self.updated) * self.rate
                )
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                delay = (1 - self.tokens) / self.rate
            self.sleep(delay)
