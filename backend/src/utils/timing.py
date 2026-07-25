"""Tiny wall-clock stopwatch for PHI-safe latency instrumentation.

Usage:
    with stopwatch() as ms:
        result = await do_work()
    log_request_event(..., latency_ms=ms())

`ms()` returns elapsed milliseconds (rounded) and can be called during or after the
`with` block. Only a duration number is ever produced — no request content.
"""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager


@contextmanager
def stopwatch() -> Iterator[Callable[[], float]]:
    """Yield a callable returning elapsed milliseconds since the block was entered."""
    start = time.perf_counter()
    yield lambda: round((time.perf_counter() - start) * 1000, 1)
