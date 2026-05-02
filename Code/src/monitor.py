import numpy as np
from collections import deque
from typing import List, Tuple

class StragglerMonitor:
    """Online straggler detector based on median-smoothed per-worker timings."""

    def __init__(self, threshold: float = 1.5, window_size: int = 5):
        self.threshold = threshold
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        self.last_straggler_worker_id = -1
        self.last_score = 0.0

    def update(
        self, worker_times: List[float], iteration: int
    ) -> Tuple[bool, int, float]:
        """Updates the monitor with new worker timings and returns straggler status."""
        if not worker_times:
            return False, -1, 0.0

        self.history.append(worker_times)

        # Use the most recent set of times for detection
        latest_times = np.array(self.history[-1])

        t_max = np.max(latest_times)
        t_median = np.median(latest_times)

        if t_median == 0:
            score = 0.0
        else:
            score = t_max / t_median

        self.last_score = score
        straggler_detected = score > self.threshold

        if straggler_detected:
            self.last_straggler_worker_id = int(np.argmax(latest_times))
        else:
            # We don't reset the worker ID, so we can report who the *last* straggler was
            pass

        return straggler_detected, self.last_straggler_worker_id, score
