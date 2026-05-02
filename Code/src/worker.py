import time
import numpy as np
from typing import Tuple

def worker_compute_task(args: Tuple[int, int, float, bool, int, float, int, bool, float]) -> Tuple[np.ndarray, float, bool]:
    """Worker-side computation task executed in a separate process."""
    (
        worker_id,
        gradient_size,
        base_compute_time,
        straggler_active,
        straggler_worker,
        straggler_delay_factor,
        seed,
        failed,
        sleep_scale,
    ) = args

    if failed:
        # Simulate a worker that has crashed and is unresponsive
        time.sleep(3600) # Sleep for a long time
        return np.array([]), -1.0, True # Should not be reached

    rng = np.random.default_rng(seed)
    worker_time = base_compute_time * rng.uniform(0.8, 1.2)
    if straggler_active and worker_id == straggler_worker:
        worker_time *= straggler_delay_factor

    grad = rng.normal(0.0, 0.1, gradient_size)
    if sleep_scale > 0:
        time.sleep(worker_time * sleep_scale)

    return grad, worker_time, False
