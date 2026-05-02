import numpy as np

class SSPEnforcer:
    """Enforces the Stale Synchronous Parallel invariant during PS mode.

    The invariant: the fastest worker may not be more than `staleness_bound`
    iterations ahead of the slowest active worker.  When a worker would exceed
    this bound, _enforce() returns True and the caller skips applying that
    worker's gradient for this iteration (simulating a blocking wait).
    """

    def __init__(self, staleness_bound: int):
        self.staleness_bound = staleness_bound
        self.worker_versions: np.ndarray = np.array([])

    def reset(self, num_workers: int) -> None:
        """Initializes worker versions at the start of training."""
        self.worker_versions = np.zeros(num_workers, dtype=int)

    def advance(self, worker_id: int) -> None:
        """Increments the version for a worker that has completed an iteration."""
        self.worker_versions[worker_id] += 1

    def is_blocked(self, worker_id: int) -> bool:
        """
        Checks if a worker should be blocked to enforce the SSP bound.
        A worker is blocked if its version is too far ahead of the slowest worker.
        """
        if self.staleness_bound <= 0:
            return False
            
        min_version = np.min(self.worker_versions)
        my_version = self.worker_versions[worker_id]
        return my_version >= min_version + self.staleness_bound

    def get_staleness(self, global_version: int) -> int:
        """Calculates the current maximum staleness in the system."""
        if self.worker_versions.size == 0:
            return 0
        return global_version - int(np.min(self.worker_versions))
