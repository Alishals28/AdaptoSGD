import numpy as np
from typing import List, Tuple

class RingAllReduceBackend:
    """Synchronous Ring AllReduce communication backend."""

    def __init__(self, num_workers: int):
        self.num_workers = num_workers

    def aggregate(
        self, gradients: List[np.ndarray], worker_times: List[float]
    ) -> Tuple[np.ndarray, float]:
        """
        Aggregates gradients using a simulated Ring AllReduce.
        The key simulation aspect is the synchronization barrier.
        """
        # In Ring AllReduce, all workers must complete before aggregation.
        # The total time for the step is determined by the slowest worker.
        comm_time = max(worker_times) if worker_times else 0
        
        if len(gradients) < self.num_workers:
            # This simulates the blocking nature. If not all workers reported,
            # no gradient is returned, and the model is not updated.
            return np.array([]), comm_time

        # All workers are present, so we can aggregate.
        aggregated_gradient = np.sum(gradients, axis=0) / self.num_workers
        return aggregated_gradient, comm_time


class ParameterServerBackend:
    """Asynchronous Parameter Server communication backend."""

    def __init__(self, num_workers: int):
        self.num_workers = num_workers

    def aggregate(
        self, gradients: List[np.ndarray], worker_times: List[float]
    ) -> Tuple[np.ndarray, float]:
        """
        Aggregates gradients using a simulated Parameter Server.
        This is asynchronous, so we process whatever gradients are available.
        """
        # In PS, communication time is the sum of individual transfers,
        # but since it's async, we model it as the average time.
        comm_time = np.mean(worker_times) if worker_times else 0

        if not gradients:
            return np.array([]), comm_time

        # Asynchronous update: aggregate whatever gradients have arrived.
        aggregated_gradient = np.sum(gradients, axis=0) / self.num_workers
        return aggregated_gradient, comm_time
