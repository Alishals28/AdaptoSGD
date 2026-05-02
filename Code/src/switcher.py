from typing import List, Tuple

from .config import SystemConfig, CommunicationMode
from .monitor import StragglerMonitor

class AdaptiveSwitcher:
    """Adaptive strategy selector between Ring AllReduce and Parameter Server.

    Hysteresis (Section 2.4):
        When in PS mode, the switcher does NOT immediately revert to AllReduce
        the moment a straggler clears.  Instead it requires `hysteresis_window`
        consecutive "clean" (no straggler detected) iterations before committing
        to the switch-back.  This prevents rapid thrashing between modes.
    """

    def __init__(self, config: SystemConfig):
        self.monitor = StragglerMonitor(
            threshold=config.straggler_threshold, window_size=config.monitoring_window
        )
        self.current_mode = CommunicationMode.RING_ALLREDUCE
        self.hysteresis_window = config.hysteresis_window
        self.switch_history: List[Tuple[int, str, float]] = []
        self.clean_iterations_count = 0
        self.last_straggler_detected = False

    def evaluate_and_switch(
        self, worker_times: List[float], iteration: int
    ) -> Tuple[CommunicationMode, bool]:
        """Applies switching logic based on straggler score."""
        straggler_detected, _, _ = self.monitor.update(worker_times, iteration)
        self.last_straggler_detected = straggler_detected
        switched = False

        if self.current_mode == CommunicationMode.RING_ALLREDUCE:
            if straggler_detected:
                self.current_mode = CommunicationMode.PARAMETER_SERVER
                self.clean_iterations_count = 0  # Reset counter on switch to PS
                switched = True
                self.switch_history.append((iteration, "PS", self.monitor.last_score))
        elif self.current_mode == CommunicationMode.PARAMETER_SERVER:
            if not straggler_detected:
                self.clean_iterations_count += 1
                if self.clean_iterations_count >= self.hysteresis_window:
                    self.current_mode = CommunicationMode.RING_ALLREDUCE
                    switched = True
                    self.switch_history.append(
                        (iteration, "RING_ALLREDUCE", self.monitor.last_score)
                    )
            else:
                # If a straggler is detected again, reset the clean counter
                self.clean_iterations_count = 0

        return self.current_mode, switched
