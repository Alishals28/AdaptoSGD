from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Set

import numpy as np


class CommunicationMode(Enum):
    RING_ALLREDUCE = "ring_allreduce"
    PARAMETER_SERVER = "parameter_server"
    ADAPTIVE = "adaptive"


class ExecutionBackend(Enum):
    SIMULATED = "simulated"
    MULTIPROCESS = "multiprocess"


@dataclass
class SystemConfig:
    """System-wide configuration parameters."""

    num_workers: int = 4
    gradient_size: int = 25_000_000  # ResNet-50 size
    learning_rate: float = 0.01
    num_iterations: int = 200
    base_compute_time: float = 0.1  # Seconds
    
    # Straggler simulation
    straggler_starts_at: int = 50
    straggler_ends_at: int = 150
    straggler_delay_factor: float = 3.5
    straggler_worker: int = 0

    # Failure simulation
    failure_starts_at: int = 75
    failure_ends_at: int = 125
    failed_worker_id: int = 0
    network_instability_starts_at: int = 90
    network_instability_ends_at: int = 110
    network_delay_factor: float = 1.5
    
    # Adaptive switching
    monitoring_window: int = 10  # Check every N iterations
    straggler_threshold: float = 1.5
    hysteresis_window: int = 5

    # SSP staleness bound for PS mode
    ssp_staleness: int = 4
    staleness_bound: int = 5

    # Visualization
    show_plots: bool = True
    
    # Execution
    backend: ExecutionBackend = ExecutionBackend.MULTIPROCESS
    seed: int = 42
    sleep_scale: float = 0.0


@dataclass
class MetricsSnapshot:
    """A snapshot of system metrics at a single iteration."""

    iteration: int
    mode: str
    loss: float
    throughput: float
    total_time: float
    straggler_active: bool
    straggler_detected: bool
    straggler_score: float
    staleness: int
    worker_times: List[float]
    failed_workers: Set[int]
    communication_overhead: float
    timestamp: float
    sync_params_event: bool


@dataclass
class ExperimentResult:
    """Aggregated results from a single experimental run."""
    config: SystemConfig
    mode: CommunicationMode
    condition: str
    history: List[MetricsSnapshot]
    total_runtime: float
    avg_throughput: float
    time_to_convergence: Optional[int]
    effective_throughput: float = 0.0
    switch_latency: Optional[int] = None
