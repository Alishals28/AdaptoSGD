#!/usr/bin/env python3
"""
AdaptoSGD: Runtime-Adaptive Communication Strategy Switching via Live Straggler Detection
Parallel and Distributed Computing - Semester Project

This implementation supports three distributed SGD strategies:
1. Ring AllReduce (synchronous baseline)
2. Parameter Server (asynchronous baseline)
3. AdaptoSGD (adaptive switching based on online straggler detection)

The script can run in two execution backends:
- simulated: fast analytical simulation
- multiprocess: process-level worker emulation for true parallel execution
"""

import argparse
import json
import os
import platform
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================


class CommunicationMode(Enum):
    RING_ALLREDUCE = "ring_allreduce"
    PARAMETER_SERVER = "parameter_server"
    ADAPTIVE = "adaptive"


class ExecutionBackend(Enum):
    SIMULATED = "simulated"
    MULTIPROCESS = "multiprocess"


@dataclass
class SystemConfig:
    """Configuration for distributed training simulation and emulation."""

    num_workers: int = 4
    gradient_size: int = 1000
    num_iterations: int = 200
    learning_rate: float = 0.01
    straggler_threshold: float = 1.5
    monitor_window: int = 10
    staleness_bound: int = 5
    random_seed: int = 42
    straggler_delay_factor: float = 3.5
    straggler_start_iter: int = 50
    straggler_end_iter: int = 100
    base_compute_time: float = 0.05
    convergence_threshold: float = 0.1
    execution_backend: str = ExecutionBackend.SIMULATED.value
    sleep_scale: float = 0.0

    # Extended failure scenarios for robustness evaluation.
    failure_start_iter: int = 70
    failure_end_iter: int = 90
    failed_worker_id: int = 1
    network_instability_start: int = 60
    network_instability_end: int = 120
    network_delay_factor: float = 2.0


@dataclass
class MetricsSnapshot:
    """Metrics collected at each iteration."""

    iteration: int
    loss: float
    throughput: float
    straggler_overhead: float
    gradient_staleness: int
    strategy_mode: str
    worker_times: List[float]
    failed_workers: int
    communication_overhead: float
    timestamp: float


# ============================================================================
# DISTRIBUTED WORKER TASK (FOR MULTIPROCESS EMULATION)
# ============================================================================


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
        return np.zeros(gradient_size, dtype=float), float("nan"), True

    rng = np.random.default_rng(seed)
    worker_time = base_compute_time * rng.uniform(0.8, 1.2)
    if straggler_active and worker_id == straggler_worker:
        worker_time = base_compute_time * straggler_delay_factor

    grad = rng.normal(0.0, 0.1, gradient_size)
    if sleep_scale > 0:
        time.sleep(worker_time * sleep_scale)

    return grad, float(worker_time), False


# ============================================================================
# COMPONENT 1: STRAGGLER MONITOR
# ============================================================================


class StragglerMonitor:
    """Online straggler detector based on median-smoothed per-worker timings."""

    def __init__(self, threshold: float = 1.5, window_size: int = 5):
        self.threshold = threshold
        self.window_size = window_size
        self.worker_histories: List[deque] = []
        self.straggler_detected = False
        self.straggler_id = -1

    def update(self, worker_times: List[float], iteration: int) -> Tuple[bool, int, float]:
        """Update monitor with latest iteration times."""
        del iteration  # Iteration kept for API compatibility and future extensions.

        if len(self.worker_histories) != len(worker_times):
            self.worker_histories = [deque(maxlen=self.window_size) for _ in range(len(worker_times))]

        for i, t in enumerate(worker_times):
            self.worker_histories[i].append(t)

        if any(len(h) < 3 for h in self.worker_histories):
            return False, -1, 0.0

        median_times = [float(np.median(h)) for h in self.worker_histories]
        overall_median = float(np.median(median_times))
        max_time = max(median_times)
        slowest_worker = median_times.index(max_time)

        straggler_score = max_time / overall_median if overall_median > 0 else 0.0
        detected = straggler_score > self.threshold

        if detected and not self.straggler_detected:
            self.straggler_detected = True
            self.straggler_id = slowest_worker
        elif not detected and self.straggler_detected:
            self.straggler_detected = False
            self.straggler_id = -1

        return detected, slowest_worker, straggler_score


# ============================================================================
# COMPONENT 2: ADAPTIVE SWITCHER
# ============================================================================


class AdaptiveSwitcher:
    """Adaptive strategy selector between Ring AllReduce and Parameter Server."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.monitor = StragglerMonitor(threshold=config.straggler_threshold, window_size=5)
        self.current_mode = CommunicationMode.RING_ALLREDUCE
        self.switch_history: List[Dict] = []

    def evaluate_and_switch(self, worker_times: List[float], iteration: int) -> CommunicationMode:
        detected, straggler_id, score = self.monitor.update(worker_times, iteration)

        if detected and self.current_mode == CommunicationMode.RING_ALLREDUCE:
            self.current_mode = CommunicationMode.PARAMETER_SERVER
            self.switch_history.append(
                {
                    "iteration": iteration,
                    "from": "ring_allreduce",
                    "to": "parameter_server",
                    "straggler_id": straggler_id,
                    "score": score,
                }
            )
        elif not detected and self.current_mode == CommunicationMode.PARAMETER_SERVER:
            self.current_mode = CommunicationMode.RING_ALLREDUCE
            self.switch_history.append(
                {
                    "iteration": iteration,
                    "from": "parameter_server",
                    "to": "ring_allreduce",
                    "straggler_id": -1,
                    "score": score,
                }
            )

        return self.current_mode


# ============================================================================
# COMPONENT 3: DISTRIBUTED TRAINING ENGINE
# ============================================================================


class DistributedTrainer:
    """Main distributed training simulation/emulation engine."""

    def __init__(self, config: SystemConfig, mode: CommunicationMode):
        self.config = config
        self.mode = mode
        self.metrics: List[MetricsSnapshot] = []
        self.model_params = np.zeros(config.gradient_size)
        self.model_version = 0
        self.straggler_active = False
        self.straggler_worker = -1
        self.switcher: Optional[AdaptiveSwitcher] = None
        self.current_mode = CommunicationMode.RING_ALLREDUCE
        self.switch_history: List[Dict] = []

    def _simulate_straggler(self, iteration: int, condition: str) -> None:
        if condition == "static":
            if iteration == 0:
                self.straggler_active = True
                self.straggler_worker = 0
        elif condition == "dynamic":
            if iteration == self.config.straggler_start_iter:
                self.straggler_active = True
                self.straggler_worker = 0
            elif iteration == self.config.straggler_end_iter:
                self.straggler_active = False
                self.straggler_worker = -1

    def _simulate_failures(self, iteration: int, condition: str) -> Tuple[Set[int], float]:
        failed_workers: Set[int] = set()
        network_factor = 1.0

        if condition == "worker_failure" and self.config.failure_start_iter <= iteration < self.config.failure_end_iter:
            failed_workers.add(self.config.failed_worker_id)

        if condition == "network_instability" and self.config.network_instability_start <= iteration < self.config.network_instability_end:
            network_factor = self.config.network_delay_factor

        return failed_workers, network_factor

    def _apply_gradient(self, gradient: np.ndarray) -> None:
        self.model_params -= self.config.learning_rate * gradient
        self.model_version += 1

    def _get_loss(self) -> float:
        noise = np.random.normal(0.0, 0.01)
        return float(np.mean(self.model_params**2) + noise)

    def _run_worker_step(
        self,
        iteration: int,
        failed_workers: Set[int],
        process_pool: Optional[ProcessPoolExecutor] = None,
    ) -> Tuple[List[np.ndarray], List[float], int]:
        gradients: List[np.ndarray] = []
        worker_times: List[float] = []
        failed_count = 0

        if self.config.execution_backend == ExecutionBackend.MULTIPROCESS.value:
            if process_pool is None:
                raise ValueError("process_pool is required for multiprocess backend")
            args = []
            for worker_id in range(self.config.num_workers):
                args.append(
                    (
                        worker_id,
                        self.config.gradient_size,
                        self.config.base_compute_time,
                        self.straggler_active,
                        self.straggler_worker,
                        self.config.straggler_delay_factor,
                        self.config.random_seed + (iteration * 1000) + worker_id,
                        worker_id in failed_workers,
                        self.config.sleep_scale,
                    )
                )

            results = list(process_pool.map(worker_compute_task, args))

            for grad, worker_time, failed in results:
                gradients.append(grad)
                worker_times.append(float(worker_time))
                if failed:
                    failed_count += 1
        else:
            for worker_id in range(self.config.num_workers):
                is_failed = worker_id in failed_workers
                grad, worker_time, failed = worker_compute_task(
                    (
                        worker_id,
                        self.config.gradient_size,
                        self.config.base_compute_time,
                        self.straggler_active,
                        self.straggler_worker,
                        self.config.straggler_delay_factor,
                        self.config.random_seed + (iteration * 1000) + worker_id,
                        is_failed,
                        0.0,
                    )
                )
                gradients.append(grad)
                worker_times.append(float(worker_time))
                if failed:
                    failed_count += 1

        return gradients, worker_times, failed_count

    def run(self, condition: str) -> List[MetricsSnapshot]:
        self.metrics = []
        self.model_params = np.zeros(self.config.gradient_size)
        self.model_version = 0
        self.current_mode = CommunicationMode.RING_ALLREDUCE
        self.switcher = AdaptiveSwitcher(self.config)
        self.switch_history = []
        self.straggler_active = False
        self.straggler_worker = -1

        start_time = time.time()

        def run_iteration(iteration: int, process_pool: Optional[ProcessPoolExecutor]) -> None:
            self._simulate_straggler(iteration, condition)
            failed_workers, network_factor = self._simulate_failures(iteration, condition)

            gradients, worker_times, failed_count = self._run_worker_step(iteration, failed_workers, process_pool)
            valid_times = [t for t in worker_times if np.isfinite(t)]

            if not valid_times:
                valid_times = [self.config.base_compute_time * self.config.straggler_delay_factor]

            monitor_times = []
            fallback = max(valid_times)
            for worker_time in worker_times:
                if np.isfinite(worker_time):
                    monitor_times.append(worker_time)
                else:
                    monitor_times.append(fallback * 2.0)

            if self.mode == CommunicationMode.ADAPTIVE:
                if iteration % self.config.monitor_window == 0 or self.switcher.monitor.straggler_detected:
                    new_mode = self.switcher.evaluate_and_switch(monitor_times, iteration)
                    if new_mode != self.current_mode:
                        self.current_mode = new_mode
                        self.switch_history.append(
                            {
                                "iteration": iteration,
                                "mode": new_mode.value,
                                "straggler_detected": self.switcher.monitor.straggler_detected,
                            }
                        )

            if self.mode == CommunicationMode.RING_ALLREDUCE or (
                self.mode == CommunicationMode.ADAPTIVE and self.current_mode == CommunicationMode.RING_ALLREDUCE
            ):
                max_time = max(valid_times)
                min_time = min(valid_times)
                idle_time = max_time - min_time

                communication_overhead = (0.01 * self.config.num_workers) * network_factor
                iteration_time = max_time + communication_overhead
                staleness = 0

                valid_grads = [g for i, g in enumerate(gradients) if i not in failed_workers]
                if valid_grads:
                    avg_grad = np.mean(valid_grads, axis=0)
                    self._apply_gradient(avg_grad)
            else:
                communication_overhead = (0.002 + 0.0005 * failed_count) * network_factor
                iteration_time = float(np.mean(valid_times) + communication_overhead)
                idle_time = 0.0

                for worker_id, grad in enumerate(gradients):
                    if worker_id not in failed_workers:
                        self._apply_gradient(grad)

                staleness = min(self.config.staleness_bound, max(0, self.model_version - iteration))

            loss = self._get_loss()
            throughput = 1.0 / iteration_time if iteration_time > 0 else 0.0

            snapshot = MetricsSnapshot(
                iteration=iteration,
                loss=loss,
                throughput=throughput,
                straggler_overhead=idle_time,
                gradient_staleness=staleness,
                strategy_mode=(self.current_mode.value if self.mode == CommunicationMode.ADAPTIVE else self.mode.value),
                worker_times=[float(t) if np.isfinite(t) else -1.0 for t in worker_times],
                failed_workers=failed_count,
                communication_overhead=communication_overhead,
                timestamp=time.time() - start_time,
            )
            self.metrics.append(snapshot)

        if self.config.execution_backend == ExecutionBackend.MULTIPROCESS.value:
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as process_pool:
                for iteration in range(self.config.num_iterations):
                    run_iteration(iteration, process_pool)
        else:
            for iteration in range(self.config.num_iterations):
                run_iteration(iteration, None)

        return self.metrics


# ============================================================================
# EXPERIMENTAL FRAMEWORK
# ============================================================================


class ExperimentRunner:
    """Runs and aggregates experiments across strategies and conditions."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.results: Dict[str, Dict] = {}

    def run_experiment(self, system_type: str, condition: str, seed: int) -> Dict:
        np.random.seed(seed)
        mode = CommunicationMode(system_type)
        trainer = DistributedTrainer(self.config, mode)

        metrics = trainer.run(condition)

        throughput_values = [m.throughput for m in metrics]
        loss_values = [m.loss for m in metrics]
        overhead_values = [m.straggler_overhead for m in metrics]
        staleness_values = [m.gradient_staleness for m in metrics]
        failed_values = [m.failed_workers for m in metrics]
        comm_values = [m.communication_overhead for m in metrics]

        convergence_iter = None
        for m in metrics:
            if m.loss < self.config.convergence_threshold:
                convergence_iter = m.iteration
                break

        return {
            "system": system_type,
            "condition": condition,
            "seed": seed,
            "mean_throughput": float(np.mean(throughput_values)),
            "std_throughput": float(np.std(throughput_values)),
            "final_loss": float(loss_values[-1]),
            "mean_overhead": float(np.mean(overhead_values)),
            "max_overhead": float(max(overhead_values)),
            "mean_staleness": float(np.mean(staleness_values)),
            "max_staleness": int(max(staleness_values)),
            "mean_failed_workers": float(np.mean(failed_values)),
            "mean_communication_overhead": float(np.mean(comm_values)),
            "convergence_iter": convergence_iter,
            "total_time": float(metrics[-1].timestamp),
            "raw_metrics": metrics,
            "switch_history": trainer.switch_history if system_type == CommunicationMode.ADAPTIVE.value else [],
        }

    def run_all(self, num_runs: int = 5, include_extended: bool = True) -> pd.DataFrame:
        systems = [
            CommunicationMode.RING_ALLREDUCE.value,
            CommunicationMode.PARAMETER_SERVER.value,
            CommunicationMode.ADAPTIVE.value,
        ]
        conditions = ["homogeneous", "static", "dynamic"]
        if include_extended:
            conditions.extend(["worker_failure", "network_instability"])

        all_results = []

        for system in systems:
            for condition in conditions:
                print("\n" + "=" * 60)
                print(f"Running: {system.upper()} | Condition: {condition.upper()}")
                print("=" * 60)

                run_results = []
                for run in range(num_runs):
                    result = self.run_experiment(system, condition, self.config.random_seed + run)
                    run_results.append(result)
                    all_results.append(result)
                    print(
                        f"  Run {run + 1}: "
                        f"Throughput={result['mean_throughput']:.2f} iter/s, "
                        f"Loss={result['final_loss']:.4f}, "
                        f"Overhead={result['mean_overhead']:.4f}s"
                    )

                valid_convergence = [r["convergence_iter"] for r in run_results if r["convergence_iter"] is not None]
                agg = {
                    "system": system,
                    "condition": condition,
                    "throughput_mean": float(np.mean([r["mean_throughput"] for r in run_results])),
                    "throughput_std": float(np.std([r["mean_throughput"] for r in run_results])),
                    "loss_mean": float(np.mean([r["final_loss"] for r in run_results])),
                    "overhead_mean": float(np.mean([r["mean_overhead"] for r in run_results])),
                    "staleness_mean": float(np.mean([r["mean_staleness"] for r in run_results])),
                    "failed_workers_mean": float(np.mean([r["mean_failed_workers"] for r in run_results])),
                    "comm_overhead_mean": float(np.mean([r["mean_communication_overhead"] for r in run_results])),
                    "convergence_mean": float(np.mean(valid_convergence)) if valid_convergence else None,
                }
                self.results[f"{system}_{condition}"] = agg

                print("\n  AGGREGATED:")
                print(f"    Throughput: {agg['throughput_mean']:.2f} +- {agg['throughput_std']:.2f} iter/s")
                print(f"    Final Loss: {agg['loss_mean']:.4f}")
                print(f"    Avg Overhead: {agg['overhead_mean']:.4f}s")
                print(f"    Avg Staleness: {agg['staleness_mean']:.2f}")
                print(f"    Failed Workers: {agg['failed_workers_mean']:.2f}")
                print(f"    Comm Overhead: {agg['comm_overhead_mean']:.4f}s")

        return pd.DataFrame(all_results)


# ============================================================================
# VISUALIZATION
# ============================================================================


def _safe_savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")


def generate_core_visualizations(runner: ExperimentRunner, results_df: pd.DataFrame, output_dir: str, show_plots: bool) -> None:
    os.makedirs(f"{output_dir}/figures", exist_ok=True)

    conditions = ["homogeneous", "static", "dynamic"]
    systems = [
        CommunicationMode.RING_ALLREDUCE.value,
        CommunicationMode.PARAMETER_SERVER.value,
        CommunicationMode.ADAPTIVE.value,
    ]
    labels = ["Ring AllReduce", "Parameter Server", "AdaptoSGD"]
    colors = ["#E74C3C", "#3498DB", "#2ECC71"]

    fig = plt.figure(figsize=(20, 12))

    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(conditions))
    width = 0.25
    for i, system in enumerate(systems):
        throughputs = [runner.results[f"{system}_{cond}"]["throughput_mean"] for cond in conditions]
        errors = [runner.results[f"{system}_{cond}"]["throughput_std"] for cond in conditions]
        ax1.bar(x + i * width, throughputs, width, yerr=errors, label=labels[i], color=colors[i], alpha=0.85, capsize=4)
    ax1.set_xlabel("Condition", fontweight="bold")
    ax1.set_ylabel("Throughput (iterations/second)", fontweight="bold")
    ax1.set_title("Throughput Comparison", fontweight="bold")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(["Homogeneous", "Static", "Dynamic"])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    for i, system in enumerate(systems):
        overhead = [runner.results[f"{system}_{cond}"]["overhead_mean"] for cond in conditions]
        ax2.plot(conditions, overhead, marker="o", linewidth=2.5, markersize=7, label=labels[i], color=colors[i])
    ax2.set_xlabel("Condition", fontweight="bold")
    ax2.set_ylabel("Idle Overhead (seconds)", fontweight="bold")
    ax2.set_title("Straggler Overhead", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 3, 3)
    for i, system in enumerate(systems):
        staleness = [runner.results[f"{system}_{cond}"]["staleness_mean"] for cond in conditions]
        ax3.bar(x + i * width, staleness, width, label=labels[i], color=colors[i], alpha=0.85)
    ax3.set_xlabel("Condition", fontweight="bold")
    ax3.set_ylabel("Gradient Staleness", fontweight="bold")
    ax3.set_title("Consistency Tradeoff", fontweight="bold")
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(["Homogeneous", "Static", "Dynamic"])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 3, 4)
    dynamic_results = results_df[results_df["condition"] == "dynamic"]
    for system, label, color in zip(systems, labels, colors):
        run_data = dynamic_results[dynamic_results["system"] == system].iloc[0]
        metrics = run_data["raw_metrics"]
        ax4.plot([m.iteration for m in metrics], [m.loss for m in metrics], label=label, color=color, linewidth=2)
    ax4.axvspan(50, 100, alpha=0.2, color="red", label="Dynamic Straggler")
    ax4.set_xlabel("Iteration", fontweight="bold")
    ax4.set_ylabel("Loss", fontweight="bold")
    ax4.set_title("Loss Curves (Dynamic)", fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(2, 3, 5)
    for system, label, color in zip(systems, labels, colors):
        run_data = dynamic_results[dynamic_results["system"] == system].iloc[0]
        metrics = run_data["raw_metrics"]
        ax5.plot([m.iteration for m in metrics], [m.throughput for m in metrics], label=label, color=color, linewidth=2)
    ax5.axvspan(50, 100, alpha=0.2, color="red")
    ax5.set_xlabel("Iteration", fontweight="bold")
    ax5.set_ylabel("Throughput (iter/s)", fontweight="bold")
    ax5.set_title("Throughput Over Time (Dynamic)", fontweight="bold")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(2, 3, 6)
    adaptive_dynamic = dynamic_results[dynamic_results["system"] == CommunicationMode.ADAPTIVE.value].iloc[0]
    switch_history = adaptive_dynamic["switch_history"]
    if switch_history:
        switch_iterations = [s["iteration"] for s in switch_history]
        switch_modes = [1 if s["mode"] == CommunicationMode.PARAMETER_SERVER.value else 0 for s in switch_history]
        ax6.step(switch_iterations, switch_modes, where="post", linewidth=3, color="#2ECC71")
        ax6.fill_between(switch_iterations, switch_modes, step="post", alpha=0.3, color="#2ECC71")
    ax6.axvspan(50, 100, alpha=0.2, color="red", label="Straggler Active")
    ax6.set_xlabel("Iteration", fontweight="bold")
    ax6.set_ylabel("Communication Mode", fontweight="bold")
    ax6.set_title("AdaptoSGD Switching", fontweight="bold")
    ax6.set_yticks([0, 1])
    ax6.set_yticklabels(["Ring", "PS"])
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    _safe_savefig(f"{output_dir}/figures/core_results.png")
    if show_plots:
        plt.show()
    plt.close(fig)


def generate_extended_failure_visualizations(results_df: pd.DataFrame, output_dir: str, show_plots: bool) -> None:
    os.makedirs(f"{output_dir}/figures", exist_ok=True)

    subset = results_df[results_df["condition"].isin(["worker_failure", "network_instability"])]
    if subset.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = {
        CommunicationMode.RING_ALLREDUCE.value: "Ring AllReduce",
        CommunicationMode.PARAMETER_SERVER.value: "Parameter Server",
        CommunicationMode.ADAPTIVE.value: "AdaptoSGD",
    }
    colors = {
        CommunicationMode.RING_ALLREDUCE.value: "#E74C3C",
        CommunicationMode.PARAMETER_SERVER.value: "#3498DB",
        CommunicationMode.ADAPTIVE.value: "#2ECC71",
    }

    for system in labels:
        data = subset[subset["system"] == system]
        grouped = data.groupby("condition")["mean_throughput"].mean()
        axes[0].bar([f"{labels[system]}\n{c}" for c in grouped.index], grouped.values, color=colors[system], alpha=0.75)

    axes[0].set_ylabel("Mean Throughput", fontweight="bold")
    axes[0].set_title("Failure Scenario Throughput", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, alpha=0.3)

    for system in labels:
        data = subset[subset["system"] == system]
        grouped = data.groupby("condition")["mean_failed_workers"].mean()
        axes[1].plot(grouped.index, grouped.values, marker="o", linewidth=2, label=labels[system], color=colors[system])

    axes[1].set_ylabel("Average Failed Workers", fontweight="bold")
    axes[1].set_title("Failure Exposure", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    _safe_savefig(f"{output_dir}/figures/failure_scenarios.png")
    if show_plots:
        plt.show()
    plt.close(fig)


def run_sensitivity_analysis(base_config: SystemConfig, output_dir: str, show_plots: bool) -> pd.DataFrame:
    """Sensitivity analysis for scalability and adaptation robustness."""
    os.makedirs(f"{output_dir}/data", exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)

    worker_grid = [2, 4, 8]
    threshold_grid = [1.3, 1.5, 1.8]
    delay_grid = [2.5, 3.5, 4.5]

    rows: List[Dict] = []

    for workers in worker_grid:
        for threshold in threshold_grid:
            for delay in delay_grid:
                cfg = SystemConfig(**asdict(base_config))
                cfg.num_workers = workers
                cfg.straggler_threshold = threshold
                cfg.straggler_delay_factor = delay

                runner = ExperimentRunner(cfg)
                result = runner.run_experiment(CommunicationMode.ADAPTIVE.value, "dynamic", cfg.random_seed)
                rows.append(
                    {
                        "num_workers": workers,
                        "straggler_threshold": threshold,
                        "straggler_delay_factor": delay,
                        "throughput": result["mean_throughput"],
                        "final_loss": result["final_loss"],
                        "switch_count": len(result["switch_history"]),
                    }
                )

    sensitivity_df = pd.DataFrame(rows)
    sensitivity_df.to_csv(f"{output_dir}/data/sensitivity_analysis.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for workers in worker_grid:
        subset = sensitivity_df[sensitivity_df["num_workers"] == workers]
        axes[0].plot(
            subset["straggler_threshold"],
            subset["throughput"],
            marker="o",
            linewidth=2,
            label=f"Workers={workers}",
        )
    axes[0].set_xlabel("Straggler Threshold", fontweight="bold")
    axes[0].set_ylabel("Adaptive Throughput", fontweight="bold")
    axes[0].set_title("Sensitivity: Throughput vs Threshold", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    scalability = sensitivity_df.groupby("num_workers")["throughput"].mean()
    axes[1].plot(scalability.index, scalability.values, marker="o", linewidth=3, color="#2ECC71")
    axes[1].set_xlabel("Number of Workers", fontweight="bold")
    axes[1].set_ylabel("Mean Throughput", fontweight="bold")
    axes[1].set_title("Scalability Curve", fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    _safe_savefig(f"{output_dir}/figures/sensitivity_scalability.png")
    if show_plots:
        plt.show()
    plt.close(fig)

    return sensitivity_df


# ============================================================================
# REPRODUCIBILITY / REPORTING
# ============================================================================


def write_reproducibility_manifest(config: SystemConfig, args: argparse.Namespace, output_dir: str) -> None:
    os.makedirs(f"{output_dir}/data", exist_ok=True)

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "config": asdict(config),
        "cli_args": vars(args),
        "seed_policy": "base_seed + run_id; per-worker seed = base + iteration*1000 + worker_id",
    }

    with open(f"{output_dir}/data/reproducibility_manifest.json", "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)


# ============================================================================
# CLI / MAIN
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AdaptoSGD distributed systems evaluation harness")

    parser.add_argument("--num-runs", type=int, default=5, help="Runs per system-condition pair")
    parser.add_argument("--num-iterations", type=int, default=200, help="Training iterations per run")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--gradient-size", type=int, default=1000, help="Model gradient dimensionality")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--backend",
        type=str,
        choices=[ExecutionBackend.SIMULATED.value, ExecutionBackend.MULTIPROCESS.value],
        default=ExecutionBackend.SIMULATED.value,
        help="Execution backend",
    )
    parser.add_argument("--sleep-scale", type=float, default=0.0, help="Wall-clock emulation scale for multiprocess mode")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Disable interactive plot display")
    parser.add_argument("--no-extended", action="store_true", help="Skip worker-failure and network-instability scenarios")
    parser.add_argument("--no-sensitivity", action="store_true", help="Skip sensitivity/scalability analysis")
    parser.add_argument("--quick", action="store_true", help="Quick smoke run")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        args.num_runs = min(args.num_runs, 2)
        args.num_iterations = min(args.num_iterations, 40)
        args.no_sensitivity = True

    print("=" * 70)
    print("ADAPTOSGD: Runtime-Adaptive Communication Strategy Switching")
    print("=" * 70)

    config = SystemConfig(
        num_workers=args.num_workers,
        gradient_size=args.gradient_size,
        num_iterations=args.num_iterations,
        random_seed=args.seed,
        execution_backend=args.backend,
        sleep_scale=args.sleep_scale,
    )

    include_extended = not args.no_extended
    show_plots = not args.no_plots

    conditions_count = 3 + (2 if include_extended else 0)
    total_experiments = 3 * conditions_count * args.num_runs

    print("\nConfiguration:")
    print(f"  Workers: {config.num_workers}")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Backend: {config.execution_backend}")
    print(f"  Straggler Threshold: {config.straggler_threshold}x")
    print(f"  Straggler Delay: {config.straggler_delay_factor}x")
    print(f"  Runs per condition: {args.num_runs}")
    print(f"  Total experiments: {total_experiments}")

    runner = ExperimentRunner(config)
    results_df = runner.run_all(num_runs=args.num_runs, include_extended=include_extended)

    generate_core_visualizations(runner, results_df, args.output_dir, show_plots)
    generate_extended_failure_visualizations(results_df, args.output_dir, show_plots)

    sensitivity_df = None
    if not args.no_sensitivity:
        print("\nRunning sensitivity analysis...")
        sensitivity_df = run_sensitivity_analysis(config, args.output_dir, show_plots)
        print("Sensitivity analysis complete.")

    os.makedirs(f"{args.output_dir}/data", exist_ok=True)

    summary_rows = []
    known_systems = [
        CommunicationMode.RING_ALLREDUCE.value,
        CommunicationMode.PARAMETER_SERVER.value,
        CommunicationMode.ADAPTIVE.value,
    ]
    system_display = {
        CommunicationMode.RING_ALLREDUCE.value: "Ring AllReduce",
        CommunicationMode.PARAMETER_SERVER.value: "Parameter Server",
        CommunicationMode.ADAPTIVE.value: "AdaptoSGD",
    }

    for key, result in sorted(runner.results.items()):
        system = next((name for name in known_systems if key.startswith(name + "_")), None)
        if system is None:
            continue
        condition = key[len(system) + 1 :]
        summary_rows.append(
            {
                "System": system_display[system],
                "Condition": condition.replace("_", " ").title(),
                "Throughput (iter/s)": f"{result['throughput_mean']:.2f} +- {result['throughput_std']:.2f}",
                "Final Loss": f"{result['loss_mean']:.4f}",
                "Avg Overhead (s)": f"{result['overhead_mean']:.4f}",
                "Avg Staleness": f"{result['staleness_mean']:.2f}",
                "Avg Failed Workers": f"{result['failed_workers_mean']:.2f}",
                "Avg Comm Overhead (s)": f"{result['comm_overhead_mean']:.4f}",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{args.output_dir}/data/results_summary.csv", index=False)
    results_df.to_csv(f"{args.output_dir}/data/all_runs.csv", index=False)

    if sensitivity_df is not None:
        sensitivity_df.to_csv(f"{args.output_dir}/data/sensitivity_analysis.csv", index=False)

    write_reproducibility_manifest(config, args, args.output_dir)

    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("\nAll experiments completed successfully.")
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
