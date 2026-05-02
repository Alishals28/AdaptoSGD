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
import dataclasses
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    SystemConfig,
    MetricsSnapshot,
    ExperimentResult,
    CommunicationMode,
    ExecutionBackend,
)
from src.worker import worker_compute_task
from src.monitor import StragglerMonitor
from src.switcher import AdaptiveSwitcher
from src.backends import RingAllReduceBackend, ParameterServerBackend
from src.ssp import SSPEnforcer
from src.analysis import compute_monitoring_overhead


# ============================================================================
# DISTRIBUTED TRAINING ENGINE
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
        
        self.switcher = AdaptiveSwitcher(config)
        self.current_mode = self.switcher.current_mode

        self._ssp = SSPEnforcer(config.ssp_staleness)
        self._ring_backend = RingAllReduceBackend(config.num_workers)
        self._ps_backend = ParameterServerBackend(config.num_workers)

    def _simulate_straggler(self, iteration: int, condition: str) -> None:
        if condition == "static":
            if iteration == 0:
                self.straggler_active = True
                self.straggler_worker = 0
        elif condition == "dynamic":
            if iteration == self.config.straggler_starts_at:
                self.straggler_active = True
                self.straggler_worker = 0
            elif iteration == self.config.straggler_ends_at:
                self.straggler_active = False
                self.straggler_worker = -1

    def _simulate_failures(self, iteration: int, condition: str) -> Tuple[Set[int], float]:
        failed_workers: Set[int] = set()
        network_factor = 1.0

        if condition == "worker_failure" and self.config.failure_starts_at <= iteration < self.config.failure_ends_at:
            failed_workers.add(self.config.failed_worker_id)

        if condition == "network_instability" and self.config.network_instability_starts_at <= iteration < self.config.network_instability_ends_at:
            network_factor = self.config.network_delay_factor

        return failed_workers, network_factor

    def _apply_gradient(self, gradient: np.ndarray) -> None:
        self.model_params -= self.config.learning_rate * gradient
        self.model_version += 1

    # NEW — SYNC_PARAMS broadcast (Section 4.3).
    def _sync_params_broadcast(self) -> None:
        """Simulate broadcasting the global model to all workers.

        In a real system this would send self.model_params to every worker so
        they all start the next Ring AllReduce round with identical parameters,
        re-establishing strong consistency after a PS phase.  Here we record
        the event; the model_params tensor is already the single source of
        truth in this simulation, so no data copy is needed.
        """
        # Mark that a sync happened so MetricsSnapshot can expose it.
        self._sync_event_this_iter = True

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

        if self.config.backend == ExecutionBackend.MULTIPROCESS:
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
                        self.config.seed + (iteration * 1000) + worker_id,
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
                        self.config.seed + (iteration * 1000) + worker_id,
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

        # Reset SSP enforcer for a fresh run.
        self._ssp.reset(self.config.num_workers)

        start_time = time.time()

        def run_iteration(iteration: int, process_pool: Optional[ProcessPoolExecutor]) -> None:
            # Reset per-iteration SYNC_PARAMS flag.
            self._sync_event_this_iter = False

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
                if iteration % self.config.monitoring_window == 0 or self.switcher.last_straggler_detected:
                    new_mode, switched = self.switcher.evaluate_and_switch(monitor_times, iteration)
                    if new_mode != self.current_mode:
                        self.current_mode = new_mode
                        self.switch_history.append(
                            {
                                "iteration": iteration,
                                "mode": new_mode.value,
                                "straggler_detected": self.switcher.last_straggler_detected,
                            }
                        )
                    # Trigger SYNC_PARAMS broadcast when reverting to AllReduce.
                    if switched and new_mode == CommunicationMode.RING_ALLREDUCE:
                        self._sync_params_broadcast()

            # ----------------------------------------------------------------
            # Ring AllReduce path
            # ----------------------------------------------------------------
            if self.mode == CommunicationMode.RING_ALLREDUCE or (
                self.mode == CommunicationMode.ADAPTIVE
                and self.current_mode == CommunicationMode.RING_ALLREDUCE
            ):
                
                avg_grad, iter_time = self._ring_backend.aggregate(
                    [g for i, g in enumerate(gradients) if i not in failed_workers],
                    valid_times,
                )
                
                communication_overhead = (0.01 * self.config.num_workers) * network_factor
                iteration_time = iter_time + communication_overhead
                staleness = 0

                if avg_grad.size > 0:
                    self._apply_gradient(avg_grad)

            # ----------------------------------------------------------------
            # Parameter Server path
            # ----------------------------------------------------------------
            else:
                staleness = self._ssp.get_staleness(self.model_version)
                
                # In PS mode, we process gradients as they come.
                # We simulate this by iterating through workers one by one.
                total_comm_time = 0
                for worker_id, (grad, w_time) in enumerate(zip(gradients, worker_times)):
                    if worker_id in failed_workers or not np.isfinite(w_time):
                        continue

                    # SSP enforcement
                    if not self._ssp.is_blocked(worker_id):
                        self._apply_gradient(grad)
                        self._ssp.advance(worker_id)
                        total_comm_time += (0.002 * network_factor)

                communication_overhead = total_comm_time
                iteration_time = float(np.mean(valid_times)) + total_comm_time


            loss = self._get_loss()
            throughput = 1.0 / iteration_time if iteration_time > 0 else 0.0

            snapshot = MetricsSnapshot(
                iteration=iteration,
                mode=self.current_mode.value,
                loss=loss,
                throughput=throughput,
                total_time=iteration_time,
                straggler_active=self.straggler_active,
                straggler_detected=self.switcher.last_straggler_detected,
                straggler_score=self.switcher.monitor.last_score,
                staleness=staleness,
                worker_times=[float(t) if np.isfinite(t) else -1.0 for t in worker_times],
                failed_workers=failed_workers,
                communication_overhead=communication_overhead,
                timestamp=time.time() - start_time,
                sync_params_event=self._sync_event_this_iter,  # NEW
            )
            self.metrics.append(snapshot)

        if self.config.backend == ExecutionBackend.MULTIPROCESS:
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

    def __init__(self, base_config: SystemConfig):
        self.base_config = base_config
        self.results: List[ExperimentResult] = []

    def run_experiment(
        self, mode: CommunicationMode, condition: str
    ) -> ExperimentResult:
        """Runs a single experiment for a given mode and condition."""
        config = dataclasses.replace(self.base_config)
        print(f"🚀 Running experiment: {mode.value} under {condition}...")

        trainer = DistributedTrainer(config, mode)
        
        start_time = time.monotonic()
        history = trainer.run(condition)
        total_runtime = time.monotonic() - start_time

        avg_throughput = np.mean([m.throughput for m in history if m.throughput > 0])
        mean_staleness = float(np.mean([m.staleness for m in history])) if history else 0.0
        capped_staleness = min(mean_staleness, config.staleness_bound)
        staleness_factor = max(0.0, 1 - (capped_staleness / config.staleness_bound))
        effective_throughput = avg_throughput * staleness_factor
        
        time_to_convergence = next(
            (m.iteration for m in history if m.loss < 0.1), None
        )

        # --- NEW: Calculate Switch Latency ---
        switch_latency = None
        if mode == CommunicationMode.ADAPTIVE and trainer.switcher.switch_history:
            # Find the first switch to Parameter Server mode
            first_switch = next(
                (s for s in trainer.switcher.switch_history if s[1] == "PS"), None
            )
            if first_switch:
                switch_iteration = first_switch[0]
                # Latency is the difference from when the straggler was introduced
                if switch_iteration >= config.straggler_starts_at:
                    switch_latency = switch_iteration - config.straggler_starts_at
        # --- END NEW ---

        result = ExperimentResult(
            config=config,
            mode=mode,
            condition=condition,
            history=history,
            total_runtime=total_runtime,
            avg_throughput=avg_throughput,
            effective_throughput=effective_throughput,
            time_to_convergence=time_to_convergence,
            switch_latency=switch_latency,
        )
        self.results.append(result)
        
        print(f"   -> Finished in {total_runtime:.2f}s. Avg throughput: {avg_throughput:.2f} iter/s.")
        if time_to_convergence:
            print(f"   -> Reached convergence at iteration {time_to_convergence}.")
        if switch_latency is not None:
            print(f"   -> Detected switch latency: {switch_latency} iterations.")

        return result

    def run_all(self, conditions: List[str], num_runs: int) -> pd.DataFrame:
        all_results: List[ExperimentResult] = []
        for mode in [CommunicationMode.RING_ALLREDUCE, CommunicationMode.PARAMETER_SERVER, CommunicationMode.ADAPTIVE]:
            for condition in conditions:
                for _ in range(num_runs):
                    all_results.append(self.run_experiment(mode, condition))

        return pd.DataFrame([res.__dict__ for res in all_results])


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
        CommunicationMode.RING_ALLREDUCE,
        CommunicationMode.PARAMETER_SERVER,
        CommunicationMode.ADAPTIVE,
    ]
    labels = {
        CommunicationMode.RING_ALLREDUCE: "Ring AllReduce",
        CommunicationMode.PARAMETER_SERVER: "Parameter Server",
        CommunicationMode.ADAPTIVE: "AdaptoSGD",
    }
    colors = {
        CommunicationMode.RING_ALLREDUCE: "#E74C3C",
        CommunicationMode.PARAMETER_SERVER: "#3498DB",
        CommunicationMode.ADAPTIVE: "#2ECC71",
    }

    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(2, 4)

    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(conditions))
    width = 0.25
    for i, system in enumerate(systems):
        throughputs = [
            results_df[(results_df["mode"] == system) & (results_df["condition"] == cond)]["avg_throughput"].mean()
            for cond in conditions
        ]
        errors = [
            results_df[(results_df["mode"] == system) & (results_df["condition"] == cond)]["avg_throughput"].std()
            for cond in conditions
        ]
        ax1.bar(x + i * width, throughputs, width, yerr=errors, label=labels[system], color=colors[system], alpha=0.85, capsize=4)
    ax1.set_xlabel("Condition", fontweight="bold")
    ax1.set_ylabel("Throughput (iterations/second)", fontweight="bold")
    ax1.set_title("A. Throughput Comparison", fontweight="bold", loc="left")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([c.title() for c in conditions])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- NEW: Convergence Comparison Plot ---
    ax_conv = fig.add_subplot(gs[0, 1])
    for i, system in enumerate(systems):
        convergence_times = []
        for cond in conditions:
            subset = results_df[(results_df["mode"] == system) & (results_df["condition"] == cond)]
            # Filter out runs that did not converge (value is None or NaN)
            valid_times = subset["time_to_convergence"].dropna()
            convergence_times.append(valid_times.mean() if not valid_times.empty else runner.base_config.num_iterations)

        ax_conv.bar(x + i * width, convergence_times, width, label=labels[system], color=colors[system], alpha=0.85)
    
    ax_conv.set_xlabel("Condition", fontweight="bold")
    ax_conv.set_ylabel("Iterations to Converge", fontweight="bold")
    ax_conv.set_title("B. Time to Convergence", fontweight="bold", loc="left")
    ax_conv.set_xticks(x + width)
    ax_conv.set_xticklabels([c.title() for c in conditions])
    ax_conv.legend()
    ax_conv.grid(True, alpha=0.3)
    # --- END NEW ---

    ax3 = fig.add_subplot(gs[0, 2])
    for system in systems:
        staleness_data = []
        for cond in conditions:
            subset = results_df[(results_df["mode"] == system) & (results_df["condition"] == cond)]
            per_run = []
            for _, row in subset.iterrows():
                history = row["history"]
                if history:
                    per_run.append(float(np.mean([m.staleness for m in history])))
            staleness_data.append(float(np.mean(per_run)) if per_run else float("nan"))
        ax3.plot(conditions, staleness_data, marker="o", linewidth=2.5, markersize=7, label=labels[system], color=colors[system])
    ax3.set_xlabel("Condition", fontweight="bold")
    ax3.set_ylabel("Average Gradient Staleness", fontweight="bold")
    ax3.set_title("C. Consistency Tradeoff", fontweight="bold", loc="left")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax_eff = fig.add_subplot(gs[0, 3])
    for i, system in enumerate(systems):
        eff_values = []
        for cond in conditions:
            subset = results_df[(results_df["mode"] == system) & (results_df["condition"] == cond)]
            eff_values.append(float(np.mean(subset["effective_throughput"])) if not subset.empty else float("nan"))
        ax_eff.bar(x + i * width, eff_values, width, label=labels[system], color=colors[system], alpha=0.85)
    ax_eff.set_xlabel("Condition", fontweight="bold")
    ax_eff.set_ylabel("Effective Throughput", fontweight="bold")
    ax_eff.set_title("D. Effective Throughput", fontweight="bold", loc="left")
    ax_eff.set_xticks(x + width)
    ax_eff.set_xticklabels([c.title() for c in conditions])
    ax_eff.legend()
    ax_eff.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0])
    dynamic_results = results_df[results_df["condition"] == "dynamic"]
    for system in systems:
        run_data = dynamic_results[dynamic_results["mode"] == system]
        if not run_data.empty:
            history = run_data.iloc[0]["history"]
            ax4.plot([m.iteration for m in history], [m.loss for m in history], label=labels[system], color=colors[system], linewidth=2)
    
    ax4.axvspan(runner.base_config.straggler_starts_at, runner.base_config.straggler_ends_at, alpha=0.2, color="red", label="Dynamic Straggler")
    ax4.set_xlabel("Iteration", fontweight="bold")
    ax4.set_ylabel("Loss", fontweight="bold")
    ax4.set_title("D. Loss Curves (Dynamic)", fontweight="bold", loc="left")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    for system in systems:
        run_data = dynamic_results[dynamic_results["mode"] == system]
        if not run_data.empty:
            history = run_data.iloc[0]["history"]
            ax5.plot([m.iteration for m in history], [m.throughput for m in history], label=labels[system], color=colors[system], linewidth=2)
    
    ax5.axvspan(runner.base_config.straggler_starts_at, runner.base_config.straggler_ends_at, alpha=0.2, color="red")
    ax5.set_xlabel("Iteration", fontweight="bold")
    ax5.set_ylabel("Throughput (iter/s)", fontweight="bold")
    ax5.set_title("E. Throughput Over Time (Dynamic)", fontweight="bold", loc="left")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    adaptive_dynamic = dynamic_results[dynamic_results["mode"] == CommunicationMode.ADAPTIVE]
    if not adaptive_dynamic.empty:
        history = adaptive_dynamic.iloc[0]["history"]
        switch_points = [h for h in history if h.mode != history[h.iteration-1].mode] if len(history) > 1 else [] # Simplified
        
        mode_changes = []
        last_mode = None
        for h in history:
            if h.mode != last_mode:
                mode_changes.append((h.iteration, 1 if h.mode == 'parameter_server' else 0))
                last_mode = h.mode
        
        if mode_changes:
            its, modes = zip(*mode_changes)
            ax6.step(its, modes, where="post", linewidth=3, color=colors[CommunicationMode.ADAPTIVE])
            ax6.fill_between(its, modes, step="post", alpha=0.3, color=colors[CommunicationMode.ADAPTIVE])

    ax6.axvspan(runner.base_config.straggler_starts_at, runner.base_config.straggler_ends_at, alpha=0.2, color="red", label="Straggler Active")
    ax6.set_xlabel("Iteration", fontweight="bold")
    ax6.set_ylabel("Communication Mode", fontweight="bold")
    ax6.set_title("F. AdaptoSGD Switching", fontweight="bold", loc="left")
    ax6.set_yticks([0, 1])
    ax6.set_yticklabels(["Ring", "PS"])
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Placeholder for the 4th plot in the bottom row
    ax_placeholder2 = fig.add_subplot(gs[1, 3])
    ax_placeholder2.axis('off')


    _safe_savefig(f"{output_dir}/figures/core_results.png")
    if show_plots:
        plt.show()
    plt.close(fig)


def generate_extended_failure_visualizations(results_df: pd.DataFrame, output_dir: str, show_plots: bool) -> None:
    os.makedirs(f"{output_dir}/figures", exist_ok=True)

    failure_conditions = ["worker_failure", "network_instability"]
    subset = results_df[results_df["condition"].isin(failure_conditions)]
    if subset.empty:
        print("No failure scenario data to visualize.")
        return

    systems = [
        CommunicationMode.RING_ALLREDUCE,
        CommunicationMode.PARAMETER_SERVER,
        CommunicationMode.ADAPTIVE,
    ]
    labels = {
        CommunicationMode.RING_ALLREDUCE: "Ring AllReduce",
        CommunicationMode.PARAMETER_SERVER: "Parameter Server",
        CommunicationMode.ADAPTIVE: "AdaptoSGD",
    }
    colors = {
        CommunicationMode.RING_ALLREDUCE: "#E74C3C",
        CommunicationMode.PARAMETER_SERVER: "#3498DB",
        CommunicationMode.ADAPTIVE: "#2ECC71",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Extended Analysis: Failure & Instability Scenarios", fontsize=16, fontweight="bold")

    # 1. Throughput under failure
    ax = axes[0, 0]
    x = np.arange(len(failure_conditions))
    width = 0.25
    for i, system in enumerate(systems):
        throughputs = [
            subset[(subset["mode"] == system) & (subset["condition"] == cond)]["avg_throughput"].mean()
            for cond in failure_conditions
        ]
        ax.bar(x + i * width, throughputs, width, label=labels[system], color=colors[system], alpha=0.85)
    ax.set_ylabel("Mean Throughput (iter/s)", fontweight="bold")
    ax.set_title("A. Throughput Under Failure", fontweight="bold", loc="left")
    ax.set_xticks(x + width)
    ax.set_xticklabels([c.replace("_", " ").title() for c in failure_conditions])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Loss stability during failure
    ax = axes[0, 1]
    failure_results = results_df[results_df["condition"] == "worker_failure"]
    for system in systems:
        run_data = failure_results[failure_results["mode"] == system]
        if not run_data.empty:
            history = run_data.iloc[0]["history"]
            ax.plot([m.iteration for m in history], [m.loss for m in history], label=labels[system], color=colors[system], linewidth=2)
    
    config = failure_results.iloc[0]["config"]
    ax.axvspan(config.failure_starts_at, config.failure_ends_at, alpha=0.2, color="grey", label="Failure Window")
    ax.set_xlabel("Iteration", fontweight="bold")
    ax.set_ylabel("Loss", fontweight="bold")
    ax.set_title("B. Loss Stability (Worker Failure)", fontweight="bold", loc="left")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Recovery time after failure
    ax = axes[1, 0]
    recovery_times = {}
    for system in systems:
        run_data = failure_results[failure_results["mode"] == system]
        if not run_data.empty:
            history = run_data.iloc[0]["history"]
            config = run_data.iloc[0]["config"]
            
            post_failure_throughput = [
                m.throughput for m in history if m.iteration > config.failure_ends_at
            ]
            pre_failure_throughput = [
                m.throughput for m in history if m.iteration < config.failure_starts_at
            ]
            
            avg_pre_failure_tp = np.mean(pre_failure_throughput) if pre_failure_throughput else 0
            
            recovered_iter = next(
                (m.iteration for m in history if m.iteration > config.failure_ends_at and m.throughput >= avg_pre_failure_tp * 0.9),
                None
            )
            
            if recovered_iter:
                recovery_times[labels[system]] = recovered_iter - config.failure_ends_at
            else:
                recovery_times[labels[system]] = -1 # Did not recover

    ax.bar(recovery_times.keys(), recovery_times.values(), color=[colors[s] for s in systems])
    ax.set_ylabel("Iterations to Recover Throughput", fontweight="bold")
    ax.set_title("C. Recovery Time After Failure", fontweight="bold", loc="left")
    ax.grid(True, axis='y', alpha=0.3)


    # 4. How AdaptoSGD vs PS handle a failed worker differently
    ax = axes[1, 1]
    ps_failed_workers = failure_results[failure_results["mode"] == CommunicationMode.PARAMETER_SERVER].iloc[0]["history"]
    adapto_failed_workers = failure_results[failure_results["mode"] == CommunicationMode.ADAPTIVE].iloc[0]["history"]

    ax.plot([m.iteration for m in ps_failed_workers], [len(m.failed_workers) for m in ps_failed_workers], label="Parameter Server", color=colors[CommunicationMode.PARAMETER_SERVER])
    ax.plot([m.iteration for m in adapto_failed_workers], [len(m.failed_workers) for m in adapto_failed_workers], label="AdaptoSGD", color=colors[CommunicationMode.ADAPTIVE], linestyle='--')
    
    ax.axvspan(config.failure_starts_at, config.failure_ends_at, alpha=0.2, color="grey", label="Failure Window")
    ax.set_xlabel("Iteration", fontweight="bold")
    ax.set_ylabel("Number of Failed Workers Detected", fontweight="bold")
    ax.set_title("D. Failed Worker Handling", fontweight="bold", loc="left")
    ax.legend()
    ax.grid(True, alpha=0.3)


    _safe_savefig(f"{output_dir}/figures/extended_failure_analysis.png")
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
                result = runner.run_experiment(CommunicationMode.ADAPTIVE, "dynamic")
                rows.append(
                    {
                        "num_workers": workers,
                        "straggler_threshold": threshold,
                        "straggler_delay_factor": delay,
                        "throughput": result.avg_throughput,
                        "final_loss": result.history[-1].loss if result.history else float("nan"),
                        "switch_count": sum(
                            1
                            for idx in range(1, len(result.history))
                            if result.history[idx].mode != result.history[idx - 1].mode
                        ),
                        "sync_event_count": sum(1 for m in result.history if m.sync_params_event),
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

    scalability_df = sensitivity_df.groupby("num_workers")["throughput"].mean().reset_index()
    
    # Calculate speedup relative to the first worker count
    if not scalability_df.empty:
        baseline_throughput = scalability_df.iloc[0]["throughput"]
        scalability_df["speedup"] = scalability_df["throughput"] / baseline_throughput
        scalability_df["efficiency"] = scalability_df["speedup"] / scalability_df["num_workers"]

        # Speedup Plot
        ax_speedup = axes[1].twinx()
        ax_speedup.plot(scalability_df["num_workers"], scalability_df["speedup"], marker='s', linestyle='--', color='purple', label='Speedup')
        ax_speedup.set_ylabel("Speedup", fontweight="bold", color='purple')
        ax_speedup.tick_params(axis='y', labelcolor='purple')
        
        # Efficiency Plot
        fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))

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

    config_dict = asdict(config)
    config_dict["backend"] = config.backend.value

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "config": config_dict,
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
    parser.add_argument("--output-dir", type=str, default="../output", help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Disable interactive plot display")
    parser.add_argument("--no-extended", action="store_true", help="Skip worker-failure and network-instability scenarios")
    parser.add_argument("--no-sensitivity", action="store_true", help="Skip sensitivity/scalability analysis")
    parser.add_argument("--quick", action="store_true", help="Quick smoke run")
    # NEW — expose hysteresis window via CLI.
    parser.add_argument("--hysteresis-window", type=int, default=5, help="Consecutive clean iterations before PS→Ring revert")

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
        seed=args.seed,
        backend=ExecutionBackend(args.backend),
        hysteresis_window=args.hysteresis_window,  # NEW
    )

    include_extended = not args.no_extended
    show_plots = not args.no_plots

    conditions_count = 3 + (2 if include_extended else 0)
    total_experiments = 3 * conditions_count * args.num_runs

    print("\nConfiguration:")
    print(f"  Workers: {config.num_workers}")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Backend: {config.backend}")
    print(f"  Straggler Threshold: {config.straggler_threshold}x")
    print(f"  Straggler Delay: {config.straggler_delay_factor}x")
    print(f"  Hysteresis Window: {config.hysteresis_window} iters")  # NEW
    print(f"  Staleness Bound (SSP): {config.ssp_staleness}")       # NEW
    print(f"  Runs per condition: {args.num_runs}")
    print(f"  Total experiments: {total_experiments}")

    runner = ExperimentRunner(config)
    results_list = []
    
    conditions_to_run = ["homogeneous", "static", "dynamic"]
    if include_extended:
        conditions_to_run.extend(["worker_failure", "network_instability"])

    for mode in [CommunicationMode.RING_ALLREDUCE, CommunicationMode.PARAMETER_SERVER, CommunicationMode.ADAPTIVE]:
        for condition in conditions_to_run:
            for _ in range(args.num_runs):
                results_list.append(runner.run_experiment(mode, condition))

    results_df = pd.DataFrame([res.__dict__ for res in results_list])
    
    # --- C-6 Trade-off Analysis ---
    overhead_abs, overhead_pct = compute_monitoring_overhead(results_df)
    print("\n" + "="*70)
    print("C-6 Novelty Analysis: Cost of Monitoring")
    print(f"  AdaptoSGD throughput (homogeneous): {results_df[(results_df['mode'] == CommunicationMode.ADAPTIVE) & (results_df['condition'] == 'homogeneous')]['avg_throughput'].mean():.2f} iter/s")
    print(f"  Ring AllReduce throughput (homogeneous): {results_df[(results_df['mode'] == CommunicationMode.RING_ALLREDUCE) & (results_df['condition'] == 'homogeneous')]['avg_throughput'].mean():.2f} iter/s")
    print(f"  -> Absolute overhead: {overhead_abs:.2f} iter/s")
    print(f"  -> Relative overhead: {overhead_pct:.2f}%")
    print("="*70)
    # --- End C-6 Analysis ---

    generate_core_visualizations(runner, results_df, args.output_dir, show_plots)
    generate_extended_failure_visualizations(results_df, args.output_dir, show_plots)

    sensitivity_df = None
    if not args.no_sensitivity:
        print("\nRunning sensitivity analysis...")
        sensitivity_df = run_sensitivity_analysis(config, args.output_dir, show_plots)
        print("Sensitivity analysis complete.")
        if sensitivity_df is not None:
            generate_amdahl_plot(sensitivity_df, args.output_dir, show_plots)

    os.makedirs(f"{args.output_dir}/data", exist_ok=True)

    summary_rows = []
    mode_values = results_df["mode"].apply(
        lambda mode: mode.value if hasattr(mode, "value") else mode
    )
    results_df = results_df.assign(mode_value=mode_values)
    for (mode_value, condition), subset in results_df.groupby(["mode_value", "condition"], sort=False):
        throughputs = subset["avg_throughput"].tolist()
        throughput_mean = float(np.mean(throughputs)) if throughputs else float("nan")
        throughput_std = float(np.std(throughputs)) if throughputs else float("nan")

        final_losses = []
        staleness_vals = []
        overhead_vals = []
        failed_counts = []
        comm_overheads = []
        for _, row in subset.iterrows():
            history = row["history"]
            if history:
                final_losses.append(history[-1].loss)
                staleness_vals.append(float(np.mean([m.staleness for m in history])))
                overhead_vals.append(float(np.mean([m.communication_overhead for m in history])))
                failed_counts.append(float(np.mean([len(m.failed_workers) for m in history])))
                comm_overheads.append(float(np.mean([m.communication_overhead for m in history])))

        summary_rows.append(
            {
            "System": mode_value.replace("_", " ").title(),
                "Condition": condition.replace("_", " ").title(),
                "Throughput (iter/s)": f"{throughput_mean:.2f} +- {throughput_std:.2f}",
                "Effective Throughput": f"{np.mean(subset['effective_throughput']):.2f}",
                "Final Loss": f"{np.mean(final_losses):.4f}" if final_losses else "nan",
                "Avg Overhead (s)": f"{np.mean(overhead_vals):.4f}" if overhead_vals else "nan",
                "Avg Staleness": f"{np.mean(staleness_vals):.2f}" if staleness_vals else "nan",
                "Avg Failed Workers": f"{np.mean(failed_counts):.2f}" if failed_counts else "nan",
                "Avg Comm Overhead (s)": f"{np.mean(comm_overheads):.4f}" if comm_overheads else "nan",
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