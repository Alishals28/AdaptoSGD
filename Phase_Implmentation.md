Critical (code broken)
High (missing feature)
Medium (needs improvement)
Structure / organization
1

Phase 1 — Fix core code correctness
2 bugs, ~50 lines of changes
Critical bugs — breaks the design doc promise
!
Fix hysteresis counter in AdaptiveSwitcher
The design doc says PS → AllReduce only happens after σ ≤ τ for δ=5 consecutive clean iterations. The code switches back instantly on the first clean check. Add a clean_count counter. When switching TO PS: reset it. Each clean check increments it. Only switch back when it hits 5.
switcher.py / AdaptiveSwitcher.evaluate_and_switch()
!
Fix StragglerMonitor — don't clear detection in one shot
The monitor sets straggler_detected = False on the very first reading below threshold. This bypasses the hysteresis logic even if you fix the switcher. The monitor should only report "not detected" after N consecutive clean readings, OR leave the cooldown entirely to the Switcher and always return the raw score. Cleanest fix: monitor always returns the raw score; let the Switcher own the hysteresis state.
monitor.py / StragglerMonitor.update()
Missing metric — promised in proposal

Implement switch latency metric
Your proposal defines this metric: "Time from straggler onset to confirmed mode change." The straggler starts at iteration 50. The system switches at some later iteration X. Switch latency = X - 50. You already log switch_history with iteration numbers. Just compute and store the lag. Report it in the summary.
ExperimentRunner.run_experiment() — add switch_latency field

2
Phase 2 — Modularize into src/ structure
Refactor only, no new logic
File structure required by design doc
→
Create src/monitor.py
Move StragglerMonitor class here. The design doc explicitly maps this to monitor.py. This isn't optional — your professor has a component table that lists this exact file.
StragglerMonitor → src/monitor.py
→
Create src/switcher.py
Move AdaptiveSwitcher here. Include the now-fixed hysteresis logic.
AdaptiveSwitcher → src/switcher.py
→
Create src/backends.py
Extract the Ring AllReduce and Parameter Server communication logic into clean classes: RingAllReduceBackend and ParameterServerBackend. Each class has an aggregate(gradients, worker_times, ...) method that returns the averaged gradient + timing stats. The design doc maps ring_allreduce.py and parameter_server.py — consolidate them here.
Ring + PS logic → src/backends.py
→
Refactor adaptosgd_complete.py → main.py
The main file becomes an orchestrator only. It imports from src/, wires things together, runs experiments, generates plots. No business logic lives here anymore.
adaptosgd_complete.py → main.py (imports src/*)
→
Create src/config.py and src/worker.py
Move SystemConfig, MetricsSnapshot, CommunicationMode, ExecutionBackend into config.py. Move worker_compute_task() into worker.py. Keeps imports clean and mirrors the design doc component table exactly.
src/config.py + src/worker.py
3

Phase 3 — Add missing analysis & visualizations
Required by guidelines for final presentation
Missing from guidelines Deliverable 3
A
Add Amdahl's Law performance modeling
Guidelines Deliverable 2 explicitly requires "performance modeling (Amdahl, Gustafson, or message complexity)." You have message complexity math in the doc. But Amdahl's Law is the one your professor is likely checking. Add a function that plots theoretical speedup vs workers using S(N) = 1 / (f + (1-f)/N) where f is the serial fraction (barrier wait time). Show where your experimental results fall on this curve. This is a ~30-line addition to the visualization code.
New: generate_amdahl_plot() in main.py
B
Improve scalability visualization — speedup + efficiency
Current scalability plot shows raw throughput vs workers. That's weak. You need: (1) Speedup = throughput(N) / throughput(1). (2) Parallel efficiency = speedup / N. Show all 3 systems on both metrics. The efficiency curve is especially powerful — it shows that AdaptoSGD maintains efficiency under stragglers while AllReduce collapses.
Extend run_sensitivity_analysis() — add speedup/efficiency subplot
C
Add convergence comparison visualization
Guidelines require "measurable improvement." Loss curves alone aren't enough. Add a dedicated plot: iteration-to-convergence (loss below threshold) for all 3 systems across all 3 conditions, as a grouped bar chart with error bars. This directly answers "is AdaptoSGD better?" with a number.
New subplot in generate_core_visualizations()
D
Add C-6 trade-off analysis — cost of monitoring
For C-6 you must "analyze trade-offs." The trade-off is: AdaptoSGD adds monitoring overhead. Under homogeneous conditions (no stragglers), does that overhead cost anything? Compute: AdaptoSGD throughput in homogeneous condition vs pure AllReduce throughput. That difference IS the cost of being adaptive. If it's negligible, that's your C-6 argument: "we pay near-zero overhead for the ability to adapt."
New: compute_monitoring_overhead() — reported in summary
E
Strengthen failure scenario analysis
Worker failure and network instability scenarios exist but the analysis is thin — just a throughput bar. Add: (1) loss stability during failure window, (2) recovery time after failure clears, (3) how AdaptoSGD vs PS handle a failed worker differently. Show that AdaptoSGD doesn't catastrophically fail when a worker drops.
Extend generate_extended_failure_visualizations()
4
Phase 4 — Final report & presentation prep
Documentation only
i
Document the monitoring frequency decision
The code checks every 10 iterations normally, but every iteration when a straggler is active. This is smart behavior but it's undocumented anywhere. Add it to the design doc and be ready to explain it in the presentation. It's actually a good design choice — just needs to be intentional, not accidental.
ii
Write dedicated C-6 novelty section
20% of marks. Must include: (1) what is novel — the runtime switching, not either backend alone, (2) experimental proof — specific numbers from dynamic straggler condition, (3) trade-off table — monitoring cost vs throughput gain, (4) comparison to BytePS — why theirs is static and yours is dynamic.
iii
Ensure reproducibility manifest covers everything
The manifest already logs platform, versions, seed policy. Make sure it also logs the hysteresis parameter (δ=5) and the monitor window (every 10 iterations). Every hyperparameter that affects results must be logged. Guidelines say all results must be reproducible.
✕
What to skip — not worth the time
Per your professor's constraints
✕
Real socket / gRPC / mpi4py networking
The plan mentions this but your professor's guidelines say "projects must demonstrate distributed behavior across multiple processes." You already have multiprocessing via ProcessPoolExecutor. That satisfies the constraint. Adding real sockets is a massive scope increase with zero grade benefit. Your simulation already models the communication math correctly. Skip it.
✕
Real neural network training
Replacing the numpy gradient simulation with actual PyTorch training adds weeks of work and introduces GPU dependency. Your current simulation models the distributed systems behavior correctly. The professor said "pure ML without distributed evaluation is not permitted" — you have the distributed evaluation. The ML fidelity is not the point.