
3

Automatic Zoom
 
 
 
PARALLEL AND DISTRIBUTED COMPUTING 
 
AdaptoSGD 
Runtime-Adaptive Communication Strategy Switching 
via Live Straggler Detection 
Full System Design Document 
 
 
Assignment # 02 
 
 
Class: BSCS 13D 
 
 
Name Qalam ID 
Alisha Siddiqui 464647 
Asawer Ayesha 470860 
 
 
 
  
 
 
1. Introduction 
1.1 The Problem 
Distributed SGD training relies on one of two communication architectures.  
Ring AllReduce works well when all workers run at the same speed, because it uses a synchronization 
barrier. Every worker waits for every other worker before moving on. One slow worker stalls the entire 
job. 
Parameter Server removes that barrier. Workers push gradients and pull updates asynchronously, so no 
one waits. But it introduces gradient staleness and concentrates all network traffic at the server. 
Neither architecture adapts at runtime. Ring AllReduce is optimal when all workers run at the same speed 
but becomes catastrophic when one worker is slow. In that case, Parameter Server is the right choice. 
Parameter Server tolerates stragglers via asynchrony but wastes communication bandwidth even when 
workers are running perfectly fine and introduces gradient staleness. 
Real clusters have dynamic worker performance, often within the same training run. A worker slows 
down due to CPU contention, memory pressure, or OS scheduling, then recovers minutes later. No 
existing system performs runtime-adaptive communication strategy switching based on live worker 
performance monitoring. 
1.2 Our Contribution 
AdaptoSGD monitors per-worker iteration times every ten training steps. If a straggler is detected, it 
switches the communication backend to Parameter Server. When the straggler clears, it switches back to 
Ring AllReduce. The switching logic is the novel contribution of this project. 
Our hypothesis: an adaptive system that that monitors per-worker iteration time and dynamically switches 
between synchronous Ring AllReduce and asynchronous Parameter Server gradient aggregation will 
achieve higher training throughput than either strategy alone in the presence of dynamic straggler 
behavior, while maintaining convergence stability within bounded staleness guarantees. 
2. System Architecture 
The system has three components that run in a continuous loop: the Straggler Monitor, the Adaptive 
Switcher, and the Communication Backend. Workers sit below all three and are unaware of which mode 
is active. 
2.1   Architectural Overview 
Component Role File Class/Module 
Component Role / Responsibility File Class / Module 
Worker 
Processes 
(W₁–Wₙ) 
Maintain local model copies, compute 
gradients on data shards, and report 
iteration times to the Monitor. 
— — 
Monitor 
Process 
Collects per-iteration worker timings, 
computes straggler score ( \sigma = 
\frac{t_{max}}{\text{median}(t_i)} ), 
and publishes it. 
monitor.py StragglerMonitor 
Strategy 
Switcher 
Reads straggler score and applies 
switching policy with hysteresis. 
Updates current mode in shared 
memory. 
switcher.py AdaptiveSwitcher 
 
 
Parameter 
Server (PS) 
Active in PS mode; aggregates 
gradients asynchronously and 
broadcasts updated parameters. 
parameter_server.py — 
Ring 
Coordinator 
Active in AllReduce mode; manages 
ring topology and coordinates reduce-
scatter and all-gather phases. 
ring_allreduce.py — 
2.2   Layered Architecture Diagram 
The architecture diagram below shows how data and control signals flow: 
 
2.3 Data Flow Per Iteration 
Each training iteration follows this sequence: 
• Each worker computes a local gradient over its mini-batch (simulated as a NumPy random 
vector). It records its computation time. 
• Worker timing data is sent to the Monitor process via a multiprocessing Queue. 
• Every 10 iterations, the Monitor computes the straggler_score and sends a decision signal to the 
Switcher. 
 
 
• The Switcher updates the current_mode flag (with hysteresis applied). This flag is stored in 
shared memory (multiprocessing.Value). 
• The Backend reads current_mode and routes the gradient aggregation through either Ring 
AllReduce or Parameter Server accordingly. 
• Workers receive the updated model parameters and begin the next iteration. 
 
2.4   State Machine 
The Switcher operates as a finite state machine with two states: ALLREDUCE and PS. Transitions are 
governed by the straggler score and a hysteresis counter δ to prevent oscillation: 
• ALLREDUCE → PS: triggered when σ > τ for any single iteration. Immediate switch. 
• PS → ALLREDUCE: triggered only after σ ≤ τ for δ = 5 consecutive iterations. This hysteresis 
prevents thrashing when straggler behavior is transient. 
• Both transitions are logged with iteration number and σ value for post-experiment analysis. 
 
3. Communication Model 
3.1 Ring AllReduce Mode 
Topology: Workers form a logical ring where worker i sends to worker (i+1) mod N and receives from 
worker (i-1) mod N. 
Operation proceeds in two phases: 
• Scatter-Reduce Phase: Each worker sends and receives N-1 messages, each of size G/N bytes 
(where G is total gradient size). After this phase, each worker holds the partial sum for one 
segment of the gradient. 
• AllGather Phase: Each worker broadcasts its partial result around the ring. Another N-1 messages 
per worker. 
Total messages per worker: 2(N-1). Data sent per worker: 2G(N-1)/N, where G is the total gradient size. 
For N=4 this is 1.5G sent and 1.5G received, which is close to optimal. 
The synchronization barrier is the fatal flaw under straggler conditions. Every worker waits for the 
slowest one. A straggler running at 3.5x normal speed reduces effective cluster throughput to roughly 
29% of the ideal rate. 
3.2 Parameter Server Mode 
Topology: Star topology. One dedicated server process holds the global model parameters. N worker 
processes push gradients and pull updated parameters. 
Total messages per iteration: 2N.  
The server receives N gradients per iteration, which creates a bandwidth bottleneck at large scale.  
Workers push gradients and immediately pull the latest parameters without waiting for other workers. The 
server applies gradients as they arrive (asynchronous SGD). This eliminates straggler blocking at the cost 
of gradient staleness and a bandwidth bottleneck at the server at large scale. 
3.3   Communication Model Summary Table 
Property Ring AllReduce Parameter Server 
Topology Logical ring among N workers Star: N workers to 1 PS 
