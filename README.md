# 🚀 AdaptoSGD  
### Runtime-Adaptive Communication Strategy Switching via Live Straggler Detection

---

## 📌 Overview

**AdaptoSGD** is a distributed training system that dynamically switches between **Ring AllReduce** and **Parameter Server (PS)** architectures at runtime based on live worker performance.

Modern distributed deep learning systems typically commit to a single communication strategy before training begins. However, real-world environments are dynamic — worker speeds fluctuate due to system noise, contention, and network variability.

AdaptoSGD addresses this limitation by **detecting stragglers in real time** and **adapting the communication strategy accordingly**, improving overall training efficiency.

---

## 🎯 Motivation

Distributed SGD faces a fundamental tradeoff:

| Strategy | Strength | Weakness |
|----------|--------|---------|
| **Ring AllReduce** | Optimal bandwidth, strong consistency | Fails under stragglers (blocking) |
| **Parameter Server** | Handles stragglers (async) | Gradient staleness + server bottleneck |

👉 Existing systems choose **one** — **AdaptoSGD chooses both, dynamically**.

---

## 💡 Key Idea

AdaptoSGD continuously monitors worker performance and computes a **straggler score**:

\[
\sigma = \frac{\max(t_i)}{\text{median}(t_i)}
\]

- If **σ > τ** → switch to **Parameter Server (async mode)**  
- If **σ ≤ τ consistently** → switch back to **Ring AllReduce (sync mode)**  

---

## 🧠 System Architecture

AdaptoSGD consists of three core components:

### 1. Straggler Monitor
- Collects per-worker iteration times  
- Computes straggler score (σ)  

### 2. Adaptive Switcher
- Applies switching policy with hysteresis  
- Controls system mode (**AllReduce ↔ PS**)  

### 3. Communication Backend
Executes gradient aggregation using:
- **Ring AllReduce (synchronous)**  
- **Parameter Server (asynchronous)**  

---

## 🔄 System Workflow

1. Workers compute gradients  
2. Execution time is recorded  
3. Monitor calculates straggler score  
4. Switcher decides optimal strategy  
5. Backend performs aggregation accordingly  

---

## ⚙️ Communication Modes

### 🔵 Ring AllReduce (Synchronous)
- **Strong consistency**  
- **Zero gradient staleness**  
- High performance in homogeneous environments  
- ❌ Sensitive to stragglers  

---

### 🟢 Parameter Server (Asynchronous, SSP)
- **Bounded staleness (SSP model)**  
- No blocking on slow workers  
- Better utilization under heterogeneity  
- ❌ Centralized bottleneck  

---

## 🔁 Adaptive Switching Logic

| Transition | Condition |
|-----------|----------|
| **AllReduce → PS** | σ > τ (immediate) |
| **PS → AllReduce** | σ ≤ τ for δ consecutive iterations |

- Uses **hysteresis** to prevent oscillations  
- Switching occurs **only at iteration boundaries**  

---

## 📊 Experimental Setup

### Systems Compared:
- **Baseline 1:** Ring AllReduce  
- **Baseline 2:** Parameter Server  
- **Proposed:** AdaptoSGD  

### Conditions Tested:
1. **No Stragglers** (homogeneous workers)  
2. **Static Straggler** (one permanently slow worker)  
3. **Dynamic Straggler** (appears & disappears)  

---

## 📈 Performance Metrics

- **Training Throughput** (iterations/sec)  
- **Time to Convergence**  
- **Straggler Overhead**  
- **Gradient Staleness**  
- **Switch Latency**  

---

## 🧪 Expected Results

| Scenario | Best System |
|---------|------------|
| No Stragglers | AllReduce |
| Static Straggler | Parameter Server |
| Dynamic Straggler | ✅ **AdaptoSGD** |

---

## 🧮 Performance Insights

- Stragglers reduce synchronous throughput drastically  
- Probability of stragglers increases with scale  
- Adaptive switching improves efficiency in real-world conditions  

---

## 🛠️ Tech Stack

- **Language:** Python 3  
- **Parallelism:** `multiprocessing`  
- **Computation:** NumPy  
- **Visualization:** Matplotlib  
- **Data Logging:** CSV + Pandas  

---

## 📁 Project Structure
AdaptoSGD/
│
├── workers/
├── monitor.py
├── switcher.py
├── ring_allreduce.py
├── parameter_server.py
├── backend.py
├── experiments/
├── results/
└── README.md


---

## 🔬 Key Features

- ✅ Runtime adaptive strategy switching  
- ✅ Live straggler detection  
- ✅ Bounded staleness (SSP guarantee)  
- ✅ No model modification required  
- ✅ Lightweight monitoring overhead  

---

## ⚠️ Limitations

- No Byzantine fault tolerance  
- Single point of failure in PS mode  
- Simulated environment (multiprocessing)  
- No dynamic worker join/recovery  

---

## 🧩 Research Contribution

AdaptoSGD fills a critical gap in distributed training systems:

> Existing systems either adapt *within* an architecture or combine architectures statically —  
> **AdaptoSGD is the first to adapt *between* architectures at runtime.**

---

## 📚 References

- Dean et al. (2012) — Distributed Deep Learning  
- Li et al. (2014) — Parameter Server  
- Ho et al. (2013) — SSP Model  
- Sergeev & Del Balso (2018) — Horovod  
- Jiang et al. (2020) — BytePS  
- Shi et al. (2020) — Distributed DL Survey  

---

## 👩‍💻 Authors

- **Asawer Ayesha** (470860)  
- **Alisha Siddiqui** (464647)  

**Department of Computing**  
NUST SEECS  

---

## ⭐ Final Note

AdaptoSGD is designed as a **research prototype** to demonstrate that:

> *Distributed training systems should not be static — they should adapt to the system they run on.*
