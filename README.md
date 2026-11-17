# Multi-Agent Stigmergy & Institutional Memory Simulation
**Research Project:** Externalized Memory and Institutional Rule Systems in Multi-Agent Societies  
**Authors:** Candan Tuna Mesut & Kayra Sevim  
**Supervisor:** Dr. Zeki Doruk Erden  

---

## 📌 Executive Summary
This repository contains a computational Multi-Agent Reinforcement Learning (MARL) framework exploring how artificial populations develop, utilize, and adapt to **externalized spatial memory (stigmergic pheromone maps)** in resource-scarce commons dilemmas.

Agents operate in a $20 \times 20$ grid world where they must balance individual metabolic survival against collective resource depletion. 

---

## 🔬 Latest Architectural Milestones & Revisions
Following meeting feedback and diagnostics, the environment was upgraded with four core structural revisions:

1. **Explicit Memory Writing Action (`WRITE_TO_MAP`):**
   - Action space expanded to 6 discrete choices: `MOVE_UP (0)`, `MOVE_DOWN (1)`, `MOVE_LEFT (2)`, `MOVE_RIGHT (3)`, `STAY (4)`, and `WRITE_TO_MAP (5)`.
   - Marking is **not hardcoded**; agents independently learn when to deposit pheromone signals on the shared memory matrix via policy gradient optimization.
2. **Gradual, Depletable & Replenishing Resource Pools:**
   - Replaced single-bite discrete food items with persistent resource pools (`capacity = 30.0` units).
   - Feeding occurs gradually (`feed_rate = 3.0` units/step), allowing multiple agents to co-feed concurrently.
   - Pools continuously regenerate (`replenish_rate = 0.2` units/step), creating dynamic commons dilemma and sharing incentives.
3. **Per-Timestep Living Cost (Eliminated Freezing / Camping):**
   - Replaced per-movement penalties with an intrinsic metabolic existence cost (`0.2 HP/step`).
   - Movement incurs zero extra penalty relative to staying still, eliminating artificial incentives to camp in place.
4. **Dual Visibility Experimental Framework:**
   - Tested under two controlled conditions: **Full-Map Visibility** (global vectors) vs. **Local Vision + Stigmergy** ($3 \times 3$ neighborhood + pheromone tracking).

---

## 📊 Latest Experimental Results (150 Episodes PPO)

Comparative evaluation over 10 test episodes with deterministic evaluation seeds:

| Experimental Configuration | Survival Lifespan | Food Consumed | Map Writes / Ep | Cumulative Reward | Survival Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full-Map Visibility (Global Vectors)** | 150.0 ± 0.0 steps | 0.7 ± 0.9 units | 129.9 ± 88.3 | 1.4 | **100%** |
| **Local Vision + Stigmergy Memory** | **150.0 ± 0.0 steps** | **46.7 ± 42.8 units** | **21.8 ± 25.2** | **33.9** | **100%** |

### 📈 Key Scientific Findings:
- **Massive Foraging Efficiency via Stigmergy:** Agents relying on Local Vision paired with the Stigmergic Shared Map collected **>60x more food** (46.7 vs. 0.7 units) and achieved a significantly higher average reward (+33.9 vs. +1.4).
- **Emergent Strategic Marking:** Over 150 training episodes, random map writing spamming decreased from ~108 writes/ep down to 47.3 writes/ep, while food collection grew steadily from 5.0 to 26.4 units/ep, showing deliberate rather than noise-driven marking behavior.
- **Domino Interdependence:** Mutual health penalties upon teammate death created an obligate mutual survival norm, yielding a 100% survival rate across the full 150-step lifespan.

---

## 🏗️ Environment Specifications

### Action Space (`Discrete(6)`)
| ID | Action | Description |
|---|---|---|
| `0` | `MOVE_UP` | Move one cell north |
| `1` | `MOVE_DOWN` | Move one cell south |
| `2` | `MOVE_LEFT` | Move one cell west |
| `3` | `MOVE_RIGHT` | Move one cell east |
| `4` | `STAY` | Stand still (feed if on resource pool) |
| `5` | `WRITE_TO_MAP` | Deposit pheromone mark on current grid cell |

### Observation Space (`Box(-1.0, 1.0, shape=(30,))`)
Both `local` and `full` modes provide a uniform 30-dimensional observation vector:
- `obs[0:2]`: Normalized agent coordinates $(x, y)$
- `obs[2]`: Health ratio ($\text{HP} / 60.0$)
- `obs[3:12]`: Neighboring agent presence (Local $3 \times 3$) OR Global teammate relative vectors (Full)
- `obs[12:21]`: Neighboring food pool levels (Local $3 \times 3$) OR Global pool relative vectors (Full)
- `obs[21:30]`: $3 \times 3$ local shared map pheromone intensity values

---

## 🚀 How to Run and Reproduce

### 1. Run Unit Tests
Verify that all mechanics, pools, and actions operate properly:
```bash
python3 test_env.py
```

### 2. Train PPO Agents (Local Stigmergy Mode)
```bash
python3 train.py --visibility local --episodes 300
```
*Trained model weights are automatically saved to `marl_survival_model_local.pt`.*

### 3. Run the Dual Visibility Benchmark
Train and compare Full-Map Visibility vs. Local Stigmergy under identical conditions:
```bash
python3 compare_visibility.py --episodes 150
```

### 4. Interactive Simulation Playback (Terminal & Web)
Run simulation playback and export the browser viewer:
```bash
python3 visualize.py --visibility local --steps 80
open simulation_playback.html
```

---

## 📁 Repository Structure
```
marl_ppo/
├── config.py                 # Hyperparameters (pool capacity, decay rates, rewards)
├── env.py                    # PettingZoo ParallelEnv with gradual pools & stigmergy
├── model.py                  # PyTorch Actor-Critic neural network (obs: 30, actions: 6)
├── train.py                  # PPO training pipeline with CLI visibility flag
├── compare_visibility.py     # Dual visibility experiment runner (Full vs. Local)
├── compare_algorithms.py     # Algorithm comparison (Random vs. PPO vs. Evolution)
├── train_evolution.py        # Neuroevolution / Genetic Algorithm baseline
├── test_env.py               # Unit tests verifying environment mechanics
├── visualize.py              # CLI renderer & interactive HTML playback exporter
├── simulation_playback.html  # Standalone interactive browser playback player
└── README.md                 # Project documentation & experimental report
```
