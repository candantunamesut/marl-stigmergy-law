# Multi-Agent Stigmergy & Institutional Memory Simulation
**Project:** Externalized Memory and Institutional Rule Systems in Multi-Agent Societies  
**Authors:** Candan Tuna Mesut & Kayra Sevim  
**Supervisor:** Dr. Zeki Doruk Erden  

---

## 📌 Executive Summary
This repository contains a computational Multi-Agent Reinforcement Learning (MARL) and Evolutionary Simulation framework investigating how artificial agents develop, utilize, and adapt to **externalized shared memory (stigmergic pheromone maps)** in resource-scarce environments.

The project models the **Tragedy of the Commons** dilemma where agents harvest scarce, slowly replenishing food hotspots while managing their metabolic decay.

---

## 🔬 Core Mechanics & Supervisor Revisions
Following meeting feedback with Dr. Zeki Doruk Erden, the environment was upgraded with:
1. **Zero-Cost Stigmergic Marking (`MAP_ACTION_COST = 0.0` & `WRITE_HEALTH_DECAY = 0.0`):** Marking the shared map does not penalize rewards or drain agent health, allowing communication to emerge freely.
2. **Calibrated Metabolic Decay (`HEALTH_DECAY = 0.2 HP/step`):** Allows agents up to ~150 exploration steps without food, solving the sparse-reward starvation bottleneck.
3. **$3 \times 3$ Spatial Pheromone Sensing:** Agents observe the $3 \times 3$ local neighborhood of the shared pheromone matrix (`obs[21:30]`), enabling them to detect and follow spatial pheromone gradients.
4. **Passive Map Perception:** Reading the shared map is part of the 30-dimensional observation vector, eliminating wasted turns.
5. **Dual Paradigm Comparison:** Implements both **PPO (Proximal Policy Optimization)** and **Neuroevolution (Genetic Algorithm)**.

---

## 📁 File Structure
- `config.py`: Environment configurations, reward weights, and metabolic rates.
- `env.py`: PettingZoo-compatible $20 \times 20$ grid world with dynamic pheromone dissipation.
- `model.py`: PyTorch Actor-Critic architecture with flat-weight interfaces for evolutionary optimization.
- `train.py`: Deep Reinforcement Learning pipeline using Multi-Agent PPO.
- `train_evolution.py`: Population-based Genetic Algorithm optimizing episodic cumulative fitness.
- `compare_algorithms.py`: Rigorous evaluation benchmark comparing Random Baseline, PPO, and Neuroevolution.
- `visualize.py`: Step-by-step terminal visualizer displaying agents, food, and pheromone intensities.

---

## 🚀 How to Run and Evaluate

### 1. Run PPO Training
```bash
python3 train.py
```
*Trains homogeneous PPO agents, logging survival steps, food consumption, map writes, and rewards.*

### 2. Run Neuroevolution (Genetic Algorithm)
```bash
python3 train_evolution.py
```
*Optimizes policy weights directly on total episodic lifespan and food collection across generations.*

### 3. Run Side-by-Side Algorithm Comparison
```bash
python3 compare_algorithms.py
```
*Evaluates trained checkpoints on identical test seeds and prints a comparative statistical table.*

### 4. Step-by-Step Visual Simulation
```bash
python3 visualize.py --steps 80
```
*Displays the $20 \times 20$ grid live, showing agent movements (`0-3`), food hotspots (`F`), and pheromone strength (`*`, `~`).*
