import os
import json
import time
import copy
import numpy as np
import torch

try:
    from env import MARLSurvivalEnv
    from model import ActorCritic
    from config import CONFIG
except ImportError:
    from marl_survival_experiment.env import MARLSurvivalEnv
    from marl_survival_experiment.model import ActorCritic
    from marl_survival_experiment.config import CONFIG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVO_MODEL_PATH = os.path.join(BASE_DIR, "evolution_best_model.pt")
EVO_HISTORY_PATH = os.path.join(BASE_DIR, "evolution_history.json")


def evaluate_individual(env: MARLSurvivalEnv, model: ActorCritic, weights: np.ndarray, num_episodes: int = 3, max_steps: int = 120, base_seed: int = 7000) -> float:
    """Evaluates an individual parameter vector over deterministic seeds and returns fitness."""
    model.set_flat_weights(weights)
    model.eval()

    total_fitness = 0.0

    with torch.no_grad():
        for ep in range(num_episodes):
            obs_dict, _ = env.reset(seed=base_seed + ep * 13)
            ep_reward = 0.0
            step = 0

            while env.agents and step < max_steps:
                step += 1
                actions = {}
                for agent in env.agents:
                    obs_t = torch.tensor(obs_dict[agent], dtype=torch.float32).unsqueeze(0)
                    logits, _ = model(obs_t)
                    actions[agent] = torch.argmax(logits, dim=-1).item()

                obs_dict, rewards, terminations, truncations, infos = env.step(actions)
                ep_reward += sum(rewards.values())

            # Fitness balances survival steps and collective team rewards
            fitness = step * 2.0 + ep_reward
            total_fitness += fitness

    return float(total_fitness / num_episodes)


def train_evolution(
    population_size: int = 20,
    generations: int = 25,
    top_k: int = 4,
    mutation_power: float = 0.05,
    max_steps_per_episode: int = 100,
    save_path: str = EVO_MODEL_PATH,
):
    print("=" * 65)
    print("STARTING NEUROEVOLUTION (GENETIC ALGORITHM) FOR MARL SURVIVAL")
    print(f"Population: {population_size} | Generations: {generations} | Elites (Top-K): {top_k}")
    print("=" * 65)

    env = MARLSurvivalEnv(config=CONFIG, max_steps=max_steps_per_episode)
    template_model = ActorCritic(obs_dim=30, action_dim=6, hidden_dim=128)
    base_weights = template_model.get_flat_weights()
    num_params = len(base_weights)

    # Initialize population with random perturbations around base weights
    population = [
        base_weights + np.random.randn(num_params).astype(np.float32) * 0.1
        for _ in range(population_size)
    ]

    best_overall_fitness = -1e9
    best_overall_weights = None
    history = []

    start_time = time.time()

    for gen in range(1, generations + 1):
        fitness_scores = []
        for ind_idx, ind in enumerate(population):
            score = evaluate_individual(env, template_model, ind, num_episodes=2, max_steps=max_steps_per_episode, base_seed=gen * 100)
            fitness_scores.append(score)

        fitness_scores = np.array(fitness_scores)
        sorted_indices = np.argsort(fitness_scores)[::-1]

        best_gen_fitness = fitness_scores[sorted_indices[0]]
        avg_gen_fitness = np.mean(fitness_scores)

        if best_gen_fitness > best_overall_fitness:
            best_overall_fitness = best_gen_fitness
            best_overall_weights = copy.deepcopy(population[sorted_indices[0]])

        elapsed = time.time() - start_time
        print(
            f"Gen {gen:2d}/{generations} | Best Fitness: {best_gen_fitness:7.1f} | "
            f"Avg Fitness: {avg_gen_fitness:7.1f} | Overall Best: {best_overall_fitness:7.1f} | Elapsed: {elapsed:.1f}s"
        )

        history.append({
            "generation": gen,
            "best_fitness": float(best_gen_fitness),
            "avg_fitness": float(avg_gen_fitness),
        })

        # Selection: top_k elites
        elites = [population[idx] for idx in sorted_indices[:top_k]]

        # Next generation: Elites + mutated offspring
        new_population = [copy.deepcopy(elite) for elite in elites]

        while len(new_population) < population_size:
            # Pick a random elite parent
            parent = elites[np.random.randint(0, top_k)]
            # Mutate with Gaussian noise
            noise = np.random.randn(num_params).astype(np.float32) * mutation_power
            child = parent + noise
            new_population.append(child)

        population = new_population

    # Save best individual weights
    template_model.set_flat_weights(best_overall_weights)
    torch.save(template_model.state_dict(), save_path)
    print(f"\n[OK] Best evolutionary policy saved to: {save_path}")

    with open(EVO_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return best_overall_fitness


if __name__ == "__main__":
    train_evolution(population_size=15, generations=10, top_k=3, max_steps_per_episode=80)
