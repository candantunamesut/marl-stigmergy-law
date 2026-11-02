import os
import json
import time
import copy
import numpy as np
import torch

try:
    from env import MARLSurvivalEnv, ACTION_NAMES
    from model import ActorCritic
    from config import CONFIG
except ImportError:
    from marl_ppo.env import MARLSurvivalEnv, ACTION_NAMES
    from marl_ppo.model import ActorCritic
    from marl_ppo.config import CONFIG

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class NeuroevolutionTrainer:
    """
    Genetic / Neuroevolution Algorithm for Multi-Agent Survival and Stigmergy.
    Directly optimizes cumulative episodic fitness (survival steps + food found + cooperation),
    providing a non-gradient global optimization alternative to PPO.
    """
    def __init__(
        self,
        env: MARLSurvivalEnv,
        pop_size: int = 20,
        elite_count: int = 2,
        mutation_rate: float = 0.1,
        mutation_power: float = 0.08,
        device: str = "cpu",
    ):
        self.env = env
        self.pop_size = pop_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.mutation_power = mutation_power
        self.device = torch.device(device)

        # Template network
        self.template_model = ActorCritic(obs_dim=30, action_dim=6, hidden_dim=128).to(self.device)
        self.param_dim = len(self.template_model.get_flat_weights())

        # Initialize population with random weights
        self.population = [
            self.template_model.get_flat_weights() + np.random.randn(self.param_dim) * 0.1
            for _ in range(self.pop_size)
        ]

    def evaluate_individual(self, weights: np.ndarray, num_episodes: int = 2, max_steps: int = 200, seed_base: int = 2000):
        """
        Evaluate an individual policy across multiple episodes.
        Returns total cumulative reward, survival steps, food found, and map writes.
        """
        self.template_model.set_flat_weights(weights)
        self.template_model.eval()

        ep_rewards = []
        ep_lengths = []
        ep_foods = []
        ep_map_writes = []

        for ep in range(num_episodes):
            obs_dict, _ = self.env.reset(seed=seed_base + ep)
            episode_reward = {a: 0.0 for a in self.env.possible_agents}
            food_found_in_ep = 0
            map_writes_in_ep = 0
            step = 0

            while self.env.agents and step < max_steps:
                step += 1
                actions = {}
                with torch.no_grad():
                    for agent in self.env.agents:
                        obs_tensor = torch.tensor(
                            obs_dict[agent], dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        logits, _ = self.template_model(obs_tensor)
                        action = torch.argmax(logits, dim=-1).item()
                        actions[agent] = action

                        if action == 5:  # WRITE_TO_MAP
                            map_writes_in_ep += 1

                obs_dict, rewards, terminations, truncations, infos = self.env.step(actions)
                for agent, r in rewards.items():
                    episode_reward[agent] += r
                    if infos.get(agent, {}).get("food_found"):
                        food_found_in_ep += 1

            mean_r = np.mean([episode_reward[a] for a in self.env.possible_agents])
            ep_rewards.append(mean_r)
            ep_lengths.append(step)
            ep_foods.append(food_found_in_ep)
            ep_map_writes.append(map_writes_in_ep)

        avg_reward = float(np.mean(ep_rewards))
        avg_steps = float(np.mean(ep_lengths))
        avg_foods = float(np.mean(ep_foods))
        avg_writes = float(np.mean(ep_map_writes))

        # Fitness: primarily total episodic survival & food reward
        fitness = avg_reward + (avg_foods * 20.0) + (avg_steps * 0.1)

        return {
            "fitness": fitness,
            "reward": avg_reward,
            "steps": avg_steps,
            "food": avg_foods,
            "writes": avg_writes,
        }

    def train(
        self,
        generations: int = 40,
        eval_episodes_per_ind: int = 2,
        max_steps_per_episode: int = 200,
        save_path: str = None,
    ):
        if save_path is None:
            save_path = os.path.join(CURRENT_DIR, "evolution_best_model.pt")

        print("=" * 65)
        print("STARTING NEUROEVOLUTION TRAINING (GENETIC ALGORITHM + STIGMERGY)")
        print(f"Population: {self.pop_size} | Generations: {generations} | Elites: {self.elite_count}")
        print("=" * 65)

        history = []
        best_overall_fitness = -float("inf")
        best_overall_weights = None
        start_time = time.time()

        for gen in range(1, generations + 1):
            gen_results = []
            seed_base = 3000 + gen * 10

            for i, ind in enumerate(self.population):
                res = self.evaluate_individual(
                    ind, num_episodes=eval_episodes_per_ind,
                    max_steps=max_steps_per_episode, seed_base=seed_base
                )
                gen_results.append(res)

            fitnesses = [r["fitness"] for r in gen_results]
            sorted_indices = np.argsort(fitnesses)[::-1]  # descending order

            best_idx = sorted_indices[0]
            best_res = gen_results[best_idx]
            avg_fitness = float(np.mean(fitnesses))
            avg_steps = float(np.mean([r["steps"] for r in gen_results]))
            avg_food = float(np.mean([r["food"] for r in gen_results]))
            avg_writes = float(np.mean([r["writes"] for r in gen_results]))
            avg_reward = float(np.mean([r["reward"] for r in gen_results]))

            if best_res["fitness"] > best_overall_fitness:
                best_overall_fitness = best_res["fitness"]
                best_overall_weights = copy.deepcopy(self.population[best_idx])

            elapsed = time.time() - start_time
            print(
                f"Gen {gen:2d}/{generations} | Best Fit: {best_res['fitness']:7.1f} | "
                f"Avg Fit: {avg_fitness:7.1f} | Surv. Steps: {avg_steps:5.1f} | "
                f"Food: {avg_food:4.1f} | Writes: {avg_writes:4.1f} | Time: {elapsed:.1f}s"
            )

            history.append({
                "generation": gen,
                "best_fitness": best_res["fitness"],
                "avg_fitness": avg_fitness,
                "avg_survival_steps": avg_steps,
                "avg_food_found": avg_food,
                "avg_map_writes": avg_writes,
                "avg_reward": avg_reward,
            })

            # Next Generation Breeding
            new_pop = []
            # 1. Elitism: Retain top performers
            for e in range(self.elite_count):
                new_pop.append(copy.deepcopy(self.population[sorted_indices[e]]))

            # 2. Tournament Selection + Crossover + Mutation
            while len(new_pop) < self.pop_size:
                # Tournament selection (size 3)
                p1_cands = np.random.choice(sorted_indices[: self.pop_size // 2], size=2, replace=False)
                p2_cands = np.random.choice(sorted_indices[: self.pop_size // 2], size=2, replace=False)
                parent1 = self.population[p1_cands[0]]
                parent2 = self.population[p2_cands[0]]

                # Uniform crossover
                mask = np.random.rand(self.param_dim) < 0.5
                child = np.where(mask, parent1, parent2)

                # Mutation
                mutation_mask = np.random.rand(self.param_dim) < self.mutation_rate
                noise = np.random.randn(self.param_dim) * self.mutation_power
                child += mutation_mask * noise

                new_pop.append(child)

            self.population = new_pop

        # Save best model
        self.template_model.set_flat_weights(best_overall_weights)
        torch.save(self.template_model.state_dict(), save_path)
        print(f"\nBest Evolutionary policy saved to: {save_path}")

        return history


def main():
    env = MARLSurvivalEnv(config=CONFIG)
    trainer = NeuroevolutionTrainer(env, pop_size=20, elite_count=2)

    history = trainer.train(
        generations=30,
        eval_episodes_per_ind=2,
        max_steps_per_episode=200,
        save_path=os.path.join(CURRENT_DIR, "evolution_best_model.pt"),
    )

    history_path = os.path.join(CURRENT_DIR, "evolution_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
