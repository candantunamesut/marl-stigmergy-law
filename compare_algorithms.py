import os
import json
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


def evaluate_policy(env, model, num_episodes: int = 15, max_steps: int = 200, seed_base: int = 8000, is_random: bool = False):
    survivals = []
    foods = []
    writes = []
    rewards = []

    for ep in range(num_episodes):
        obs_dict, _ = env.reset(seed=seed_base + ep)
        ep_rewards = {a: 0.0 for a in env.possible_agents}
        food_count = 0
        map_write_count = 0
        step = 0

        while env.agents and step < max_steps:
            step += 1
            actions = {}
            if is_random:
                for agent in env.agents:
                    actions[agent] = np.random.randint(6)
            else:
                with torch.no_grad():
                    for agent in env.agents:
                        obs_tensor = torch.tensor(
                            obs_dict[agent], dtype=torch.float32
                        ).unsqueeze(0)
                        logits, _ = model(obs_tensor)
                        action = torch.argmax(logits, dim=-1).item()
                        actions[agent] = action

            for a, act in actions.items():
                if act == 5:
                    map_write_count += 1

            obs_dict, step_rewards, terminations, truncations, infos = env.step(actions)
            for agent, r in step_rewards.items():
                ep_rewards[agent] += r
                if infos.get(agent, {}).get("food_found"):
                    food_count += 1

        survivals.append(step)
        foods.append(food_count)
        writes.append(map_write_count)
        rewards.append(np.mean(list(ep_rewards.values())))

    return {
        "mean_survival_steps": float(np.mean(survivals)),
        "std_survival_steps": float(np.std(survivals)),
        "mean_food_found": float(np.mean(foods)),
        "std_food_found": float(np.std(foods)),
        "mean_map_writes": float(np.mean(writes)),
        "mean_reward": float(np.mean(rewards)),
    }


def run_benchmark():
    env = MARLSurvivalEnv(config=CONFIG)

    results = {}

    # 1. Random Baseline
    print("Evaluating Random Baseline...")
    results["Random Baseline"] = evaluate_policy(env, None, num_episodes=15, is_random=True)

    # 2. PPO Model
    ppo_path = os.path.join(CURRENT_DIR, "marl_survival_model.pt")
    if os.path.exists(ppo_path):
        print("Evaluating Trained PPO Model...")
        ppo_model = ActorCritic(obs_dim=30, action_dim=6, hidden_dim=128)
        ppo_model.load_state_dict(torch.load(ppo_path, map_location="cpu"))
        ppo_model.eval()
        results["PPO (Actor-Critic)"] = evaluate_policy(env, ppo_model, num_episodes=15)
    else:
        print("PPO checkpoint not found yet.")

    # 3. Evolution Model
    evo_path = os.path.join(CURRENT_DIR, "evolution_best_model.pt")
    if os.path.exists(evo_path):
        print("Evaluating Trained Neuroevolution Model...")
        evo_model = ActorCritic(obs_dim=30, action_dim=6, hidden_dim=128)
        evo_model.load_state_dict(torch.load(evo_path, map_location="cpu"))
        evo_model.eval()
        results["Evolution (Genetic Alg.)"] = evaluate_policy(env, evo_model, num_episodes=15)
    else:
        print("Evolution checkpoint not found yet.")

    # Print Comparison Table
    print("\n" + "=" * 80)
    print("ALGORITHM BENCHMARK COMPARISON (15 Evaluation Episodes)")
    print("=" * 80)
    headers = f"{'Algorithm / Method':<28} | {'Surv. Steps':<14} | {'Food Found':<12} | {'Map Writes':<12} | {'Mean Reward':<12}"
    print(headers)
    print("-" * 80)
    for name, m in results.items():
        row = (
            f"{name:<28} | "
            f"{m['mean_survival_steps']:6.1f} ± {m['std_survival_steps']:4.1f} | "
            f"{m['mean_food_found']:5.1f} ± {m['std_food_found']:3.1f} | "
            f"{m['mean_map_writes']:6.1f}       | "
            f"{m['mean_reward']:8.1f}"
        )
        print(row)
    print("=" * 80)

    out_file = os.path.join(CURRENT_DIR, "algorithm_comparison.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Comparison results saved to: {out_file}\n")


if __name__ == "__main__":
    run_benchmark()
