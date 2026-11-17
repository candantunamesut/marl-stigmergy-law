import os
import json
import argparse
import numpy as np
import torch

try:
    from env import MARLSurvivalEnv, WRITE_TO_MAP
    from model import ActorCritic
    from train import PPOTrainer
    from config import CONFIG
except ImportError:
    from marl_ppo.env import MARLSurvivalEnv, WRITE_TO_MAP
    from marl_ppo.model import ActorCritic
    from marl_ppo.train import PPOTrainer
    from marl_ppo.config import CONFIG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate_trained_model(env: MARLSurvivalEnv, model: ActorCritic, num_episodes: int = 15, max_steps: int = 150, base_seed: int = 7700):
    survivals = []
    foods = []
    map_writes = []
    rewards = []

    device = next(model.parameters()).device
    model.eval()
    for ep in range(num_episodes):
        obs_dict, _ = env.reset(seed=base_seed + ep * 13)
        ep_reward = {a: 0.0 for a in env.possible_agents}
        ep_food = 0
        ep_writes = 0
        step = 0

        while env.agents and step < max_steps:
            step += 1
            actions = {}
            with torch.no_grad():
                for a in env.agents:
                    obs_t = torch.tensor(obs_dict[a], dtype=torch.float32, device=device).unsqueeze(0)
                    logits, _ = model(obs_t)
                    act = torch.argmax(logits, dim=-1).item()
                    actions[a] = act
                    if act == WRITE_TO_MAP:
                        ep_writes += 1

            obs_dict, step_rewards, terminations, truncations, infos = env.step(actions)
            for a, r in step_rewards.items():
                ep_reward[a] += r
                if infos.get(a, {}).get("food_found"):
                    ep_food += 1

        survivals.append(step)
        foods.append(ep_food)
        map_writes.append(ep_writes)
        rewards.append(np.mean(list(ep_reward.values())))

    return {
        "mean_steps": float(np.mean(survivals)),
        "std_steps": float(np.std(survivals)),
        "mean_food": float(np.mean(foods)),
        "std_food": float(np.std(foods)),
        "mean_writes": float(np.mean(map_writes)),
        "std_writes": float(np.std(map_writes)),
        "mean_reward": float(np.mean(rewards)),
    }


def run_experiment(episodes: int = 200, eval_episodes: int = 15, steps_cap: int = 150):
    print("=" * 80)
    print("DUAL VISIBILITY COMPARATIVE EXPERIMENT: FULL-MAP vs. LOCAL STIGMERGY")
    print(f"Training Episodes per condition: {episodes} | Evaluation Episodes: {eval_episodes}")
    print("=" * 80)

    results = {}

    for mode in ["full", "local"]:
        print(f"\n>>> Running Condition: {mode.upper()} VISIBILITY")
        cfg = CONFIG.copy()
        cfg["VISIBILITY_MODE"] = mode

        model_path = os.path.join(BASE_DIR, f"marl_survival_model_{mode}.pt")
        env = MARLSurvivalEnv(config=cfg, max_steps=steps_cap)
        trainer = PPOTrainer(env, lr=5e-4)

        # Train PPO on condition
        trainer.train(
            total_episodes=episodes,
            max_steps_per_episode=steps_cap,
            update_frequency_steps=1024,
            log_interval=max(1, episodes // 4),
            save_path=model_path,
        )

        # Evaluate PPO on condition
        metrics = evaluate_trained_model(env, trainer.model, num_episodes=eval_episodes, max_steps=steps_cap)
        label = "Full-Map Visibility (Upper Bound)" if mode == "full" else "Local Vision + Stigmergy Map"
        results[label] = metrics

    # Print Summary Table
    print("\n" + "=" * 85)
    print("EXPERIMENTAL RESULTS: FULL-MAP VISIBILITY vs. LOCAL STIGMERGY MEMORY")
    print("=" * 85)
    header = f"{'Configuration':<34} | {'Surv. Steps':<14} | {'Food Consumed':<14} | {'Map Writes':<14} | {'Avg Reward':<10}"
    print(header)
    print("-" * 85)
    for label, m in results.items():
        row = (
            f"{label:<34} | "
            f"{m['mean_steps']:5.1f} ± {m['std_steps']:3.1f}   | "
            f"{m['mean_food']:5.1f} ± {m['std_food']:3.1f}   | "
            f"{m['mean_writes']:5.1f} ± {m['std_writes']:3.1f}   | "
            f"{m['mean_reward']:7.1f}"
        )
        print(row)
    print("=" * 85)

    out_json = os.path.join(BASE_DIR, "visibility_comparison.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Comparison report saved to: {out_json}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=150, help="Episodes to train each mode")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Episodes to evaluate")
    parser.add_argument("--steps", type=int, default=150, help="Max steps per episode")
    args = parser.parse_args()

    run_experiment(episodes=args.episodes, eval_episodes=args.eval_episodes, steps_cap=args.steps)
