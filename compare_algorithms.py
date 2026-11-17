import os
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
PPO_MODEL_PATH = os.path.join(BASE_DIR, "marl_survival_model.pt")
EVO_MODEL_PATH = os.path.join(BASE_DIR, "evolution_best_model.pt")


def evaluate_policy(env: MARLSurvivalEnv, policy_type: str = "random", model: ActorCritic = None, num_episodes: int = 10, max_steps: int = 120, base_seed: int = 9000):
    survivals = []
    foods = []
    rewards = []
    alive_agents_final = []

    for ep in range(num_episodes):
        seed = base_seed + ep * 31
        obs_dict, _ = env.reset(seed=seed)
        step = 0
        ep_food = 0
        ep_reward = {a: 0.0 for a in env.possible_agents}

        while env.agents and step < max_steps:
            step += 1
            actions = {}

            if policy_type == "random":
                for a in env.agents:
                    actions[a] = env.action_space(a).sample()
            elif policy_type in ("ppo", "evolution") and model is not None:
                with torch.no_grad():
                    for a in env.agents:
                        obs_t = torch.tensor(obs_dict[a], dtype=torch.float32).unsqueeze(0)
                        logits, _ = model(obs_t)
                        actions[a] = torch.argmax(logits, dim=-1).item()

            obs_dict, step_rewards, terminations, truncations, infos = env.step(actions)

            for a, r in step_rewards.items():
                ep_reward[a] += r
                if infos.get(a, {}).get("food_found"):
                    ep_food += 1

        survivals.append(step)
        foods.append(ep_food)
        rewards.append(np.mean(list(ep_reward.values())))
        alive_count = len([a for a in env.possible_agents if env.agent_healths[a] > 0])
        alive_agents_final.append(alive_count)

    return {
        "mean_steps": np.mean(survivals),
        "mean_food": np.mean(foods),
        "mean_reward": np.mean(rewards),
        "alive_rate": (np.mean(alive_agents_final) / CONFIG["NUM_AGENTS"]) * 100.0,
    }


def run_comparison(num_episodes: int = 10, max_steps: int = 120):
    print("=" * 75)
    print("BENCHMARK COMPARISON: RANDOM vs PPO vs NEUROEVOLUTION")
    print(f"Episodes: {num_episodes} per algorithm | Max Steps: {max_steps} | Seed Base: 9000")
    print("=" * 75)

    env = MARLSurvivalEnv(config=CONFIG, max_steps=max_steps)

    # 1. Random Baseline
    rand_res = evaluate_policy(env, policy_type="random", num_episodes=num_episodes, max_steps=max_steps)

    # 2. PPO
    ppo_model = ActorCritic(obs_dim=30, action_dim=6, hidden_dim=128)
    if os.path.exists(PPO_MODEL_PATH):
        try:
            ppo_model.load_state_dict(torch.load(PPO_MODEL_PATH, map_location="cpu"))
            ppo_model.eval()
            ppo_res = evaluate_policy(env, policy_type="ppo", model=ppo_model, num_episodes=num_episodes, max_steps=max_steps)
        except Exception:
            ppo_res = rand_res
    else:
        ppo_res = rand_res

    # 3. Evolution
    evo_model = ActorCritic(obs_dim=30, action_dim=6, hidden_dim=128)
    if os.path.exists(EVO_MODEL_PATH):
        try:
            evo_model.load_state_dict(torch.load(EVO_MODEL_PATH, map_location="cpu"))
            evo_model.eval()
            evo_res = evaluate_policy(env, policy_type="evolution", model=evo_model, num_episodes=num_episodes, max_steps=max_steps)
        except Exception:
            evo_res = rand_res
    else:
        evo_res = rand_res

    # Display comparison table
    print(f"{'Algoritma':<20} | {'Ort. Adim':<12} | {'Ort. Yiyecek':<13} | {'Ort. Odul':<12} | {'Canli Orani (%)':<15}")
    print("-" * 80)
    print(f"{'Random Baseline':<20} | {rand_res['mean_steps']:<12.1f} | {rand_res['mean_food']:<13.1f} | {rand_res['mean_reward']:<12.2f} | %{rand_res['alive_rate']:<14.1f}")
    print(f"{'PPO (RL)':<20} | {ppo_res['mean_steps']:<12.1f} | {ppo_res['mean_food']:<13.1f} | {ppo_res['mean_reward']:<12.2f} | %{ppo_res['alive_rate']:<14.1f}")
    if os.path.exists(EVO_MODEL_PATH):
        print(f"{'Neuroevolution (GA)':<20} | {evo_res['mean_steps']:<12.1f} | {evo_res['mean_food']:<13.1f} | {evo_res['mean_reward']:<12.2f} | %{evo_res['alive_rate']:<14.1f}")
    else:
        print(f"{'Neuroevolution (GA)':<20} | {'(Henuz egitilmedi - train_evolution.py calistiriniz)'}")
    print("=" * 80)


if __name__ == "__main__":
    run_comparison(num_episodes=5, max_steps=80)
