import os
import argparse
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


def visualize_episode(
    model_path: str = None,
    max_steps: int = 100,
    seed: int = 42,
    mode: str = "ansi"
):
    print("=" * 65)
    print("RUNNING VISUAL SIMULATION OF TRAINED MARL SURVIVAL AGENTS")
    print("=" * 65)

    env = MARLSurvivalEnv(config=CONFIG, max_steps=max_steps)
    model = ActorCritic(obs_dim=30, action_dim=6, hidden_dim=128)

    if model_path is None:
        ppo_path = os.path.join(CURRENT_DIR, "marl_survival_model.pt")
        evo_path = os.path.join(CURRENT_DIR, "evolution_best_model.pt")
        if os.path.exists(ppo_path):
            model_path = ppo_path
        elif os.path.exists(evo_path):
            model_path = evo_path

    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"Loaded trained policy weights from {model_path}")
            model.eval()
        except Exception as e:
            print(f"Could not load weights ({e}). Running with random initialization.")
    else:
        print("No checkpoint found. Running with untrained baseline.")

    obs_dict, _ = env.reset(seed=seed)

    for step in range(1, max_steps + 1):
        if not env.agents:
            print(f"\nAll agents perished at step {step - 1}.")
            break

        actions = {}
        with torch.no_grad():
            for agent in env.agents:
                obs_t = torch.tensor(obs_dict[agent], dtype=torch.float32).unsqueeze(0)
                logits, _ = model(obs_t)
                action = torch.argmax(logits, dim=-1).item()
                actions[agent] = action

        # Print current grid
        print(f"\n--- STEP {step} ---")
        grid = [["." for _ in range(env.grid_size)] for _ in range(env.grid_size)]

        # Place pheromone marks (~ for faint, * for strong)
        for y in range(env.grid_size):
            for x in range(env.grid_size):
                val = env.shared_map[y, x]
                if val > 0.5:
                    grid[y][x] = "*"
                elif val > 0.1:
                    grid[y][x] = "~"

        # Place foods
        for r in env.rewards_data:
            if r["active"]:
                rx, ry = r["pos"]
                grid[ry][rx] = "F"

        # Place agents
        for agent in env.agents:
            ax, ay = env.agent_positions[agent]
            agent_id = agent.split("_")[-1]
            grid[ay][ax] = agent_id

        for row in grid:
            print(" ".join(row))

        # Status summary
        hp_str = ", ".join([f"{a}: {env.agent_healths[a]:.1f}HP" for a in env.agents])
        act_str = ", ".join([f"{a}: {ACTION_NAMES.get(actions[a], str(actions[a]))}" for a in env.agents])
        print(f"Healths: [{hp_str}]")
        print(f"Actions: [{act_str}]")
        print(f"Shared Map (Max signal: {np.max(env.shared_map):.2f}, Marked cells: {np.count_nonzero(env.shared_map > 0.05)})")

        obs_dict, rewards, terminations, truncations, infos = env.step(actions)

        for agent, r in rewards.items():
            if infos.get(agent, {}).get("food_found"):
                print(f"  >>> {agent} CONSUMED FOOD (+50.0)!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to .pt model")
    parser.add_argument("--steps", type=int, default=80, help="Max steps to render")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    visualize_episode(model_path=args.model, max_steps=args.steps, seed=args.seed)
