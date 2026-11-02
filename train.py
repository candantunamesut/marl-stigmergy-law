import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from env import MARLSurvivalEnv, ACTION_NAMES
    from model import ActorCritic
    from config import CONFIG
except ImportError:
    from marl_ppo.env import MARLSurvivalEnv, ACTION_NAMES
    from marl_ppo.model import ActorCritic
    from marl_ppo.config import CONFIG

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class PPOTrainer:
    """
    Multi-Agent PPO Trainer for MARL Survival & Stigmergy Emergence.
    Uses shared actor-critic parameters across homogeneous agents.
    """
    def __init__(
        self,
        env: MARLSurvivalEnv,
        lr: float = 5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        initial_ent_coef: float = 0.08,
        final_ent_coef: float = 0.02,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "auto",
    ):
        self.env = env
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.ent_coef = initial_ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.model = ActorCritic(obs_dim=30, action_dim=6, hidden_dim=128).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

    def train(
        self,
        total_episodes: int = 500,
        max_steps_per_episode: int = 200,
        update_frequency_steps: int = 1024,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        log_interval: int = 50,
        save_path: str = None,
    ):
        if save_path is None:
            save_path = os.path.join(CURRENT_DIR, "marl_survival_model.pt")

        print("=" * 65)
        print("STARTING MARL REINFORCEMENT LEARNING TRAINING (PPO + STIGMERGY)")
        print(f"Device: {self.device} | Total Episodes: {total_episodes} | Step Cap: {max_steps_per_episode}")
        print("=" * 65)

        b_obs = []
        b_actions = []
        b_logprobs = []
        b_rewards = []
        b_dones = []
        b_values = []

        history = []
        start_time = time.time()

        ep_rewards = []
        ep_lengths = []
        ep_foods = []
        ep_map_writes = []

        global_step = 0

        for episode in range(1, total_episodes + 1):
            progress = episode / float(total_episodes)
            self.ent_coef = self.initial_ent_coef + progress * (self.final_ent_coef - self.initial_ent_coef)

            obs_dict, _ = self.env.reset(seed=1000 + episode)
            episode_reward = {a: 0.0 for a in self.env.possible_agents}
            food_found_in_ep = 0
            map_writes_in_ep = 0
            ep_step = 0

            while self.env.agents and ep_step < max_steps_per_episode:
                ep_step += 1
                global_step += 1

                step_obs = {}
                step_actions = {}
                step_logprobs = {}
                step_values = {}

                with torch.no_grad():
                    for agent in self.env.agents:
                        obs_tensor = torch.tensor(
                            obs_dict[agent], dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        action, logprob, _, value = self.model.get_action_and_value(obs_tensor)

                        act_int = action.item()
                        step_obs[agent] = obs_dict[agent]
                        step_actions[agent] = act_int
                        step_logprobs[agent] = logprob.item()
                        step_values[agent] = value.item()

                        if act_int == 5:  # WRITE_TO_MAP
                            map_writes_in_ep += 1

                next_obs_dict, rewards, terminations, truncations, infos = self.env.step(step_actions)

                for agent in step_actions:
                    b_obs.append(step_obs[agent])
                    b_actions.append(step_actions[agent])
                    b_logprobs.append(step_logprobs[agent])
                    scaled_r = rewards.get(agent, 0.0) * 0.1
                    b_rewards.append(scaled_r)
                    is_done = terminations.get(agent, False) or truncations.get(agent, False)
                    b_dones.append(is_done)
                    b_values.append(step_values[agent])

                    episode_reward[agent] += rewards.get(agent, 0.0)
                    if infos.get(agent, {}).get("food_found"):
                        food_found_in_ep += 1

                obs_dict = next_obs_dict

                if len(b_obs) >= update_frequency_steps:
                    self._update_ppo(
                        b_obs, b_actions, b_logprobs, b_rewards, b_dones, b_values,
                        ppo_epochs=ppo_epochs, batch_size=batch_size
                    )
                    b_obs, b_actions, b_logprobs, b_rewards, b_dones, b_values = [], [], [], [], [], []

            mean_ep_reward = np.mean([episode_reward[a] for a in self.env.possible_agents])
            ep_rewards.append(mean_ep_reward)
            ep_lengths.append(ep_step)
            ep_foods.append(food_found_in_ep)
            ep_map_writes.append(map_writes_in_ep)

            if episode % log_interval == 0 or episode == 1:
                avg_r = np.mean(ep_rewards[-log_interval:])
                avg_l = np.mean(ep_lengths[-log_interval:])
                avg_f = np.mean(ep_foods[-log_interval:])
                avg_m = np.mean(ep_map_writes[-log_interval:])
                elapsed = time.time() - start_time
                print(
                    f"Ep {episode:3d}/{total_episodes} | Steps: {global_step:5d} | "
                    f"Surv. Steps: {avg_l:5.1f} | Food/Ep: {avg_f:4.1f} | "
                    f"Map Writes: {avg_m:4.1f} | Avg Reward: {avg_r:7.2f} | Time: {elapsed:.1f}s"
                )

                history.append({
                    "episode": episode,
                    "avg_reward": float(avg_r),
                    "avg_survival_steps": float(avg_l),
                    "avg_food_found": float(avg_f),
                    "avg_map_writes": float(avg_m),
                })

        torch.save(self.model.state_dict(), save_path)
        print(f"\nModel checkpoint saved to: {save_path}")

        return history

    def _update_ppo(
        self, b_obs, b_actions, b_logprobs, b_rewards, b_dones, b_values,
        ppo_epochs: int, batch_size: int
    ):
        obs_t = torch.tensor(np.array(b_obs), dtype=torch.float32, device=self.device)
        act_t = torch.tensor(np.array(b_actions), dtype=torch.int64, device=self.device)
        logp_t = torch.tensor(np.array(b_logprobs), dtype=torch.float32, device=self.device)
        rew_t = np.array(b_rewards, dtype=np.float32)
        done_t = np.array(b_dones, dtype=np.bool_)
        val_t = np.array(b_values, dtype=np.float32)

        # Compute Generalized Advantage Estimation (GAE)
        n = len(b_obs)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae_lam = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - float(done_t[t])
                next_value = 0.0
            else:
                next_non_terminal = 1.0 - float(done_t[t])
                next_value = val_t[t + 1]

            delta = rew_t[t] + self.gamma * next_value * next_non_terminal - val_t[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

        returns = advantages + val_t
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Advantage normalization
        adv_std = adv_t.std()
        if adv_std > 1e-6:
            adv_t = (adv_t - adv_t.mean()) / (adv_std + 1e-8)

        # Optimization epochs
        indices = np.arange(n)
        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                _, new_logp, entropy, new_val = self.model.get_action_and_value(
                    obs_t[batch_idx], act_t[batch_idx]
                )

                log_ratio = new_logp - logp_t[batch_idx]
                ratio = torch.exp(log_ratio)

                # Policy Loss with PPO clipping
                mb_adv = adv_t[batch_idx]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                v_loss = 0.5 * ((new_val.squeeze(-1) - ret_t[batch_idx]) ** 2).mean()

                # Entropy Loss
                entropy_loss = entropy.mean()

                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def evaluate(self, num_episodes: int = 5, max_steps: int = 200, temperature: float = 0.5):
        print("\n" + "=" * 65)
        print(f"EVALUATING TRAINED PPO POLICY OVER {num_episodes} EPISODES (Temp={temperature})")
        print("=" * 65)

        self.model.eval()
        eval_survivals = []
        eval_foods = []
        eval_map_writes = []
        eval_rewards = []

        for ep in range(1, num_episodes + 1):
            obs_dict, _ = self.env.reset(seed=5000 + ep)
            ep_rewards = {a: 0.0 for a in self.env.possible_agents}
            food_count = 0
            map_write_count = 0
            step = 0

            while self.env.agents and step < max_steps:
                step += 1
                actions = {}
                with torch.no_grad():
                    for agent in self.env.agents:
                        obs_tensor = torch.tensor(
                            obs_dict[agent], dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        logits, _ = self.model(obs_tensor)
                        if temperature > 0.0:
                            dist = torch.distributions.Categorical(logits=logits / temperature)
                            action = dist.sample().item()
                        else:
                            action = torch.argmax(logits, dim=-1).item()
                        actions[agent] = action

                        if action == 5:
                            map_write_count += 1

                obs_dict, rewards, terminations, truncations, infos = self.env.step(actions)
                for agent, r in rewards.items():
                    ep_rewards[agent] += r
                    if infos.get(agent, {}).get("food_found"):
                        food_count += 1

            mean_r = np.mean(list(ep_rewards.values()))
            eval_survivals.append(step)
            eval_foods.append(food_count)
            eval_map_writes.append(map_write_count)
            eval_rewards.append(mean_r)

            print(
                f"Eval Ep {ep}: Survived {step:3d} steps | Foods Found: {food_count} | "
                f"Map Writes: {map_write_count} | Avg Reward: {mean_r:.2f}"
            )

        print("-" * 65)
        print("EVALUATION SUMMARY (TRAINED PPO AGENTS):")
        print(f"  Average Survival Steps: {np.mean(eval_survivals):.1f} / {max_steps}")
        print(f"  Average Food Found Per Ep: {np.mean(eval_foods):.1f}")
        print(f"  Average Map Signaling Count: {np.mean(eval_map_writes):.1f}")
        print(f"  Average Cumulative Reward: {np.mean(eval_rewards):.2f}")
        print("=" * 65)

        return {
            "mean_survival_steps": float(np.mean(eval_survivals)),
            "mean_food_found": float(np.mean(eval_foods)),
            "mean_map_writes": float(np.mean(eval_map_writes)),
            "mean_reward": float(np.mean(eval_rewards)),
        }


def main():
    env = MARLSurvivalEnv(config=CONFIG)
    trainer = PPOTrainer(env, lr=5e-4, initial_ent_coef=0.08, final_ent_coef=0.02)

    history = trainer.train(
        total_episodes=500,
        max_steps_per_episode=200,
        update_frequency_steps=1024,
        log_interval=50,
        save_path=os.path.join(CURRENT_DIR, "marl_survival_model.pt"),
    )

    history_path = os.path.join(CURRENT_DIR, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    eval_results = trainer.evaluate(num_episodes=5, max_steps=200, temperature=0.5)

    eval_path = os.path.join(CURRENT_DIR, "eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)


if __name__ == "__main__":
    main()
