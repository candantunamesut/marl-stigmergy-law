import sys
import numpy as np

try:
    from env import MARLSurvivalEnv, ACTION_NAMES
    from config import CONFIG
except ImportError:
    from marl_survival_experiment.env import MARLSurvivalEnv, ACTION_NAMES
    from marl_survival_experiment.config import CONFIG


def test_mechanics():
    print("=" * 60)
    print("TESTING UPDATED MARL SURVIVAL ENVIRONMENT RULES")
    print("=" * 60)

    env = MARLSurvivalEnv(config=CONFIG, max_steps=150)
    obs_dict, infos = env.reset(seed=123)

    # 1. Check Initial Health and Observation Shapes
    print(f"[Check 1] Initial agents: {env.agents}")
    assert len(env.agents) == CONFIG["NUM_AGENTS"]
    for agent in env.agents:
        obs = obs_dict[agent]
        assert obs.shape == (21,), f"Expected shape (21,), got {obs.shape}"
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0), "Obs values out of [0, 1] range"
        assert env.agent_healths[agent] == 60.0, f"Expected 60.0 HP, got {env.agent_healths[agent]}"
        assert env.action_space(agent).n == 5, f"Expected 5 actions, got {env.action_space(agent).n}"
    print("[OK] Initial health = 60.0 HP, obs shape = (21,), action space = Discrete(5) confirmed.")

    # 2. Check Metabolic Health Decay (0.2 HP per step)
    # Take STAY action for all agents
    stay_actions = {a: 4 for a in env.agents}
    next_obs, rewards, terminations, truncations, infos = env.step(stay_actions)
    for agent in env.agents:
        expected_hp = 60.0 - 0.2
        assert abs(env.agent_healths[agent] - expected_hp) < 1e-4, f"Health decay error: {env.agent_healths[agent]}"
    print("[OK] Step metabolic decay (-0.2 HP/step) confirmed.")

    # 3. Check Food Consumption (+60 HP, +60 Reward for Finder, +10% / +6.0 for Teammates)
    # Manually place agent_0 on food 0 and lower agent_0 health to 20
    env.agent_healths["agent_0"] = 20.0
    food_pos = env.rewards_data[0]["pos"].copy()
    env.rewards_data[0]["active"] = True
    env.agent_positions["agent_0"] = food_pos.copy()
    # Move other agents away
    env.agent_positions["agent_1"] = np.array([(food_pos[0] + 5) % 20, (food_pos[1] + 5) % 20])

    step_actions = {a: 4 for a in env.agents}  # STAY
    next_obs, rewards, terminations, truncations, infos = env.step(step_actions)

    assert env.agent_healths["agent_0"] == 60.0, f"Expected 60.0 HP after food, got {env.agent_healths['agent_0']}"
    # Rewards check
    finder_reward = rewards["agent_0"]
    expected_finder = 60.0 + CONFIG["REWARD_VALUES"]["STEP_PENALTY"]
    assert abs(finder_reward - expected_finder) < 1e-4, f"Expected finder reward {expected_finder}, got {finder_reward}"

    teammate_reward = rewards["agent_1"]
    expected_teammate = 6.0 + CONFIG["REWARD_VALUES"]["STEP_PENALTY"]
    assert abs(teammate_reward - expected_teammate) < 1e-4, f"Expected teammate shared reward {expected_teammate}, got {teammate_reward}"
    print("[OK] Food reward (+60 HP & +60 Reward) and Teammate 10% share (+6.0 Reward) confirmed.")

    # 4. Check Teammate Death Penalty (10 HP lost when an agent dies)
    # Set agent_0 health to 0.1, so on next step it decays and dies
    hp_before = {a: env.agent_healths[a] for a in env.agents if a != "agent_0"}
    env.agent_healths["agent_0"] = 0.1

    step_actions = {a: 4 for a in env.agents}
    next_obs, rewards, terminations, truncations, infos = env.step(step_actions)

    assert "agent_0" not in env.agents, "agent_0 should be terminated and removed."
    for a, before_hp in hp_before.items():
        # lost: 0.2 (step decay) + 10.0 (death penalty) = 10.2
        expected_remaining = max(0.0, before_hp - 0.2 - 10.0)
        assert abs(env.agent_healths[a] - expected_remaining) < 1e-4, (
            f"{a} expected HP {expected_remaining}, got {env.agent_healths[a]}"
        )
    print("[OK] Teammate death penalty (-10 HP to all alive agents) confirmed.")

    print("\nALL ENVIRONMENT UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_mechanics()
