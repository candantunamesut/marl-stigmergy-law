import sys
import numpy as np

try:
    from env import MARLSurvivalEnv, ACTION_NAMES, WRITE_TO_MAP, STAY, MOVE_RIGHT
    from config import CONFIG
except ImportError:
    from marl_ppo.env import MARLSurvivalEnv, ACTION_NAMES, WRITE_TO_MAP, STAY, MOVE_RIGHT
    from marl_ppo.config import CONFIG


def test_updated_mechanics():
    print("=" * 65)
    print("TESTING REVISED MARL ENVIRONMENT (STIGMERGY & GRADUAL POOLS)")
    print("=" * 65)

    # 1. Test Observation Shape & Action Space in Local Mode
    cfg_local = CONFIG.copy()
    cfg_local["VISIBILITY_MODE"] = "local"
    env_local = MARLSurvivalEnv(config=cfg_local, max_steps=150)
    obs_dict, _ = env_local.reset(seed=42)

    assert len(env_local.agents) == cfg_local["NUM_AGENTS"]
    for agent in env_local.agents:
        obs = obs_dict[agent]
        assert obs.shape == (30,), f"Expected local obs shape (30,), got {obs.shape}"
        assert env_local.action_space(agent).n == 6, f"Expected 6 actions, got {env_local.action_space(agent).n}"
        assert env_local.agent_healths[agent] == 60.0
    print("[OK] Local mode: obs shape = (30,), action space = Discrete(6) confirmed.")

    # 2. Test Observation Shape in Full Visibility Mode
    cfg_full = CONFIG.copy()
    cfg_full["VISIBILITY_MODE"] = "full"
    env_full = MARLSurvivalEnv(config=cfg_full, max_steps=150)
    obs_full, _ = env_full.reset(seed=42)
    for agent in env_full.agents:
        assert obs_full[agent].shape == (30,), f"Expected full obs shape (30,), got {obs_full[agent].shape}"
    print("[OK] Full visibility mode: obs shape = (30,) confirmed.")

    # 3. Test Per-Timestep Living Cost (Movement vs Stay cost identical)
    # Step with STAY
    env_local.reset(seed=100)
    stay_actions = {a: STAY for a in env_local.agents}
    env_local.step(stay_actions)
    for a in env_local.agents:
        assert abs(env_local.agent_healths[a] - (60.0 - 0.2)) < 1e-4, "Living cost error on STAY"

    # Step with MOVE_RIGHT
    move_actions = {a: MOVE_RIGHT for a in env_local.agents}
    env_local.step(move_actions)
    for a in env_local.agents:
        assert abs(env_local.agent_healths[a] - (60.0 - 0.4)) < 1e-4, "Living cost error on MOVE"
    print("[OK] Per-timestep living cost verified: moving and staying incur identical -0.2 HP decay.")

    # 4. Test Explicit WRITE_TO_MAP Action
    env_local.reset(seed=200)
    pos_0 = env_local.agent_positions["agent_0"].copy()
    assert env_local.shared_map[pos_0[1], pos_0[0]] == 0.0

    mark_actions = {"agent_0": WRITE_TO_MAP, "agent_1": STAY, "agent_2": STAY, "agent_3": STAY}
    _, _, _, _, infos = env_local.step(mark_actions)

    assert env_local.shared_map[pos_0[1], pos_0[0]] > 0.5, "Pheromone was not deposited on map"
    assert infos["agent_0"].get("wrote_to_map") is True, "wrote_to_map info flag missing"
    print("[OK] WRITE_TO_MAP deposits pheromone on shared map successfully.")

    # 5. Test Gradual Resource Pool Feeding & Multi-Agent Sharing
    env_local.reset(seed=300)
    pool = env_local.rewards_data[0]
    initial_amount = pool["amount"]
    pool_pos = pool["pos"].copy()

    # Place agent_0 and agent_1 on the same pool, and lower their health to 30 HP
    env_local.agent_positions["agent_0"] = pool_pos.copy()
    env_local.agent_positions["agent_1"] = pool_pos.copy()
    env_local.agent_healths["agent_0"] = 30.0
    env_local.agent_healths["agent_1"] = 30.0
    # Place agent_2 and agent_3 away
    env_local.agent_positions["agent_2"] = np.array([(pool_pos[0] + 5) % 20, (pool_pos[1] + 5) % 20])
    env_local.agent_positions["agent_3"] = np.array([(pool_pos[0] + 6) % 20, (pool_pos[1] + 6) % 20])

    feed_actions = {a: STAY for a in env_local.agents}
    _, rewards, _, _, _ = env_local.step(feed_actions)

    # Each agent should consume feed_rate (3.0 units), total consumed = 6.0 units
    # Pool replenishes by replenish_rate (0.2 units), so net pool amount = initial - 6.0 + 0.2
    expected_amount = initial_amount - 6.0 + 0.2
    assert abs(pool["amount"] - expected_amount) < 1e-3, f"Expected pool amount {expected_amount}, got {pool['amount']}"
    # Health recovered: 30.0 - 0.2 (decay) + 3.0 * 5.0 (gain) = 44.8 HP
    assert abs(env_local.agent_healths["agent_0"] - 44.8) < 1e-4
    assert abs(env_local.agent_healths["agent_1"] - 44.8) < 1e-4

    # Reward: feeding reward (3.0 * 2.0 = 6.0) + team share from other feeding agent (10% of 6.0 = 0.6) = 6.6
    assert abs(rewards["agent_0"] - 6.6) < 1e-4
    print("[OK] Gradual pool multi-agent feeding & continuous replenishment verified.")

    # 6. Test Team Death Penalty Cascade
    # Move all agents away from all food pools
    env_local.agent_positions["agent_0"] = np.array([0, 0])
    env_local.agent_positions["agent_1"] = np.array([1, 1])
    env_local.agent_positions["agent_2"] = np.array([2, 2])
    env_local.agent_positions["agent_3"] = np.array([3, 3])
    for r in env_local.rewards_data:
        r["pos"] = np.array([15, 15])  # Place food far away

    env_local.agent_healths["agent_0"] = 0.1
    hp_1_before = env_local.agent_healths["agent_1"]
    env_local.step({a: STAY for a in env_local.agents})

    assert "agent_0" not in env_local.agents, "agent_0 should have died"
    expected_hp_1 = hp_1_before - 0.2 - 10.0  # decay + teammate death penalty
    assert abs(env_local.agent_healths["agent_1"] - expected_hp_1) < 1e-4
    print("[OK] Team death penalty (-10 HP) confirmed.")

    print("\nALL MECHANICS AND REVISIONS TESTED AND FULLY VERIFIED!")


if __name__ == "__main__":
    test_updated_mechanics()
