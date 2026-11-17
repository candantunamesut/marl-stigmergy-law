# Configuration for Custom MARL Environment & Stigmergy Exploration
# Updated with Supervisor Decisions: Gradual Depletable Rewards, Explicit Map Writing,
# Per-Timestep Living Cost, and Dual Visibility Configurations (Local vs. Full).

CONFIG = {
    # Environment Grid & Entity Counts
    "GRID_SIZE": 20,
    "NUM_AGENTS": 4,
    "NUM_REWARDS": 3,
    "MAX_HEALTH": 60.0,

    # 1. Per-Timestep Living Cost (Replaced Movement Penalty)
    # Every step alive incurs this metabolic cost regardless of moving, staying, or marking.
    "HEALTH_DECAY_PER_STEP": 0.2,

    # Team Interdependence Mechanics
    "TEAM_DEATH_HEALTH_PENALTY": 10.0,  # Health lost by teammates when an agent dies
    "SHARED_REWARD_RATIO": 0.10,        # Teammates receive 10% of food consumed

    # 2. Gradual & Replenishable Resource Pools (Replaced Single-Bite Discrete Food)
    "RESOURCE_POOL_CAPACITY": 30.0,     # Max food units per pool
    "FEED_RATE_PER_STEP": 3.0,          # Max units consumed per agent per step
    "REPLENISH_RATE_PER_STEP": 0.2,     # Amount the pool regenerates per step
    "HEALTH_PER_FOOD_UNIT": 5.0,        # HP restored per food unit consumed (3.0 * 5.0 = 15.0 HP/step)
    "FOOD_REWARD_SCALE": 2.0,           # Finder reward per unit consumed (3.0 * 2.0 = 6.0 reward/step)

    # 3. Explicit Shared Map (Stigmergy Memory)
    "MAP_DEPOSIT_VALUE": 1.0,           # Trace strength deposited on WRITE_TO_MAP action
    "MAP_DECAY_RATE": 0.85,             # Pheromone evaporation rate
    "MAP_DECAY_INTERVAL": 5,            # Steps between evaporations

    # 4. Visibility Configuration: "local" (3x3 neighborhood) or "full" (global relative vectors)
    "VISIBILITY_MODE": "local",

    # Rewards and Penalties
    "REWARD_VALUES": {
        "DEATH": -100.0,
        "STEP_PENALTY": 0.0,            # Zero movement penalty (movement does not cost extra HP)
        "MAP_ACTION_COST": 0.0,         # Zero cost for writing to map
    },
}
