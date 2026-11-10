# Configuration for Custom MARL Environment & Exploration Setup

CONFIG = {
    "GRID_SIZE": 20,
    "NUM_AGENTS": 4,
    "NUM_REWARDS": 3,
    "MAX_HEALTH": 60.0,
    "HEALTH_DECAY_PER_STEP": 0.2,
    "TEAM_DEATH_HEALTH_PENALTY": 10.0,  # Health lost by teammates when an agent dies
    "SHARED_REWARD_RATIO": 0.10,        # Teammates receive 10% of food reward
    "REWARD_RESPAWN_STEPS": 75,
    "REWARD_VALUES": {
        "FOOD_FOUND": 60.0,             # Reward given to the finder agent
        "DEATH": -100.0,
        "STEP_PENALTY": -0.1,
    },
}
