# Configuration for Custom MARL Environment & Exploration Setup

CONFIG = {
    "GRID_SIZE": 20,
    "NUM_AGENTS": 4,
    "NUM_REWARDS": 3,
    "MAX_HEALTH": 30.0,
    "HEALTH_DECAY_PER_STEP": 0.2,       # Lowered from 1.0 -> Gives ~150 steps survival without food
    "WRITE_HEALTH_DECAY": 0.0,          # Zero health loss when writing to map (free marking per supervisor feedback)
    "REWARD_RESPAWN_STEPS": 60,         # Resource recharge period
    "MAP_DECAY_RATE": 0.85,             # Pheromone evaporation factor
    "MAP_DECAY_INTERVAL": 5,            # Every N steps apply evaporation
    "REWARD_VALUES": {
        "FOOD_FOUND": 50.0,             # Reward for consuming food
        "DEATH": -20.0,                 # Reduced from -100.0 to prevent freeze paralysis
        "STEP_PENALTY": -0.01,          # Small exploration incentive
        "MAP_ACTION_COST": 0.0,         # Zero penalty for writing to map (supervisor feedback)
    },
    "OBS_VIEW_RADIUS": 1,               # 3x3 surrounding window (radius=1)
}
