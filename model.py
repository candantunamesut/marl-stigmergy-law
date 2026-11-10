import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from typing import Tuple, Optional
import numpy as np


class ActorCritic(nn.Module):
    """
    Multi-Agent Actor-Critic Network for MARL Survival & Collective Exploration.
    Shares parameters across homogeneous agents to maximize sample efficiency.
    Includes flat-weight interface for neuroevolution / genetic optimization.
    """
    def __init__(self, obs_dim: int = 21, action_dim: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

        # Actor (Policy) Head -> Action Logits (5 movement actions)
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic (Value) Head -> State Value V(s)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared_net(x)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shared_net(x)
        return self.critic_head(features)

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared_net(x)
        logits = self.actor_head(features)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic_head(features)

        return action, log_prob, entropy, value

    # Flat weight interface for Neuroevolution / Genetic Algorithms
    def get_flat_weights(self) -> np.ndarray:
        """Flattens all trainable parameters into a 1D numpy array."""
        params = [p.data.cpu().numpy().flatten() for p in self.parameters()]
        return np.concatenate(params)

    def set_flat_weights(self, flat_weights: np.ndarray) -> None:
        """Restores model parameters from a 1D numpy array."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            shape = p.shape
            chunk = flat_weights[offset : offset + numel]
            p.data.copy_(torch.from_numpy(chunk.reshape(shape)).to(p.device))
            offset += numel
