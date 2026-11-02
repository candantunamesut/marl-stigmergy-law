import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from typing import Tuple, Optional


class ActorCritic(nn.Module):
    """
    Multi-Agent Actor-Critic Network for MARL Survival & Stigmergy Exploration.
    Supports both PPO gradient updates and Evolutionary Algorithm flat parameter mutations.
    """
    def __init__(self, obs_dim: int = 30, action_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

        # Actor (Policy) Head -> Action Logits
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

    # Helper methods for Evolutionary Algorithm (Neuroevolution)
    def get_flat_weights(self) -> np.ndarray:
        """Return all model parameters as a 1D numpy array."""
        params = []
        for param in self.parameters():
            params.append(param.detach().cpu().numpy().ravel())
        return np.concatenate(params)

    def set_flat_weights(self, flat_weights: np.ndarray):
        """Set model parameters from a 1D numpy array."""
        idx = 0
        with torch.no_grad():
            for param in self.parameters():
                size = param.numel()
                shape = param.shape
                chunk = flat_weights[idx : idx + size]
                param.copy_(torch.from_numpy(chunk.reshape(shape)).to(param.device))
                idx += size
