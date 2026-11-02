"""Policy inference in plain NumPy, used by rollout workers and by evaluation.

Workers run the policy themselves instead of asking the learner for an action on each control
tick. On a loaded machine that removes the per-tick wait for the slowest process, and it keeps
workers free of torch, which saves several hundred MB each.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def _forward(layers: list[tuple[np.ndarray, np.ndarray]], x: np.ndarray) -> np.ndarray:
    for weight, bias in layers[:-1]:
        x = np.tanh(x @ weight.T + bias)
    weight, bias = layers[-1]
    return x @ weight.T + bias


def _stack(weights: dict[str, np.ndarray], prefix: str) -> list[tuple[np.ndarray, np.ndarray]]:
    count = sum(1 for key in weights if key.startswith(f"{prefix}w"))
    return [(weights[f"{prefix}w{i}"], weights[f"{prefix}b{i}"]) for i in range(count)]


class NumpyPolicy:
    """Tanh MLP actor with frozen observation normalisation, in single precision."""

    def __init__(self, weights: dict[str, np.ndarray]) -> None:
        self.obs_mean = weights["obs_mean"]
        self.obs_std = weights["obs_std"]
        self.actor = _stack(weights, "a")

    @classmethod
    def load(cls, path: str | Path) -> NumpyPolicy:
        with np.load(path) as stored:
            return cls({key: stored[key] for key in stored.files})

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return np.clip((obs.astype(np.float32) - self.obs_mean) / self.obs_std, -10.0, 10.0)

    def mean(self, obs: np.ndarray) -> np.ndarray:
        return _forward(self.actor, self.normalize(obs))

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.clip(self.mean(obs), -1.0, 1.0)


class NumpyActorCritic(NumpyPolicy):
    """Adds the critic and the Gaussian sampling that data collection needs."""

    def __init__(self, weights: dict[str, np.ndarray]) -> None:
        super().__init__(weights)
        self.critic = _stack(weights, "c")
        self.log_std = weights["log_std"]
        self.std = np.exp(self.log_std)

    def value(self, obs: np.ndarray) -> np.ndarray:
        return _forward(self.critic, self.normalize(obs))[..., 0]

    def sample(
        self, obs: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        normalized = self.normalize(obs)
        mean = _forward(self.actor, normalized)
        noise = rng.standard_normal(mean.shape, dtype=np.float32)
        action = mean + self.std * noise
        log_prob = -(0.5 * noise * noise + self.log_std + LOG_SQRT_2PI).sum(axis=-1)
        return action, log_prob, _forward(self.critic, normalized)[..., 0]
