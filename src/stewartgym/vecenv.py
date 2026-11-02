"""Parallel rollout collection.

Each worker owns a few environments and collects a whole rollout on its own with the weights it
was sent. The learner only meets the workers between rollouts, so one process losing its core
to another job delays that worker's batch and does not stall every control tick of the others.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Protocol

import numpy as np

from stewartgym.env import EPISODE_STEPS, OBS_DIM, BallBalanceEnv
from stewartgym.policy import NumpyActorCritic, NumpyPolicy

MAX_WORKERS = 8
ACT_DIM = 2


@dataclass
class Rollout:
    """Arrays are (steps, envs, ...). Rewards already include time-limit bootstrapping."""

    obs: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    last_values: np.ndarray
    episode_returns: list[float]
    episode_lengths: list[int]
    episode_drops: list[bool]


class Collector(Protocol):
    num_envs: int
    obs_dim: int
    act_dim: int

    def collect(self, weights: dict, steps: int, level: float, gamma: float) -> Rollout: ...


class EnvSlot:
    """One ball-balancing env plus the bookkeeping that seeds each of its episodes separately."""

    def __init__(self, seed: int, index: int, randomize: bool) -> None:
        self.env = BallBalanceEnv()
        self.seed = seed
        self.index = index
        self.randomize = randomize
        self.episodes = 0

    def reset(self, level: float) -> np.ndarray:
        sequence = np.random.SeedSequence([self.seed, self.index, self.episodes])
        self.episodes += 1
        return self.env.reset(
            sequence,
            randomize_level=level if self.randomize else None,
            curriculum_level=level,
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool]:
        observation, reward, terminated, truncated, _ = self.env.step(action)
        return observation, reward, terminated, truncated


class RolloutRunner:
    """Keeps envs mid-episode across rollouts, so episodes span collection boundaries."""

    def __init__(self, slots: list, seed: int, reward_scale: float = 0.1) -> None:
        self.slots = slots
        self.rng = np.random.default_rng(seed)
        self.reward_scale = reward_scale
        self.obs: np.ndarray | None = None
        self.running_return = np.zeros(len(slots))
        self.running_length = np.zeros(len(slots), dtype=int)

    def collect(self, weights: dict, steps: int, level: float, gamma: float) -> Rollout:
        policy = NumpyActorCritic(weights)
        count = len(self.slots)
        if self.obs is None:
            self.obs = np.stack([slot.reset(level) for slot in self.slots])
        obs_dim = self.obs.shape[1]
        act_dim = weights["log_std"].shape[0]
        rollout = Rollout(
            obs=np.zeros((steps, count, obs_dim), dtype=np.float32),
            actions=np.zeros((steps, count, act_dim), dtype=np.float32),
            log_probs=np.zeros((steps, count), dtype=np.float32),
            values=np.zeros((steps, count), dtype=np.float32),
            rewards=np.zeros((steps, count), dtype=np.float32),
            dones=np.zeros((steps, count), dtype=np.float32),
            last_values=np.zeros(count, dtype=np.float32),
            episode_returns=[],
            episode_lengths=[],
            episode_drops=[],
        )
        for step in range(steps):
            actions, log_probs, values = policy.sample(self.obs, self.rng)
            rollout.obs[step] = self.obs
            rollout.actions[step] = actions
            rollout.log_probs[step] = log_probs
            rollout.values[step] = values
            for i, slot in enumerate(self.slots):
                observation, reward, terminated, truncated = slot.step(actions[i])
                self.running_return[i] += reward
                self.running_length[i] += 1
                scaled = reward * self.reward_scale
                if truncated:
                    # A time limit is not a failure: keep the value of the state it cut off.
                    scaled += gamma * float(policy.value(observation))
                rollout.rewards[step, i] = scaled
                if terminated or truncated:
                    rollout.dones[step, i] = 1.0
                    rollout.episode_returns.append(float(self.running_return[i]))
                    rollout.episode_lengths.append(int(self.running_length[i]))
                    rollout.episode_drops.append(bool(terminated))
                    self.running_return[i] = 0.0
                    self.running_length[i] = 0
                    observation = slot.reset(level)
                self.obs[i] = observation
        rollout.last_values[:] = policy.value(self.obs)
        return rollout


def merge_rollouts(parts: list[Rollout]) -> Rollout:
    arrays = {
        name: np.concatenate([getattr(part, name) for part in parts], axis=1)
        for name in ("obs", "actions", "log_probs", "values", "rewards", "dones")
    }
    return Rollout(
        **arrays,
        last_values=np.concatenate([part.last_values for part in parts]),
        episode_returns=[value for part in parts for value in part.episode_returns],
        episode_lengths=[value for part in parts for value in part.episode_lengths],
        episode_drops=[value for part in parts for value in part.episode_drops],
    )


def evaluate_policy(
    env: BallBalanceEnv, weights: dict, seeds: list[int], randomize: bool
) -> list[float]:
    policy = NumpyPolicy(weights)
    returns = []
    for seed in seeds:
        observation = env.reset(seed, randomize_level=1.0 if randomize else None)
        total = 0.0
        for _ in range(EPISODE_STEPS):
            observation, reward, terminated, truncated, _ = env.step(policy.act(observation))
            total += reward
            if terminated or truncated:
                break
        returns.append(total)
    return returns


def _worker(
    conn: Connection, seed: int, worker: int, envs_per_worker: int, randomize: bool
) -> None:
    slots = [EnvSlot(seed, worker * envs_per_worker + i, randomize) for i in range(envs_per_worker)]
    runner = RolloutRunner(slots, seed=np.random.SeedSequence([seed, worker]).generate_state(1)[0])
    try:
        while True:
            command, payload = conn.recv()
            if command == "collect":
                conn.send(runner.collect(*payload))
            elif command == "evaluate":
                conn.send(evaluate_policy(slots[0].env, *payload))
            elif command == "close":
                break
    finally:
        conn.close()


class WorkerPool:
    """Ball-balancing environments on `workers` processes."""

    obs_dim = OBS_DIM
    act_dim = ACT_DIM

    def __init__(self, workers: int, envs_per_worker: int, randomize: bool, seed: int) -> None:
        if not 1 <= workers <= MAX_WORKERS:
            raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
        # Spawn, not fork: it matches on macOS and Linux and does not inherit library threads.
        context = mp.get_context("spawn")
        self.num_envs = workers * envs_per_worker
        self._connections = []
        self._processes = []
        for worker in range(workers):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker,
                args=(child, seed, worker, envs_per_worker, randomize),
                daemon=True,
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)

    def _gather(self) -> list:
        return [connection.recv() for connection in self._connections]

    def collect(self, weights: dict, steps: int, level: float, gamma: float) -> Rollout:
        for connection in self._connections:
            connection.send(("collect", (weights, steps, level, gamma)))
        return merge_rollouts(self._gather())

    def evaluate(self, weights: dict, seeds: list[int], randomize: bool) -> list[float]:
        count = len(self._connections)
        for index, connection in enumerate(self._connections):
            connection.send(("evaluate", (weights, seeds[index::count], randomize)))
        return [value for part in self._gather() for value in part]

    def close(self) -> None:
        for connection in self._connections:
            connection.send(("close", None))
            connection.close()
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
