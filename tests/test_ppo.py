import numpy as np
import pytest
import torch

from stewartgym.policy import NumpyActorCritic, NumpyPolicy
from stewartgym.ppo import (
    ActorCritic,
    PPOConfig,
    RunningNorm,
    Trainer,
    compute_gae,
    export_weights,
    ppo_loss,
)
from stewartgym.vecenv import MAX_WORKERS, RolloutRunner, WorkerPool


def test_gae_with_lambda_one_equals_discounted_returns_minus_values():
    rewards = np.array([[1.0], [2.0], [3.0]])
    values = np.array([[0.5], [0.25], [0.125]])
    dones = np.zeros((3, 1))
    advantages, returns = compute_gae(rewards, values, dones, np.array([4.0]), 0.9, 1.0)
    expected = [
        1 + 0.9 * 2 + 0.81 * 3 + 0.729 * 4,
        2 + 0.9 * 3 + 0.81 * 4,
        3 + 0.9 * 4,
    ]
    assert returns[:, 0] == pytest.approx(expected)
    assert advantages[:, 0] == pytest.approx(np.array(expected) - values[:, 0])


def test_gae_with_lambda_zero_is_the_one_step_td_error():
    rewards = np.array([[1.0], [2.0]])
    values = np.array([[0.5], [0.25]])
    advantages, _ = compute_gae(rewards, values, np.zeros((2, 1)), np.array([1.0]), 0.5, 0.0)
    assert advantages[:, 0] == pytest.approx([1 + 0.5 * 0.25 - 0.5, 2 + 0.5 * 1.0 - 0.25])


def test_gae_stops_at_episode_boundaries():
    rewards = np.array([[1.0], [10.0]])
    values = np.zeros((2, 1))
    dones = np.array([[1.0], [0.0]])
    advantages, _ = compute_gae(rewards, values, dones, np.array([0.0]), 0.99, 0.95)
    assert advantages[0, 0] == pytest.approx(1.0)
    assert advantages[1, 0] == pytest.approx(10.0)


def test_running_norm_matches_batch_statistics():
    data = np.random.default_rng(0).normal(3.0, 2.0, (5000, 2))
    norm = RunningNorm(2)
    for chunk in np.array_split(data, 13):
        norm.update(chunk)
    assert norm.mean == pytest.approx(data.mean(axis=0), abs=1e-3)
    assert norm.var == pytest.approx(data.var(axis=0), abs=1e-3)


def small_model():
    torch.manual_seed(0)
    model = ActorCritic(5, 2, (16, 16), -0.5)
    norm = RunningNorm(5)
    norm.update(np.random.default_rng(1).normal(1.0, 3.0, (64, 5)))
    return model, norm


def test_numpy_inference_matches_torch():
    model, norm = small_model()
    weights = export_weights(model, norm)
    policy = NumpyActorCritic(weights)
    obs = np.random.default_rng(2).normal(0.0, 2.0, (7, 5)).astype(np.float32)
    tensor = torch.from_numpy(norm.normalize(obs).astype(np.float32))
    with torch.no_grad():
        assert policy.mean(obs) == pytest.approx(model.actor(tensor).numpy(), abs=2e-5)
        assert policy.value(obs) == pytest.approx(model.value(tensor).numpy(), abs=2e-5)


def test_sampled_log_prob_matches_torch_normal():
    model, norm = small_model()
    policy = NumpyActorCritic(export_weights(model, norm))
    obs = np.random.default_rng(3).normal(0.0, 1.0, (9, 5)).astype(np.float32)
    actions, log_probs, _ = policy.sample(obs, np.random.default_rng(4))
    tensor = torch.from_numpy(norm.normalize(obs).astype(np.float32))
    with torch.no_grad():
        expected = model.distribution(tensor).log_prob(torch.from_numpy(actions)).sum(-1)
    assert log_probs == pytest.approx(expected.numpy(), abs=1e-3)


def test_saved_policy_ignores_critic_and_clips_actions(tmp_path):
    model, norm = small_model()
    path = tmp_path / "policy.npz"
    np.savez(path, **export_weights(model, norm))
    policy = NumpyPolicy.load(path)
    assert np.all(np.abs(policy.act(np.full(5, 100.0))) <= 1.0)


def batch_for(model, log_prob_shift=0.0):
    obs = torch.randn(32, 5, generator=torch.Generator().manual_seed(5))
    actions = torch.randn(32, 2, generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        old = model.distribution(obs).log_prob(actions).sum(-1) + log_prob_shift
    advantages = torch.linspace(-1.0, 1.0, 32)
    return obs, actions, old, advantages, torch.zeros(32)


def test_unchanged_policy_has_zero_kl_and_no_clipping():
    model, _ = small_model()
    _, stats = ppo_loss(model, *batch_for(model), PPOConfig())
    assert stats["approx_kl"] == pytest.approx(0.0, abs=1e-6)
    assert stats["clip_fraction"] == 0.0


def test_clip_bounds_the_objective_when_the_ratio_is_large():
    model, _ = small_model()
    config = PPOConfig(clip_ratio=0.2, value_coef=0.0)
    obs, actions, old, advantages, returns = batch_for(model, log_prob_shift=-5.0)
    loss, stats = ppo_loss(model, obs, actions, old, advantages, returns, config)
    assert stats["clip_fraction"] == 1.0
    # Positive advantages cap at 1.2; negative ones keep the huge ratio, hence min().
    assert stats["policy_loss"] > 0.5


class PointMass:
    """A 1-D reach task with a known answer: steer x to zero. Episodes end at 40 steps."""

    def __init__(self, index):
        self.rng = np.random.default_rng(index)
        self.x = 0.0
        self.t = 0

    def reset(self, level):
        self.x = self.rng.uniform(-1.0, 1.0)
        self.t = 0
        return np.array([self.x])

    def step(self, action):
        self.x += 0.2 * float(np.clip(action[0], -1.0, 1.0))
        self.t += 1
        return np.array([self.x]), -abs(self.x), False, self.t >= 40


class PointMassCollector:
    obs_dim = 1
    act_dim = 1
    num_envs = 16

    def __init__(self):
        slots = [PointMass(i) for i in range(self.num_envs)]
        self.runner = RolloutRunner(slots, seed=0, reward_scale=1.0)

    def collect(self, weights, steps, level, gamma):
        return self.runner.collect(weights, steps, level, gamma)


def test_ppo_learns_a_task_with_a_known_answer():
    torch.set_num_threads(1)
    config = PPOConfig(
        total_steps=24_000,
        rollout_steps=64,
        minibatch_size=128,
        hidden=(32, 32),
        learning_rate=1e-3,
        curriculum_fraction=0.01,
        init_log_std=-0.5,
    )
    trainer = Trainer(config, PointMassCollector())
    first_batch = trainer.collector.collect(
        export_weights(trainer.model, trainer.norm), 64, 1.0, 0.99
    )
    untrained = np.mean(first_batch.episode_returns)
    policy = NumpyPolicy(trainer.train())
    assert np.mean(trainer.recent_returns[-32:]) > 0.25 * untrained
    # The learned policy must push toward zero from either side.
    assert policy.mean(np.array([0.8]))[0] < -0.2
    assert policy.mean(np.array([-0.8]))[0] > 0.2


def test_truncation_keeps_the_value_of_the_cut_off_state():
    trainer = Trainer(PPOConfig(hidden=(8,)), PointMassCollector())
    weights = export_weights(trainer.model, trainer.norm)
    weights["cb1"] = weights["cb1"] + 7.0
    runner = trainer.collector.runner
    rollout = runner.collect(weights, 40, 1.0, 0.5)
    assert rollout.dones[39].all()
    # Rewards at -|x| cannot be positive, so a positive value proves the bootstrap term.
    assert np.all(rollout.rewards[39] > 0.0)
    assert np.all(rollout.rewards[38] <= 0.0)


def test_checkpoint_resume_restores_progress(tmp_path):
    config = PPOConfig(
        total_steps=2048,
        rollout_steps=32,
        minibatch_size=64,
        hidden=(8,),
        checkpoint_every=1,
    )
    first = Trainer(config, PointMassCollector(), tmp_path)
    first.train()
    resumed = Trainer(config, PointMassCollector(), tmp_path, resume=True)
    assert resumed.steps == first.steps
    assert resumed.update == first.update
    for a, b in zip(first.model.parameters(), resumed.model.parameters(), strict=True):
        assert torch.equal(a, b)


def test_worker_limit_is_enforced():
    with pytest.raises(ValueError):
        WorkerPool(MAX_WORKERS + 1, 1, False, 0)


def smoke_train(seed):
    config = PPOConfig(
        seed=seed, total_steps=64, rollout_steps=16, minibatch_size=32, hidden=(16, 16)
    )
    pool = WorkerPool(2, 2, randomize=True, seed=seed)
    try:
        return Trainer(config, pool).train()
    finally:
        pool.close()


def test_training_on_the_real_environment_is_reproducible():
    torch.set_num_threads(1)
    first, second = smoke_train(3), smoke_train(3)
    assert first.keys() == second.keys()
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert not np.array_equal(smoke_train(4)["aw0"], first["aw0"])
