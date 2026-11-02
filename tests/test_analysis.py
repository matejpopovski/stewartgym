import csv
import gzip

import numpy as np
import pytest

from stewartgym.analysis import compare_returns, paired_difference, read_rows, summarize


def row(family, controller, scenario, seed, value, dropped=False):
    return {
        "family": family,
        "controller": controller,
        "scenario": scenario,
        "episode_seed": seed,
        "training_seed": 0,
        "episode_return": value,
        "steady_state_error_m": float("nan") if dropped else 0.001,
        "settling_time_s": 1.0,
        "recovery_time_s": 1.0,
        "kick_peak_m": 0.02,
        "action_rate_rms": 0.1,
        "dropped": dropped,
        "settled": not dropped,
        "recovered": not dropped,
    }


def test_spread_is_across_policies_not_episodes():
    rows = [row("ppo", "a", "s", i, 10.0) for i in range(4)]
    rows += [row("ppo", "b", "s", i, 20.0) for i in range(4)]
    result = summarize(rows)["ppo"]["s"]["episode_return"]
    assert result["mean"] == pytest.approx(15.0)
    assert result["std"] == pytest.approx(np.std([10.0, 20.0], ddof=1))
    assert result["per_policy"] == {"a": 10.0, "b": 20.0}


def test_all_dropped_policy_reports_rates_without_crashing():
    rows = [row("pid", "p", "s", i, -5.0, dropped=True) for i in range(3)]
    result = summarize(rows)["pid"]["s"]
    assert result["drop_rate"]["mean"] == 1.0
    assert result["steady_state_error_m"]["mean"] is None


def test_paired_difference_uses_the_pairing():
    # Episodes vary a lot but the challenger is always exactly 1 better, so a paired
    # interval must be tight even though unpaired variances would swamp the effect.
    base = {i: float(100 * i) for i in range(50)}
    challenger = {"a": {i: value + 1.0 for i, value in base.items()}}
    result = paired_difference(base, challenger, samples=500)
    assert result["mean"] == pytest.approx(1.0)
    assert result["low"] == pytest.approx(1.0)
    assert result["high"] == pytest.approx(1.0)
    assert result["seeds_better"] == 1


def test_a_single_bad_training_seed_widens_the_interval():
    rng = np.random.default_rng(0)
    base = {i: float(rng.normal()) for i in range(100)}
    good = {f"s{k}": {i: v + 1.0 for i, v in base.items()} for k in range(4)}
    bad = {**good, "s4": {i: v - 8.0 for i, v in base.items()}}
    tight = paired_difference(base, good, samples=500)
    wide = paired_difference(base, bad, samples=500)
    assert wide["low"] < 0.0 < tight["low"]
    assert wide["seeds_better"] == 4


def test_verdicts_name_the_winner_or_a_tie():
    rows = []
    for seed in range(40):
        noise = np.sin(seed)
        rows.append(row("pid", "p", "wins", seed, 10.0 + noise))
        rows.append(row("ppo", "q", "wins", seed, 12.0 + noise))
        rows.append(row("pid", "p", "loses", seed, 12.0 + noise))
        rows.append(row("ppo", "q", "loses", seed, 10.0 + noise))
        rows.append(row("pid", "p", "even", seed, 10.0 + noise))
        rows.append(row("ppo", "q", "even", seed, 10.0 + noise + 0.1 * np.cos(7 * seed)))
    result = compare_returns(rows, "ppo", "pid")
    assert result["wins"]["verdict"] == "ppo"
    assert result["loses"]["verdict"] == "pid"
    assert result["even"]["verdict"] == "tie"


def test_a_baseline_family_is_averaged_over_all_of_its_policies():
    rows = []
    for seed in range(20):
        rows.append(row("ppo_a", "a0", "s", seed, 10.0))
        rows.append(row("ppo_a", "a1", "s", seed, 20.0))
        rows.append(row("ppo_b", "b0", "s", seed, 18.0))
    assert compare_returns(rows, "ppo_b", "ppo_a")["s"]["mean"] == pytest.approx(3.0)


def test_saved_rows_round_trip(tmp_path):
    path = tmp_path / "rows.csv.gz"
    original = [row("pid", "p", "s", 3, 12.5), row("pid", "p", "s", 4, 1.5, dropped=True)]
    original = [{**item, "survived_steps": 500} for item in original]
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(original[0]))
        writer.writeheader()
        writer.writerows(original)
    loaded = read_rows(path)
    assert loaded[0]["episode_return"] == 12.5
    assert loaded[0]["episode_seed"] == 3
    assert loaded[1]["dropped"] is True
    assert loaded[0]["dropped"] is False
    assert np.isnan(loaded[1]["steady_state_error_m"])
