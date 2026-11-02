"""Turn per-episode rows into per-scenario summaries and paired comparisons.

Training seeds are the unit of replication for PPO: means are taken per policy first and the
spread is reported across policies, because pooling episodes would hide a bad seed.
"""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = {
    "episode_return": np.mean,
    "steady_state_error_m": np.nanmean,
    "settling_time_s": np.median,
    "recovery_time_s": np.median,
    "kick_peak_m": np.nanmean,
    "action_rate_rms": np.mean,
}
RATES = {"dropped": "drop_rate", "settled": "settled_rate", "recovered": "recovered_rate"}


def _per_policy(rows: list[dict]) -> dict:
    """Group rows as family -> scenario -> policy name -> list of rows."""
    grouped: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        grouped[row["family"]][row["scenario"]][row["controller"]].append(row)
    return grouped


def _safe(function, values: np.ndarray) -> float:
    if np.all(np.isnan(values)):
        return float("nan")
    return float(function(values))


def policy_summary(rows: list[dict]) -> dict[str, float]:
    summary = {
        name: _safe(function, np.array([row[name] for row in rows], dtype=float))
        for name, function in METRICS.items()
    }
    for field, label in RATES.items():
        summary[label] = float(np.mean([row[field] for row in rows]))
    return summary


def summarize(rows: list[dict]) -> dict:
    """family -> scenario -> metric -> {mean, std, min, max, per_policy}."""
    output: dict = {}
    for family, scenarios in _per_policy(rows).items():
        output[family] = {}
        for scenario, policies in scenarios.items():
            per_policy = {name: policy_summary(group) for name, group in policies.items()}
            names = sorted(per_policy)
            metrics = {}
            for metric in next(iter(per_policy.values())):
                values = np.array([per_policy[name][metric] for name in names])
                finite = values[np.isfinite(values)]
                metrics[metric] = {
                    "mean": float(finite.mean()) if len(finite) else None,
                    "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
                    "min": float(finite.min()) if len(finite) else None,
                    "max": float(finite.max()) if len(finite) else None,
                    "per_policy": dict(zip(names, map(float, values), strict=True)),
                }
            output[family][scenario] = metrics
    return output


def paired_difference(
    baseline: dict[int, float],
    challengers: dict[str, dict[int, float]],
    samples: int = 4000,
    seed: int = 0,
) -> dict[str, float]:
    """Mean of challenger minus baseline with a two-level bootstrap interval.

    Each resample draws training seeds and, independently, episode seeds, so the interval
    carries both sources of variation. Episodes stay paired: the same episode seeds are
    compared on both sides. Episodes are keyed by seed and must match across controllers.
    """
    episodes = sorted(baseline)
    policies = sorted(challengers)
    base = np.array([baseline[e] for e in episodes])
    diffs = np.array([[challengers[p][e] for e in episodes] for p in policies]) - base
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for sample in range(samples):
        chosen_policies = rng.integers(0, len(policies), len(policies))
        chosen_episodes = rng.integers(0, len(episodes), len(episodes))
        means[sample] = diffs[np.ix_(chosen_policies, chosen_episodes)].mean()
    return {
        "mean": float(diffs.mean()),
        "low": float(np.percentile(means, 2.5)),
        "high": float(np.percentile(means, 97.5)),
        "seeds_better": int(np.sum(diffs.mean(axis=1) > 0)),
        "seeds_total": len(policies),
    }


def compare_returns(rows: list[dict], challenger_family: str, baseline_family: str) -> dict:
    """Paired return difference per scenario. A positive mean favours the challenger.

    A baseline family with several policies is averaged per episode, so its own seed-to-seed
    spread does not enter the interval. The PID baselines have a single controller each.
    """
    grouped = _per_policy(rows)
    results = {}
    for scenario in grouped[baseline_family]:
        per_episode: dict[int, list[float]] = defaultdict(list)
        for group in grouped[baseline_family][scenario].values():
            for row in group:
                per_episode[row["episode_seed"]].append(row["episode_return"])
        base = {seed: float(np.mean(values)) for seed, values in per_episode.items()}
        challengers = {
            name: {row["episode_seed"]: row["episode_return"] for row in group}
            for name, group in grouped[challenger_family][scenario].items()
        }
        comparison = paired_difference(base, challengers)
        if comparison["low"] > 0:
            comparison["verdict"] = challenger_family
        elif comparison["high"] < 0:
            comparison["verdict"] = baseline_family
        else:
            comparison["verdict"] = "tie"
        results[scenario] = comparison
    return results


def read_rows(path: Path) -> list[dict]:
    """Load episodes.csv.gz back into the types the evaluator produced."""
    numeric = {
        "episode_return",
        "settling_time_s",
        "steady_state_error_m",
        "kick_peak_m",
        "recovery_time_s",
        "action_rate_rms",
    }
    integers = {"training_seed", "episode_seed", "survived_steps"}
    flags = {"dropped", "settled", "recovered"}
    rows = []
    with gzip.open(path, "rt", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict = dict(raw)
            for key in numeric:
                row[key] = float(raw[key])
            for key in integers:
                row[key] = int(raw[key])
            for key in flags:
                row[key] = raw[key] == "True"
            rows.append(row)
    return rows
