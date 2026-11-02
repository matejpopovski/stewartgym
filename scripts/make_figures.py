"""Build the README figures from results/ and the saved controllers."""

import json
import multiprocessing as mp
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stewartgym.env import EPISODE_STEPS
from stewartgym.evaluation import (
    TEST_SEED_BASE,
    pid_spec,
    policy_spec,
    run_episode,
    shared_env,
)
from stewartgym.scenarios import SCENARIO_BY_NAME

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "docs"

INK = "#1d1f21"
MUTED = "#6a6d70"
GRID = "#dfe2e5"
SURFACE = "#fcfcfb"
STYLE = {
    "pid_nominal": ("PID-N, tuned on nominal", "#2a78d6", "s"),
    "pid_robust": ("PID-R, tuned on randomized", "#e34948", "D"),
    "ppo_nominal": ("PPO-N, trained on nominal", "#1baf7a", "^"),
    "ppo_dr": ("PPO-R, trained on randomized", "#eda100", "o"),
}
SCENARIO_ORDER = [
    "nominal",
    "mass_edge",
    "mass_beyond",
    "friction_edge",
    "friction_beyond",
    "latency_edge",
    "latency_beyond",
    "noise_edge",
    "noise_beyond",
    "offset_edge",
    "offset_beyond",
    "combined_edge",
    "combined_beyond",
]
LABELS = {
    "nominal": "nominal",
    "mass_edge": "mass: ball x2, plate x1.7",
    "mass_beyond": "mass: ball x4, plate x3",
    "friction_edge": "friction: rolling x10",
    "friction_beyond": "friction: rolling x40",
    "latency_edge": "latency: 60 ms",
    "latency_beyond": "latency: 100 ms",
    "noise_edge": "noise: 2 mm",
    "noise_beyond": "noise: 5 mm",
    "offset_edge": "leg offset: 1.5 mm",
    "offset_beyond": "leg offset: 4 mm",
    "combined_edge": "all at box edge",
    "combined_beyond": "all beyond box",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans Mono",
            "font.size": 8.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": False,
        }
    )


def scenario_panels(summary: dict) -> None:
    panels = [
        ("episode_return", "mean return", 1.0, False),
        ("steady_state_error_m", "steady error (mm)", 1000.0, True),
        ("settling_time_s", "median settling (s)", 1.0, False),
        ("recovery_time_s", "median recovery (s)", 1.0, False),
        ("drop_rate", "ball dropped (%)", 100.0, False),
    ]
    rows = np.arange(len(SCENARIO_ORDER))[::-1]
    figure, axes = plt.subplots(1, len(panels), figsize=(12.0, 5.6), sharey=True)
    offsets = {"pid_nominal": 0.27, "pid_robust": 0.09, "ppo_nominal": -0.09, "ppo_dr": -0.27}
    for axis, (metric, title, scale, log) in zip(axes, panels, strict=True):
        for family, (label, color, marker) in STYLE.items():
            for name, y in zip(SCENARIO_ORDER, rows + offsets[family], strict=True):
                entry = summary[family][name][metric]
                values = [v for v in entry["per_policy"].values() if np.isfinite(v)]
                if not values:
                    continue
                points = np.array(values) * scale
                axis.scatter(points, [y] * len(points), s=9, color=color, alpha=0.55, linewidths=0)
                axis.scatter(
                    points.mean(),
                    y,
                    s=30,
                    color=color,
                    marker=marker,
                    edgecolors=SURFACE,
                    linewidths=0.8,
                    label=label if name == "nominal" else None,
                    zorder=3,
                )
        axis.set_title(title, fontsize=9, loc="left", color=INK)
        if log:
            axis.set_xscale("log")
        axis.set_yticks(rows)
        axis.set_yticklabels([LABELS[name] for name in SCENARIO_ORDER])
        axis.grid(axis="y", visible=False)
        for row in rows[:-1]:
            axis.axhline(row - 0.5, color=GRID, linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.55, -0.01))
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(FIGURES / "fig_scenarios.png", dpi=150)
    plt.close(figure)


def _trace_task(task: tuple) -> np.ndarray:
    spec, params, seed = task
    trace = run_episode(shared_env(), spec.build(), seed, params)
    distance = np.full(EPISODE_STEPS, np.nan)
    distance[: len(trace.distance)] = trace.distance
    return distance


def traces(pool) -> None:
    names = [
        "nominal",
        "friction_beyond",
        "latency_beyond",
        "noise_beyond",
        "offset_beyond",
        "combined_edge",
    ]
    specs = [pid_spec(RESULTS, "robust"), policy_spec(ROOT / "policies", "dr", 0)]
    seeds = [TEST_SEED_BASE + i for i in range(48)]
    time = np.arange(EPISODE_STEPS) * 0.02
    figure, axes = plt.subplots(2, 3, figsize=(11.0, 5.2), sharex=True, sharey=True)
    for axis, name in zip(axes.ravel(), names, strict=True):
        params = SCENARIO_BY_NAME[name].params
        for spec in specs:
            tasks = [(spec, params, seed) for seed in seeds]
            distance = np.array(pool.map(_trace_task, tasks, chunksize=4)) * 100.0
            # A dropped ball stays at the plate edge so it still weighs on the quantiles.
            distance = np.where(np.isnan(distance), 15.0, distance)
            low, median, high = np.percentile(distance, [25, 50, 75], axis=0)
            label, color, _ = STYLE[spec.family]
            axis.fill_between(time, low, high, color=color, alpha=0.22, linewidth=0)
            axis.plot(time, median, color=color, linewidth=1.4, label=label)
        axis.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        axis.axvline(6.0, color=MUTED, linewidth=0.8, linestyle=":")
        axis.set_yscale("log")
        axis.set_ylim(0.03, 20)
        axis.set_title(LABELS[name], fontsize=9, loc="left", color=INK)
    for axis in axes[1]:
        axis.set_xlabel("time (s)")
    for axis in axes[:, 0]:
        axis.set_ylabel("distance to target (cm)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(FIGURES / "fig_traces.png", dpi=150)
    plt.close(figure)


def _pid_validation(task: tuple) -> float:
    spec, seed, randomize = task
    trace = run_episode(
        shared_env(),
        spec.build(),
        seed,
        randomize_level=1.0 if randomize else None,
        test_episode=False,
    )
    return trace.episode_return


def learning_curves(pool) -> None:
    seeds = range(30_000_000, 30_000_064)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    panels = [
        ("validation_nominal", "held-out nominal physics", False),
        ("validation_randomized", "held-out randomized physics", True),
    ]
    logs = sorted((RESULTS / "training").glob("ppo_*.jsonl"))
    for axis, (key, title, randomize) in zip(axes, panels, strict=True):
        for path in logs:
            family = "ppo_dr" if "_dr_" in path.name else "ppo_nominal"
            entries = [json.loads(line) for line in path.read_text().splitlines()]
            points = [(e["steps"] / 1e6, e[key]) for e in entries if key in e]
            x, y = zip(*points, strict=True)
            axis.plot(x, y, color=STYLE[family][1], linewidth=1.0, alpha=0.85)
        for label in ("nominal", "robust"):
            spec = pid_spec(RESULTS, label)
            tasks = [(spec, seed, randomize) for seed in seeds]
            score = float(np.mean(pool.map(_pid_validation, tasks, chunksize=4)))
            axis.axhline(score, color=STYLE[spec.family][1], linewidth=1.2, linestyle=(0, (5, 3)))
        axis.set_title(title, fontsize=9, loc="left", color=INK)
        axis.set_xlabel("environment steps (millions)")
    axes[0].set_ylabel("mean return, 64 episodes")
    handles = [
        plt.Line2D([], [], color=STYLE[f][1], linewidth=1.3, linestyle=style, label=STYLE[f][0])
        for f, style in (
            ("ppo_dr", "-"),
            ("ppo_nominal", "-"),
            ("pid_nominal", (0, (5, 3))),
            ("pid_robust", (0, (5, 3))),
        )
    ]
    figure.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(FIGURES / "fig_learning.png", dpi=150)
    plt.close(figure)


def tuning_progress() -> None:
    figure, axis = plt.subplots(figsize=(6.4, 3.2))
    for label, family in (("nominal", "pid_nominal"), ("robust", "pid_robust")):
        record = json.loads((RESULTS / f"pid_tuning_{label}.json").read_text())
        scores = [entry["score"] for entry in record["evaluations"]]
        best = np.maximum.accumulate(scores)
        name, color, _ = STYLE[family]
        axis.plot(np.arange(len(best)), best, color=color, linewidth=1.5, label=name)
        axis.scatter(np.arange(len(scores)), scores, s=3, color=color, alpha=0.25, linewidths=0)
    axis.set_xlabel("candidate evaluated")
    axis.set_ylabel("mean return, 32 tuning episodes")
    low = min(entry for entry in best) - 100
    axis.set_ylim(low, None)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(FIGURES / "fig_pid_tuning.png", dpi=150)
    plt.close(figure)


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    summary = json.loads((RESULTS / "summary.json").read_text())["summary"]
    scenario_panels(summary)
    tuning_progress()
    with mp.get_context("spawn").Pool(8) as pool:
        traces(pool)
        learning_curves(pool)


if __name__ == "__main__":
    main()
