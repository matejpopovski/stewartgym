"""Print the README result tables from results/summary.json.

With --check, exit non-zero if the block in README.md differs, so numbers cannot drift from
the results they came from.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START, END = "<!-- results:start -->", "<!-- results:end -->"

ROWS = [
    ("nominal", "nominal"),
    ("mass_edge", "mass: ball x2, platform x1.7"),
    ("mass_beyond", "mass: ball x4, platform x3"),
    ("friction_edge", "friction: rolling x10, sliding x0.25"),
    ("friction_beyond", "friction: rolling x40, sliding x0.1"),
    ("latency_edge", "latency 60 ms"),
    ("latency_beyond", "latency 100 ms"),
    ("noise_edge", "sensor noise 2 mm"),
    ("noise_beyond", "sensor noise 5 mm"),
    ("offset_edge", "leg offset 1.5 mm"),
    ("offset_beyond", "leg offset 4 mm"),
    ("combined_edge", "all of the above at box edge"),
    ("combined_beyond", "all of the above beyond box"),
]
VERDICT = {"ppo_dr": "PPO-R", "pid_robust": "PID-R", "tie": "tie"}


def spread(entry: dict, scale: float = 1.0, digits: int = 1) -> str:
    return f"{entry['mean'] * scale:.{digits}f} ± {entry['std'] * scale:.{digits}f}"


def single(entry: dict, scale: float = 1.0, digits: int = 1) -> str:
    return f"{entry['mean'] * scale:.{digits}f}"


def return_table(results: dict) -> list[str]:
    summary = results["summary"]
    verdicts = results["ppo_dr_vs_pid_robust"]
    lines = [
        "| Physics | PID-N | PID-R | PPO-N | PPO-R | PPO-R minus PID-R (95% CI) | Winner |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, label in ROWS:
        verdict = verdicts[name]
        lines.append(
            f"| {label} "
            f"| {single(summary['pid_nominal'][name]['episode_return'])} "
            f"| {single(summary['pid_robust'][name]['episode_return'])} "
            f"| {spread(summary['ppo_nominal'][name]['episode_return'])} "
            f"| {spread(summary['ppo_dr'][name]['episode_return'])} "
            f"| {verdict['mean']:+.1f} ({verdict['low']:+.1f}, {verdict['high']:+.1f}) "
            f"| {VERDICT[verdict['verdict']]} |"
        )
    return lines


def control_table(results: dict) -> list[str]:
    summary = results["summary"]
    lines = [
        "| Physics | Controller | Settled | Settling, median | Steady error | "
        "Recovered | Recovery, median | Kick peak | Dropped | Action rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in ROWS:
        for family, title in (("pid_robust", "PID-R"), ("ppo_dr", "PPO-R")):
            m = summary[family][name]
            error = m["steady_state_error_m"]
            peak = m["kick_peak_m"]
            lines.append(
                f"| {label} | {title} "
                f"| {single(m['settled_rate'], 100.0, 0)}% "
                f"| {single(m['settling_time_s'], 1.0, 2)} s "
                f"| {single(error, 1000.0, 2)} mm "
                f"| {single(m['recovered_rate'], 100.0, 0)}% "
                f"| {single(m['recovery_time_s'], 1.0, 2)} s "
                f"| {single(peak, 1000.0, 1)} mm "
                f"| {single(m['drop_rate'], 100.0, 1)}% "
                f"| {single(m['action_rate_rms'], 1.0, 3)} |"
            )
    return lines


def render(results: dict) -> str:
    return "\n".join(
        [
            *return_table(results),
            "",
            "<details>",
            "<summary>Control metrics for PID-R and PPO-R on every scenario (PPO-R values are "
            "means over five seeds)</summary>",
            "",
            *control_table(results),
            "",
            "</details>",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    results = json.loads((ROOT / "results" / "summary.json").read_text())
    block = render(results)
    if not args.check:
        print(block)
        return
    readme = (ROOT / "README.md").read_text()
    current = readme.split(START)[1].split(END)[0].strip()
    if current != block:
        sys.exit("README result tables differ from results/summary.json")


if __name__ == "__main__":
    main()
