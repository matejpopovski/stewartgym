"""Evaluate all controllers on all scenarios with identical episode seeds.

Writes results/episodes.csv.gz (one row per episode) and results/summary.json.
"""

import argparse
import csv
import gzip
import json
from pathlib import Path

from stewartgym.analysis import compare_returns, read_rows, summarize
from stewartgym.evaluation import evaluate_matrix, pid_spec, policy_spec
from stewartgym.scenarios import SCENARIOS

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
POLICIES = ROOT / "policies"


def simulate(args: argparse.Namespace) -> list[dict]:
    specs = [
        pid_spec(RESULTS, "nominal"),
        pid_spec(RESULTS, "robust"),
        *(policy_spec(POLICIES, "nominal", seed) for seed in args.seeds),
        *(policy_spec(POLICIES, "dr", seed) for seed in args.seeds),
    ]
    scenarios = [(scenario.name, scenario.params) for scenario in SCENARIOS]
    rows = evaluate_matrix(specs, scenarios, args.episodes, args.workers)
    with gzip.open(RESULTS / "episodes.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--reuse", action="store_true", help="re-analyse the saved episodes")
    args = parser.parse_args()

    rows = read_rows(RESULTS / "episodes.csv.gz") if args.reuse else simulate(args)

    summary = {
        "episodes_per_scenario": args.episodes,
        "training_seeds": args.seeds,
        "scenarios": {
            scenario.name: {
                "group": scenario.group,
                "severity": scenario.severity,
                "params": vars(scenario.params),
            }
            for scenario in SCENARIOS
        },
        "summary": summarize(rows),
        "ppo_dr_vs_pid_robust": compare_returns(rows, "ppo_dr", "pid_robust"),
        "ppo_dr_vs_pid_nominal": compare_returns(rows, "ppo_dr", "pid_nominal"),
        "ppo_nominal_vs_pid_nominal": compare_returns(rows, "ppo_nominal", "pid_nominal"),
        "ppo_dr_vs_ppo_nominal": compare_returns(rows, "ppo_dr", "ppo_nominal"),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
