"""Rerun part of the evaluation and compare it with results/episodes.csv.gz.

Two controllers on two conditions over all 200 episode seeds, so 800 episodes. The comparison
is exact: a controller, condition and seed must give back the same return to the last bit
after the round trip through the CSV.
"""

import argparse
import json
from pathlib import Path

from stewartgym.analysis import read_rows
from stewartgym.evaluation import evaluate_matrix, pid_spec, policy_spec
from stewartgym.scenarios import SCENARIO_BY_NAME
from stewartgym.vecenv import MAX_WORKERS

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SCENARIOS = ("nominal", "offset_beyond")
EPISODES = 200


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be between 1 and {MAX_WORKERS}")

    specs = [pid_spec(RESULTS, "robust"), policy_spec(ROOT / "policies", "dr", 0)]
    scenarios = [(name, SCENARIO_BY_NAME[name].params) for name in SCENARIOS]
    rerun = evaluate_matrix(specs, scenarios, EPISODES, args.workers)

    key = ("controller", "scenario", "episode_seed")
    saved = {tuple(row[k] for k in key): row for row in read_rows(RESULTS / "episodes.csv.gz")}
    differences = [
        abs(row["episode_return"] - saved[tuple(row[k] for k in key)]["episode_return"])
        for row in rerun
    ]
    result = {
        "compared_against": "results/episodes.csv.gz",
        "controllers": [spec.name for spec in specs],
        "scenarios": list(SCENARIOS),
        "episodes_per_pair": EPISODES,
        "episodes_compared": len(differences),
        "workers": args.workers,
        "max_abs_return_difference": max(differences),
    }
    (RESULTS / "reproducibility.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
