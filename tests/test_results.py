"""The committed results must be complete and must match what the README claims."""

import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

from stewartgym.evaluation import TEST_SEED_BASE
from stewartgym.policy import NumpyPolicy
from stewartgym.scenarios import SCENARIOS

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def rows():
    with gzip.open(ROOT / "results" / "episodes.csv.gz", "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def test_every_controller_ran_every_scenario_on_the_same_seeds(rows):
    seen = {}
    for row in rows:
        key = (row["controller"], row["scenario"])
        seen.setdefault(key, set()).add(int(row["episode_seed"]))
    assert len({key[0] for key in seen}) == 12
    assert len(seen) == 12 * len(SCENARIOS)
    expected = {TEST_SEED_BASE + i for i in range(200)}
    assert all(value == expected for value in seen.values())


def test_readme_tables_match_the_results():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "report_tables.py"), "--check"], check=True
    )


def test_trained_policies_load_and_tuned_gains_exist():
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    for seed in summary["training_seeds"]:
        for label in ("dr", "nominal"):
            policy = NumpyPolicy.load(ROOT / "policies" / f"ppo_{label}_seed{seed}.npz")
            assert policy.actor[0][0].shape[1] == 46
    for label in ("nominal", "robust"):
        record = json.loads((ROOT / "results" / f"pid_tuning_{label}.json").read_text())
        assert record["chosen"]["score"] > record["baseline"]["hand_defaults_score"]


def test_reproducibility_record_covers_800_episodes_with_no_difference():
    record = json.loads((ROOT / "results" / "reproducibility.json").read_text())
    assert record["episodes_compared"] == 800
    assert record["workers"] <= 8
    assert record["max_abs_return_difference"] == 0.0


def test_readme_step_rate_comes_from_a_run_with_the_ball_in_contact():
    record = json.loads((ROOT / "results" / "step_rate.json").read_text())
    assert record["contacts_at_end_of_every_trial"] == 1
    assert record["active_constraint_rows"] > 18
    assert record["processes"] == 1
    median = record["steps_per_second_median"]
    assert f"{median / 1000:.1f}k physics steps per second" in (ROOT / "README.md").read_text()
    one_minute_load = record["load_average_1_5_15_before"][0]
    assert f"load average of {one_minute_load:.1f}" in (ROOT / "README.md").read_text()
