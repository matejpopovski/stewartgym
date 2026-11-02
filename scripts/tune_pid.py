"""Tune the cascaded PID and write gains plus the full search log to results/.

python scripts/tune_pid.py           tunes on nominal physics
python scripts/tune_pid.py --randomize  tunes on the randomization box PPO trains on
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from stewartgym.tuning import TuningConfig, tune

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = TuningConfig(randomize=args.randomize, workers=args.workers, seed=args.seed)
    record = tune(config)
    label = "robust" if args.randomize else "nominal"
    path = ROOT / "results" / f"pid_tuning_{label}.json"
    path.write_text(json.dumps(asdict(record), indent=1) + "\n")
    print(json.dumps({"chosen": record.chosen, "baseline": record.baseline, "s": record.seconds}))


if __name__ == "__main__":
    main()
