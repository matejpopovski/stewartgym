"""Train the full matrix, one run at a time so no more than eight workers ever exist."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    runs = []
    started = time.perf_counter()
    for randomize in (True, False):
        for seed in args.seeds:
            command = [sys.executable, str(ROOT / "scripts" / "train.py"), "--seed", str(seed)]
            if randomize:
                command.append("--randomize")
            run_start = time.perf_counter()
            output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
            result = json.loads(output.strip().splitlines()[-1])
            result["wall_seconds"] = round(time.perf_counter() - run_start, 1)
            runs.append(result)
            print(json.dumps(result), flush=True)
    summary = {"total_wall_seconds": round(time.perf_counter() - started, 1), "runs": runs}
    (ROOT / "results" / "training" / "wall_clock.json").write_text(
        json.dumps(summary, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
