"""Run every model check and write the measured values to results/model_validation.json."""

import json
from pathlib import Path

from stewartgym.validation import run_all

OUTPUT = Path(__file__).resolve().parent.parent / "results" / "model_validation.json"


def main() -> None:
    results = run_all()
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
