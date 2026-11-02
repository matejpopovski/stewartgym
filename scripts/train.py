"""Train one PPO policy. Example: python scripts/train.py --seed 0 --randomize"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VALIDATION_SEEDS = list(range(30_000_000, 30_000_064))


def main() -> None:
    # Spawned workers re-import this file, so torch stays out of module scope: importing it
    # here keeps each of the eight workers several hundred MB lighter.
    import torch

    from stewartgym.ppo import PPOConfig, Trainer
    from stewartgym.vecenv import WorkerPool

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize", action="store_true", help="domain randomization on")
    parser.add_argument("--steps", type=int, default=PPOConfig.total_steps)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--envs-per-worker", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)
    name = args.name or f"ppo_{'dr' if args.randomize else 'nominal'}_seed{args.seed}"
    config = PPOConfig(seed=args.seed, total_steps=args.steps)
    pool = WorkerPool(args.workers, args.envs_per_worker, args.randomize, args.seed)

    def validate(weights: dict) -> dict:
        nominal = pool.evaluate(weights, VALIDATION_SEEDS, randomize=False)
        randomized = pool.evaluate(weights, VALIDATION_SEEDS, randomize=True)
        return {
            "validation_nominal": float(np.mean(nominal)),
            "validation_randomized": float(np.mean(randomized)),
        }

    run_dir = ROOT / "runs" / name
    try:
        trainer = Trainer(config, pool, run_dir, resume=args.resume, validator=validate)
        weights = trainer.train()
        scores = validate(weights)
    finally:
        pool.close()

    policies = ROOT / "policies"
    policies.mkdir(exist_ok=True)
    # Deployment needs only the actor. The critic stays in the run's checkpoint.
    actor = {key: value for key, value in weights.items() if not key.startswith("c")}
    np.savez(policies / f"{name}.npz", **actor)
    training = ROOT / "results" / "training"
    training.mkdir(parents=True, exist_ok=True)
    (training / f"{name}.jsonl").write_text((run_dir / "log.jsonl").read_text())
    print(json.dumps({"name": name, "seconds": round(trainer.elapsed), **scores}))


if __name__ == "__main__":
    main()
