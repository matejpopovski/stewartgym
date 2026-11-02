"""Render side-by-side GIFs from real simulation runs with mujoco.Renderer.

Episodes use the first test seed of each scenario, so the clips are not hand-picked.
"""

import argparse
import json
from pathlib import Path

from stewartgym.evaluation import TEST_SEED_BASE, PIDController, PolicyController, pid_spec
from stewartgym.render import episode_frames, save_gif, side_by_side
from stewartgym.scenarios import SCENARIO_BY_NAME

ROOT = Path(__file__).resolve().parent.parent
STRIDE = 3
FRAME_SECONDS = 0.02 * STRIDE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", nargs="+", choices=list(SCENARIO_BY_NAME))
    parser.add_argument("--camera", default="bench", choices=["bench", "top"])
    parser.add_argument("--policy", default="ppo_dr_seed0")
    args = parser.parse_args()

    gains = pid_spec(ROOT / "results", "robust").gains
    output = {}
    for name in args.scenarios:
        params = SCENARIO_BY_NAME[name].params
        left, left_trace = episode_frames(
            PIDController(gains), TEST_SEED_BASE, params, args.camera, stride=STRIDE
        )
        right, right_trace = episode_frames(
            PolicyController(ROOT / "policies" / f"{args.policy}.npz"),
            TEST_SEED_BASE,
            params,
            args.camera,
            stride=STRIDE,
        )
        frames = side_by_side(left, right, f"PID-R, {name}", f"PPO-R, {name}", FRAME_SECONDS)
        path = ROOT / "docs" / f"{name}.gif"
        save_gif(frames, path, FRAME_SECONDS)
        output[name] = {
            "pid_dropped": left_trace.dropped,
            "ppo_dropped": right_trace.dropped,
            "pid_final_distance_m": float(left_trace.distance[-1]),
            "ppo_final_distance_m": float(right_trace.distance[-1]),
            "bytes": path.stat().st_size,
        }
    print(json.dumps(output, indent=1))


if __name__ == "__main__":
    main()
