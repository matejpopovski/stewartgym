"""Measure raw mj_step throughput for the full model on one core.

The ball rests on the plate with the servos holding home, so the closure rows and the contact
are all active, which is the steady state of the task. Soft friction lets a resting ball creep,
and over minutes of simulated time it leaves the plate and the contact solve disappears, which
would inflate the rate. So every trial restarts from the settled state, and the run fails if
the contact is ever lost.

The rate covers mj_step alone: control ticks also pay for Python, IK and the policy. Load
averages are recorded because the machine is shared and the number moves with what else is
running.
"""

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mujoco

from stewartgym.env import MODEL_PATH

OUTPUT = Path(__file__).resolve().parent.parent / "results" / "step_rate.json"
SETTLE_STEPS = 2000
STEPS_PER_TRIAL = 20_000
TRIALS = 15


def cpu_name() -> str:
    if sys.platform == "darwin":
        command = ["sysctl", "-n", "machdep.cpu.brand_string"]
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
    return platform.processor() or platform.machine()


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
    settled = mujoco.MjData(model)
    mujoco.mj_copyData(settled, model, data)
    load_before = os.getloadavg()

    rates = []
    for _ in range(TRIALS):
        mujoco.mj_copyData(data, model, settled)
        started = time.perf_counter()
        for _ in range(STEPS_PER_TRIAL):
            mujoco.mj_step(model, data)
        rates.append(STEPS_PER_TRIAL / (time.perf_counter() - started))
        if data.ncon != 1:
            raise RuntimeError("the ball left the plate during a trial, so the rate is not valid")

    result = {
        "steps_per_second_median": statistics.median(rates),
        "steps_per_second_min": min(rates),
        "steps_per_second_max": max(rates),
        "steps_per_second_trials": rates,
        "steps_per_trial": STEPS_PER_TRIAL,
        "timestep_s": float(model.opt.timestep),
        "nv": int(model.nv),
        "active_constraint_rows": int(data.nefc),
        "contacts_at_end_of_every_trial": int(data.ncon),
        "processes": 1,
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "system": f"{platform.system()} {platform.release()} {platform.machine()}",
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "load_average_1_5_15_before": list(load_before),
        "load_average_1_5_15_after": list(os.getloadavg()),
        "machine_shared_with_other_jobs": True,
    }
    OUTPUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
