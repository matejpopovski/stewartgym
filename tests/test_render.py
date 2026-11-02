import numpy as np
import pytest

from stewartgym.evaluation import PIDController
from stewartgym.pid import PIDGains
from stewartgym.scenarios import NOMINAL


@pytest.mark.render
def test_frames_come_from_the_running_simulation(tmp_path):
    from stewartgym.render import episode_frames, save_gif, side_by_side

    frames, trace = episode_frames(
        PIDController(PIDGains()), seed=3, params=NOMINAL, width=120, height=90, stride=100
    )
    assert len(frames) == 5
    assert frames[0].shape == (90, 120, 3)
    assert frames[0].std() > 5.0
    assert np.abs(frames[0].astype(int) - frames[1].astype(int)).sum() > 0
    assert len(trace.distance) == 500

    joined = side_by_side(frames, frames[:3], "left", "right", 2.0)
    assert len(joined) == 5
    path = tmp_path / "clip.gif"
    save_gif(joined, path, 2.0)
    assert path.stat().st_size > 1000
