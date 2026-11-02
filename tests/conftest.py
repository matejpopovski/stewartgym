import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("MUJOCO_GL", "").lower() not in ("disable", "disabled", "off", "false", "0"):
        return
    skip = pytest.mark.skip(reason="MUJOCO_GL disables rendering on this runner")
    for item in items:
        if "render" in item.keywords:
            item.add_marker(skip)
