"""Off-screen rendering of real episodes with mujoco.Renderer.

Frames come from the same MjData the controller is acting on: the renderer hooks the episode
runner, so a GIF cannot drift from the simulation it claims to show. Importing this module does
not touch OpenGL, so the rest of the package works on machines with no GL at all.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from stewartgym.env import BallBalanceEnv
from stewartgym.evaluation import Controller, Trace, run_episode
from stewartgym.scenarios import PhysicsParams


def episode_frames(
    controller: Controller,
    seed: int,
    params: PhysicsParams,
    camera: str = "bench",
    width: int = 360,
    height: int = 270,
    stride: int = 2,
) -> tuple[list[np.ndarray], Trace]:
    """Run one episode and capture an RGB frame every `stride` control ticks."""
    env = BallBalanceEnv()
    frames: list[np.ndarray] = []
    with mujoco.Renderer(env.model, height=height, width=width) as renderer:

        def capture(current: BallBalanceEnv, step: int) -> None:
            if step % stride == 0:
                renderer.update_scene(current.data, camera=camera)
                frames.append(renderer.render().copy())

        trace = run_episode(env, controller, seed, params, on_step=capture)
    return frames, trace


def _label(frame: np.ndarray, lines: list[str]) -> Image.Image:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for row, text in enumerate(lines):
        draw.text((8, 6 + 13 * row), text, fill=(20, 24, 28), font=font)
    return image


def side_by_side(
    left: list[np.ndarray],
    right: list[np.ndarray],
    left_title: str,
    right_title: str,
    frame_seconds: float,
) -> list[Image.Image]:
    """Join two runs of one episode. The shorter one, if its ball fell, holds its last frame."""
    length = max(len(left), len(right))
    joined = []
    for index in range(length):
        seconds = f"t = {index * frame_seconds:4.1f} s"
        a = _label(left[min(index, len(left) - 1)], [left_title, seconds])
        b = _label(right[min(index, len(right) - 1)], [right_title, seconds])
        canvas = Image.new("RGB", (a.width + b.width + 4, a.height), (255, 255, 255))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width + 4, 0))
        joined.append(canvas)
    return joined


def save_gif(frames: list[Image.Image], path: Path, frame_seconds: float) -> None:
    palette = frames[len(frames) // 2].quantize(colors=96, dither=Image.Dither.NONE)
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=int(frame_seconds * 1000),
        loop=0,
        optimize=True,
    )
