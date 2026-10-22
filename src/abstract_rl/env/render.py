"""Render the frames of a recorded episode (``execute_policy_once(..., record=True)``) without a display."""

import numpy as np
from PIL import Image, ImageDraw

from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.env.rollout_env_wrapper import RolloutEnvWrapper


def _pendulum_frames(obs: np.ndarray) -> list[np.ndarray]:
    """A 250x250 frame per step: the pole (length 100) at theta = atan2(sin, cos), theta = 0 is upright."""
    frames = []
    for theta in np.arctan2(obs[:, 1], obs[:, 0]):
        img = Image.new("RGB", (250, 250), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        tip = (125 + 100 * np.sin(theta), 125 - 100 * np.cos(theta))
        draw.line([(125, 125), tip], fill=(204, 77, 77), width=12)
        draw.ellipse([120, 120, 130, 130], fill=(0, 0, 0))
        frames.append(np.asarray(img))
    return frames


def _mujoco_frames(model: object, qpos: np.ndarray) -> list[np.ndarray]:
    """A 240x240 frame per step of the MuJoCo model, posed with the recorded joint positions."""
    import mujoco

    frames = []
    renderer = mujoco.Renderer(model, 240, 240)
    data = mujoco.MjData(model)
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        frames.append(renderer.render())
    renderer.close()
    return frames


def render_frames(env_wrapper: RolloutEnvWrapper, trajectory: Trajectory) -> list[np.ndarray]:
    """
    The RGB frames of the valid steps of a recorded single-lane trajectory, one per step after the action.
    :param env_wrapper: The environment wrapper the trajectory was recorded with.
    :param trajectory: The trajectory of ``execute_policy_once(..., record=True)``.
    :return: The list of frames (height x width x 3, uint8).
    """
    mask = np.asarray(trajectory.mask, bool)
    if env_wrapper.env_name == "Pendulum-v1":
        return _pendulum_frames(np.asarray(trajectory["raw_next_states"])[mask])
    if hasattr(env_wrapper.env, "mj_model"):
        return _mujoco_frames(env_wrapper.env.mj_model, np.asarray(trajectory["qpos"])[mask])
    raise NotImplementedError(f"no renderer for environment {env_wrapper.env_name}")
