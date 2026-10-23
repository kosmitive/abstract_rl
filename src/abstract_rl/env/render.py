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


def _mountaincar_frames(obs: np.ndarray) -> list[np.ndarray]:
    """A 280x187 frame per step: the track y = sin(3x), the goal flag at x = 0.45 and the car at obs[:, 0]."""
    width, height, x_min, x_max = 280, 187, -1.2, 0.6
    scale = width / (x_max - x_min)

    def to_px(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (x - x_min) * scale, height - 10 - (np.sin(3 * x) * 0.45 + 0.45) * scale

    track = list(zip(*to_px(np.linspace(x_min, x_max, 100)), strict=True))
    fx, fy = (float(v) for v in to_px(np.array(0.45)))
    frames = []
    for pos in obs[:, 0]:
        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.line(track, fill=(0, 0, 0), width=2)
        draw.line([(fx, fy), (fx, fy - 30)], fill=(0, 0, 0), width=2)
        draw.polygon([(fx, fy - 30), (fx + 18, fy - 25), (fx, fy - 20)], fill=(204, 204, 0))
        cx, cy = (float(v) for v in to_px(np.array(pos)))
        draw.ellipse([cx - 8, cy - 16, cx + 8, cy], fill=(204, 77, 77))
        frames.append(np.asarray(img))
    return frames


# free-camera distance and look-at point of Gymnasium v5's default_camera_config for models without a camera
_FREE_CAMERAS = {
    "InvertedPendulum-v5": (2.04, (0.0, 0.0, 0.0)),
    "InvertedDoublePendulum-v5": (4.1225, (0.0, 0.0, 0.1225)),
}


def _mujoco_frames(model: object, qpos: np.ndarray, env_name: str = "") -> list[np.ndarray]:
    """
    A 240x240 frame per step of the MuJoCo model, posed with the recorded joint positions. The model's first camera
    is used if it has one (HalfCheetah's tracking camera), otherwise a free camera placed as in Gymnasium v5.
    """
    import mujoco

    camera: int | mujoco.MjvCamera = -1
    if model.ncam > 0:  # type: ignore[attr-defined]
        camera = 0
    elif env_name in _FREE_CAMERAS:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.distance, camera.lookat[:] = _FREE_CAMERAS[env_name]
    frames = []
    renderer = mujoco.Renderer(model, 240, 240)
    data = mujoco.MjData(model)
    try:
        for q in qpos:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())
    finally:
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
    if env_wrapper.env_name == "MountainCarContinuous-v0":
        return _mountaincar_frames(np.asarray(trajectory["raw_next_states"])[mask])
    if hasattr(env_wrapper.env, "mj_model"):
        return _mujoco_frames(env_wrapper.env.mj_model, np.asarray(trajectory["qpos"])[mask], env_wrapper.env_name)
    raise NotImplementedError(f"no renderer for environment {env_wrapper.env_name}")
