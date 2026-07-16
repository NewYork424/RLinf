# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Policy transforms for dual-Franka joint-space control.

State options:
  - mode='joint_only' (16D): [L_j1..j7, L_grip, R_j1..j7, R_grip]
  - mode='joint_with_tcp' (34D): above + [L_xyz(3), L_rot6d(6), R_xyz(3), R_rot6d(6)]

Actions: [L_j1..j7, L_grip, R_j1..j7, R_grip] (16D, absolute joint positions from GELLO).
"""

import dataclasses
from typing import Literal

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model
from scipy.spatial.transform import Rotation as R

from rlinf.utils.rot6d import matrix_to_rot6d

_STATE_MODE = Literal["joint_only", "joint_with_tcp"]


def _rearrange_state_joint_only(state_68: np.ndarray) -> np.ndarray:
    """Extract 16D joint state from 68D alphabetical concat.

    Source layout (RealWorldEnv._wrap_obs alphabetical):
      [0:2]    gripper_position [L, R]
      [2:16]   joint_position [L_j1..7, R_j1..7]
      [16:30]  joint_velocity
      [30:36]  tcp_force
      [36:50]  tcp_pose (xyz+quat×2)
      [50:56]  tcp_torque
      [56:68]  tcp_vel

    Output: [L_j1..7, L_grip, R_j1..7, R_grip] (16D)
    """
    s = np.asarray(state_68)
    if s.shape[-1] < 16:
        raise ValueError(f"state must be >=16D; got {s.shape}")

    gripper = s[..., 0:2]       # [L_grip, R_grip]
    joints = s[..., 2:16]       # [L_j1..7, R_j1..7]

    # Rearrange to action layout: [L_j1..7, L_grip, R_j1..7, R_grip]
    left_joints = joints[..., 0:7]
    right_joints = joints[..., 7:14]
    left_grip = gripper[..., 0:1]
    right_grip = gripper[..., 1:2]

    return np.concatenate([left_joints, left_grip, right_joints, right_grip], axis=-1)


def _rearrange_state_joint_with_tcp(state_68: np.ndarray) -> np.ndarray:
    """Extract 34D state: 16D joint + 18D TCP rot6d.

    Output: [L_j1..7, L_grip, R_j1..7, R_grip,
             L_xyz(3), L_rot6d(6), R_xyz(3), R_rot6d(6)]
    """
    s = np.asarray(state_68)
    if s.shape[-1] < 50:
        raise ValueError(f"state must be >=50D for tcp_pose extraction; got {s.shape}")

    # First 16D: joint_only
    joint_16 = _rearrange_state_joint_only(s)

    # Extract TCP: tcp_pose is [36:50] = [L_xyz(3), L_quat(4), R_xyz(3), R_quat(4)]
    tcp_pose = s[..., 36:50]
    left_xyz = tcp_pose[..., 0:3]
    left_quat = tcp_pose[..., 3:7]  # xyzw
    right_xyz = tcp_pose[..., 7:10]
    right_quat = tcp_pose[..., 10:14]

    # Convert quat → rot6d
    left_mat = R.from_quat(left_quat).as_matrix()
    right_mat = R.from_quat(right_quat).as_matrix()
    left_rot6d = matrix_to_rot6d(left_mat)
    right_rot6d = matrix_to_rot6d(right_mat)

    return np.concatenate(
        [joint_16, left_xyz, left_rot6d, right_xyz, right_rot6d], axis=-1
    )


def make_dual_franka_joint_example(state_mode: _STATE_MODE = "joint_only") -> dict:
    """Creates a random input example for the dual-Franka joint policy."""
    state_dim = 16 if state_mode == "joint_only" else 34
    return {
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/extra_view_image-0": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/extra_view_image-1": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/state": np.random.rand(state_dim).astype(np.float32),
        "actions": np.random.rand(16).astype(np.float32),
        "prompt": "pick up the cup",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to [H, W, C] uint8."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _extract_extra_views(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (base, right_wrist) from stacked (inference) or split (training)."""
    stacked = data.get("observation/extra_view_image")
    if stacked is not None:
        extra = np.asarray(stacked)
        return _parse_image(extra[0]), _parse_image(extra[1])
    return (
        _parse_image(data["observation/extra_view_image-0"]),
        _parse_image(data["observation/extra_view_image-1"]),
    )


@dataclasses.dataclass(frozen=True)
class DualFrankaJointInputs(transforms.DataTransformFn):
    """Prepare dual-Franka joint-space inputs for PI0/PI05.

    Extracts state (16D or 34D depending on state_mode), images, prompt, actions.
    """

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0
    state_mode: _STATE_MODE = "joint_only"

    def __call__(self, data: dict) -> dict:
        # Extract and rearrange state
        if self.state_mode == "joint_only":
            state = _rearrange_state_joint_only(data["observation/state"])
        else:
            state = _rearrange_state_joint_with_tcp(data["observation/state"])

        state = transforms.pad_to_dim(state, self.action_dim)

        # Parse images
        image = _parse_image(data["observation/image"])
        base_image, right_wrist_image = _extract_extra_views(data)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class DualFrankaJointOutputs(transforms.DataTransformFn):
    """Recover 16-d joint action from padded model output."""

    output_action_dim: int = 16

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.output_action_dim])}
