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

"""Dual-arm Franka env driving TCP waypoints.

Currently only ``rotation_repr='rot6d'`` is implemented:
layout ``[L_xyz(3), L_rot6d(6), L_grip(1), R_xyz(3), R_rot6d(6), R_grip(1)]``.
Each step pushes (xyz, quat) into a per-arm CartesianImpedanceTracker via
``move_tcp_pose``; tracking error is soft, not a Ruckig reflex.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.utils.rot6d import matrix_to_rot6d, rot6d_to_quat_xyzw_safe

from ..dual_franka_env import DualFrankaEnv, DualFrankaRobotConfig

ACTION_DIM_PER_ARM = 10  # xyz(3) + rot6d(6) + gripper(1)
PROPRIO_DIM_PER_ARM = 9  # xyz(3) + rot6d(6); gripper has its own slot


@dataclass
class DualFrankaTcpRobotConfig(DualFrankaRobotConfig):
    """Config for :class:`DualFrankaTcpEnv`."""

    # Only "rot6d" is implemented; other values raise NotImplementedError.
    rotation_repr: str = "rot6d"


class DualFrankaTcpEnv(DualFrankaEnv):
    """Dual-arm Franka env with TCP waypoint actions (rotation_repr-selected)."""

    CONFIG_CLS: type[DualFrankaTcpRobotConfig] = DualFrankaTcpRobotConfig

    PER_ARM_ACTION_DIM = ACTION_DIM_PER_ARM
    GRIPPER_IDX_IN_ARM = 9  # xyz(3) + rot6d(6) then gripper

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-arm previous quat for hemisphere alignment across steps.
        self._prev_step_quat = [None, None]

    def reset(self, *, seed=None, options=None):
        self._prev_step_quat = [None, None]
        return super().reset(seed=seed, options=options)

    # ---------------------------------------------------------------- spaces

    def _init_action_obs_spaces(self):
        if self.config.rotation_repr != "rot6d":
            raise NotImplementedError(
                f"DualFrankaTcpEnv currently only supports rotation_repr='rot6d', "
                f"got {self.config.rotation_repr!r}."
            )
        self._cartesian_safety_boxes()

        # rot6d range widened to [-1.5, 1.5] for headroom before Gram-Schmidt.
        rot6d_low = -1.5 * np.ones(6, dtype=np.float32)
        rot6d_high = 1.5 * np.ones(6, dtype=np.float32)
        left_low = np.concatenate(
            [self.config.ee_pose_limit_min[0, :3], rot6d_low, np.array([-1.0])]
        )
        left_high = np.concatenate(
            [self.config.ee_pose_limit_max[0, :3], rot6d_high, np.array([1.0])]
        )
        right_low = np.concatenate(
            [self.config.ee_pose_limit_min[1, :3], rot6d_low, np.array([-1.0])]
        )
        right_high = np.concatenate(
            [self.config.ee_pose_limit_max[1, :3], rot6d_high, np.array([1.0])]
        )
        act_low = np.concatenate([left_low, right_low]).astype(np.float32)
        act_high = np.concatenate([left_high, right_high]).astype(np.float32)
        self.action_space = gym.spaces.Box(act_low, act_high)

        camera_specs = self._all_camera_specs()
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "gripper_position": gym.spaces.Box(-1, 1, shape=(2,)),
                        "tcp_pose_rot6d": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(2 * PROPRIO_DIM_PER_ARM,),
                        ),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        name: gym.spaces.Box(
                            0, 255, shape=(224, 224, 3), dtype=np.uint8
                        )
                        for name, _, _ in camera_specs
                    }
                ),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    # --------------------------------------------------------- step dispatch

    def _dispatch_arm_motion(
        self,
        actions: np.ndarray,
        states: list,
        ctrls: list,
        dt: float,
    ) -> None:
        del dt

        for arm in range(2):
            xyz = actions[arm, 0:3]
            rot6d = actions[arm, 3:9]

            prev_quat = self._prev_step_quat[arm]
            if prev_quat is None:
                prev_quat = states[arm].tcp_pose[3:]
            quat = rot6d_to_quat_xyzw_safe(rot6d, fallback_quat_xyzw=prev_quat)
            if float(np.dot(quat, prev_quat)) < 0.0:
                quat = -quat
            self._prev_step_quat[arm] = quat

            ctrls[arm].move_tcp_pose(np.concatenate([xyz, quat]).astype(np.float64))

    # --------------------------------------------------- manual delta control

    def step_arm_delta(
        self,
        arm: str,
        delta_xyz=None,
        delta_rpy=None,
        gripper_open: bool | None = None,
        frame: str = "base",
    ):
        """Move one arm by a small TCP delta without commanding the other arm.

        This is a manual-control helper for PhysicalAgent primitives. It does
        not change the policy-facing ``step`` semantics: ``step`` remains the
        absolute 20-D TCP-rot6d action path used by the trained VLA.

        Args:
            arm: ``"left"`` or ``"right"``.
            delta_xyz: translation delta in meters. Defaults to zero.
            delta_rpy: roll/pitch/yaw delta in radians. Defaults to zero.
            gripper_open: ``True`` opens, ``False`` closes, ``None`` holds.
            frame: ``"base"`` applies deltas in the robot base frame;
                ``"eef"`` applies them in the selected end-effector frame.
        """
        start_time = time.time()
        arm_idx = self._manual_arm_index(arm)
        delta_xyz = self._manual_vec3(delta_xyz, "delta_xyz")
        delta_rpy = self._manual_vec3(delta_rpy, "delta_rpy")
        frame = str(frame).lower()
        if frame not in {"base", "eef"}:
            raise ValueError("frame must be 'base' or 'eef'")

        if self.config.is_dummy:
            obs = self._get_observation()
            return obs, 0.0, False, False, {"arm": arm, "dummy": True}

        states = [self._left_state, self._right_state]
        ctrls = [self._left_ctrl, self._right_ctrl]
        state = states[arm_idx]
        ctrl = ctrls[arm_idx]

        start_pose = state.tcp_pose.copy()
        start_rot = R.from_quat(start_pose[3:].copy())
        delta_rot = R.from_euler("xyz", delta_rpy.copy())
        if frame == "eef":
            target_xyz = start_pose[:3] + start_rot.apply(delta_xyz)
            target_rot = start_rot * delta_rot
        else:
            target_xyz = start_pose[:3] + delta_xyz
            target_rot = delta_rot * start_rot

        target_xyz = np.clip(
            target_xyz,
            self._xyz_safe_spaces[arm_idx].low,
            self._xyz_safe_spaces[arm_idx].high,
        )
        target_pose = np.concatenate(
            [target_xyz, target_rot.as_quat()],
        ).astype(np.float64)

        is_gripper_effective = [False, False]
        if gripper_open is not None:
            gripper_cmd = 1.0 if bool(gripper_open) else -1.0
            is_gripper_effective[arm_idx] = self._gripper_action(
                arm_idx,
                ctrl,
                state,
                gripper_cmd,
            )

        if np.any(delta_xyz != 0.0) or np.any(delta_rpy != 0.0):
            ctrl.move_tcp_pose(target_pose)
            self._prev_step_quat[arm_idx] = target_pose[3:].astype(np.float32)

        self._num_steps += 1
        if self._pace_between_action_and_state_read():
            step_time = time.time() - start_time
            time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))

        left_st_f = self._left_ctrl.get_state()
        right_st_f = self._right_ctrl.get_state()
        self._left_state = left_st_f.wait()[0]
        self._right_state = right_st_f.wait()[0]

        observation = self._get_observation()
        reward = self._calc_step_reward(is_gripper_effective)
        terminated = (reward == 1.0) and (
            self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps
        info = {
            "arm": "left" if arm_idx == 0 else "right",
            "frame": frame,
            "start_tcp_pose": start_pose.astype(np.float32),
            "target_tcp_pose": target_pose.astype(np.float32),
            "gripper_open": gripper_open,
            "gripper_effective": is_gripper_effective[arm_idx],
            "other_arm_commanded": False,
        }
        return observation, reward, terminated, truncated, info

    def step_arm_pose(
        self,
        arm: str,
        target_pose,
        gripper_open: bool | None = None,
    ):
        """Move one arm toward one absolute TCP pose in its robot base frame.

        Args:
            arm: ``"left"`` or ``"right"``.
            target_pose: Length-7 ``[xyz, quat_xyzw]`` target pose.
            gripper_open: ``True`` opens, ``False`` closes, ``None`` holds.

        Returns:
            Gym-style ``(observation, reward, terminated, truncated, info)``.
        """
        start_time = time.time()
        arm_idx = self._manual_arm_index(arm)
        target_pose = self._manual_pose7(target_pose, "target_pose")

        if self.config.is_dummy:
            obs = self._get_observation()
            return obs, 0.0, False, False, {"arm": arm, "dummy": True}

        states = [self._left_state, self._right_state]
        ctrls = [self._left_ctrl, self._right_ctrl]
        state = states[arm_idx]
        ctrl = ctrls[arm_idx]
        start_pose = state.tcp_pose.copy()

        target_pose[:3] = np.clip(
            target_pose[:3],
            self._xyz_safe_spaces[arm_idx].low,
            self._xyz_safe_spaces[arm_idx].high,
        )
        reference_quat = self._prev_step_quat[arm_idx]
        if reference_quat is None:
            reference_quat = start_pose[3:]
        if float(np.dot(target_pose[3:], reference_quat)) < 0.0:
            target_pose[3:] = -target_pose[3:]

        is_gripper_effective = [False, False]
        if gripper_open is not None:
            gripper_cmd = 1.0 if bool(gripper_open) else -1.0
            is_gripper_effective[arm_idx] = self._gripper_action(
                arm_idx,
                ctrl,
                state,
                gripper_cmd,
            )

        ctrl.move_tcp_pose(target_pose.astype(np.float64))
        self._prev_step_quat[arm_idx] = target_pose[3:].astype(np.float32)

        self._num_steps += 1
        if self._pace_between_action_and_state_read():
            step_time = time.time() - start_time
            time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))

        left_st_f = self._left_ctrl.get_state()
        right_st_f = self._right_ctrl.get_state()
        self._left_state = left_st_f.wait()[0]
        self._right_state = right_st_f.wait()[0]

        observation = self._get_observation()
        reward = self._calc_step_reward(is_gripper_effective)
        terminated = (reward == 1.0) and (
            self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps
        info = {
            "arm": "left" if arm_idx == 0 else "right",
            "frame": "base",
            "control": "absolute_pose",
            "start_tcp_pose": start_pose.astype(np.float32),
            "target_tcp_pose": target_pose.astype(np.float32),
            "gripper_open": gripper_open,
            "gripper_effective": is_gripper_effective[arm_idx],
            "other_arm_commanded": False,
        }
        return observation, reward, terminated, truncated, info

    def set_arm_gripper(self, arm: str, open: bool):
        """Open or close one gripper without sending TCP commands to either arm."""
        return self.step_arm_delta(arm=arm, gripper_open=bool(open))

    @staticmethod
    def _manual_arm_index(arm: str) -> int:
        arm_name = str(arm).lower()
        if arm_name == "left":
            return 0
        if arm_name == "right":
            return 1
        raise ValueError("arm must be 'left' or 'right'")

    @staticmethod
    def _manual_vec3(value, name: str) -> np.ndarray:
        if value is None:
            return np.zeros(3, dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (3,):
            raise ValueError(f"{name} must be a length-3 vector, got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must contain only finite values")
        return arr

    @staticmethod
    def _manual_pose7(value, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64).copy()
        if arr.shape != (7,):
            raise ValueError(f"{name} must be a length-7 pose, got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must contain only finite values")
        quat_norm = float(np.linalg.norm(arr[3:]))
        if quat_norm <= 1e-8:
            raise ValueError(f"{name} quaternion norm must be positive")
        arr[3:] /= quat_norm
        return arr

    # ------------------------------------------------------------ obs + utils

    def _get_observation(self) -> dict:
        if self.config.is_dummy:
            return self._base_observation_space.sample()
        frames = self._get_camera_frames()

        state = {
            "gripper_position": np.array(
                [
                    self._left_state.gripper_position,
                    self._right_state.gripper_position,
                ],
                dtype=np.float32,
            ),
            "tcp_pose_rot6d": self._tcp_rot6d_18d(),
        }
        return copy.deepcopy({"state": state, "frames": frames})

    def _tcp_rot6d_18d(self) -> np.ndarray:
        """[L_xyz, L_rot6d, R_xyz, R_rot6d] (no euler → no wrap artifacts)."""
        out = np.zeros(18, dtype=np.float32)
        for arm, st in enumerate((self._left_state, self._right_state)):
            base = arm * PROPRIO_DIM_PER_ARM
            out[base : base + 3] = st.tcp_pose[:3]
            mat = R.from_quat(st.tcp_pose[3:]).as_matrix()
            out[base + 3 : base + 9] = matrix_to_rot6d(mat)
        return out
