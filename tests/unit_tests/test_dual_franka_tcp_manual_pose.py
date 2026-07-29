from types import SimpleNamespace

import numpy as np

from rlinf.envs.realworld.franka.tasks.dual_franka_tcp_env import (
    DualFrankaTcpEnv,
)


class _ImmediateFuture:
    def __init__(self, value):
        self._value = value

    def wait(self):
        return [self._value]


class _FakeController:
    def __init__(self, state):
        self.state = state
        self.pose_commands = []

    def move_tcp_pose(self, pose):
        pose = np.asarray(pose, dtype=np.float64).copy()
        self.pose_commands.append(pose)
        self.state.tcp_pose = pose

    def get_state(self):
        return _ImmediateFuture(self.state)


def test_step_arm_pose_commands_only_selected_arm():
    env = DualFrankaTcpEnv.__new__(DualFrankaTcpEnv)
    env.config = SimpleNamespace(
        is_dummy=False,
        step_frequency=10.0,
        success_hold_steps=1,
        max_num_steps=300,
    )
    env._prev_step_quat = [None, None]
    env._num_steps = 0
    env._success_hold_counter = 0

    identity_pose = np.asarray(
        [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float64,
    )
    env._left_state = SimpleNamespace(tcp_pose=identity_pose.copy())
    env._right_state = SimpleNamespace(tcp_pose=identity_pose.copy())
    env._left_ctrl = _FakeController(env._left_state)
    env._right_ctrl = _FakeController(env._right_state)
    safe_space = SimpleNamespace(
        low=np.asarray([0.3, -0.8, 0.1]),
        high=np.asarray([0.8, 0.8, 0.7]),
    )
    env._xyz_safe_spaces = [safe_space, safe_space]
    env._pace_between_action_and_state_read = lambda: False
    env._get_observation = lambda: {"state": {}}
    env._calc_step_reward = lambda _effective: 0.0

    requested_pose = np.asarray(
        [0.9, 0.2, 0.6, 0.0, 0.0, 0.0, 2.0],
        dtype=np.float64,
    )
    _, _, _, _, info = env.step_arm_pose("left", requested_pose)

    assert len(env._left_ctrl.pose_commands) == 1
    assert env._right_ctrl.pose_commands == []
    assert np.allclose(env._left_ctrl.pose_commands[0][:3], [0.8, 0.2, 0.6])
    assert np.allclose(env._left_ctrl.pose_commands[0][3:], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(requested_pose, [0.9, 0.2, 0.6, 0.0, 0.0, 0.0, 2.0])
    assert info["control"] == "absolute_pose"
    assert info["other_arm_commanded"] is False
