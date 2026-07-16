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
"""Data config for dual-Franka joint-space control with OpenPI DeltaActions."""

import dataclasses

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory

from rlinf.models.embodiment.openpi.policies import dual_franka_joint_policy


@dataclasses.dataclass(frozen=True)
class DualFrankaJointDataConfig(DataConfigFactory):
    """DataConfig for dual-Franka joint-space SFT with delta actions.

    State is 16D [L_joints(7), L_grip, R_joints(7), R_grip] by default.
    Set state_mode='joint_with_tcp' for 34D (adds TCP rot6d as auxiliary info).

    Actions are 16D joint positions (absolute from GELLO). With extra_delta_transform=True,
    training converts to delta via OpenPI's DeltaActions (simple subtraction, not SE(3)).
    """

    default_prompt: str | None = None
    extra_delta_transform: bool = True  # Use delta actions for training
    state_mode: str = "joint_only"  # 'joint_only' (16D) or 'joint_with_tcp' (34D)

    def create(self, assets_dirs, model_config) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/extra_view_image-0": "extra_view_image-0",
                        "observation/extra_view_image-1": "extra_view_image-1",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                dual_franka_joint_policy.DualFrankaJointInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    state_mode=self.state_mode,
                )
            ],
            outputs=[dual_franka_joint_policy.DualFrankaJointOutputs()],
        )

        # Joint-space delta: joints do element-wise subtraction, grippers pass-through (absolute)
        if self.extra_delta_transform:
            # Mask: True for joint dimensions (do delta), False for gripper (absolute)
            joint_delta_mask = [
                True, True, True, True, True, True, True,  # Left arm 7 joints
                False,  # Left gripper (scalar absolute)
                True, True, True, True, True, True, True,  # Right arm 7 joints
                False,  # Right gripper (scalar absolute)
            ]
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(mask=joint_delta_mask)],
                outputs=[_transforms.AbsoluteActions(mask=joint_delta_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("actions",),
        )
