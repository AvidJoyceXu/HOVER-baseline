# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import torch

from neural_wbc.core.modes import NeuralWBCModes
from neural_wbc.data import get_data_path

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg
from isaaclab.utils import configclass

from .events import NeuralWBCPlayEventCfg, NeuralWBCTrainEventCfg
from .neural_wbc_env_cfg import NeuralWBCEnvCfg
from .rewards.reward_cfg_x2t2 import NeuralWBCRewardCfgX2T2
from .terrain import HARD_ROUGH_TERRAINS_CFG, flat_terrain

DISTILL_MASK_MODES_ALL = {
    "exbody": {
        "upper_body": [".*waist.*joint.*", ".*shoulder.*joint.*", ".*elbow.*joint.*"],
        "lower_body": ["root.*"],
    },
    "humanplus": {
        "upper_body": [".*waist.*joint.*", ".*shoulder.*joint.*", ".*elbow.*joint.*"],
        "lower_body": [".*hip.*joint.*", ".*knee.*joint.*", ".*ankle.*joint.*", "root.*"],
    },
    "h2o": {
        "upper_body": [
            ".*shoulder.*link.*",
            ".*elbow.*link.*",
            ".*hand.*link.*",
        ],
        "lower_body": [".*ankle.*link.*"],
    },
    "omnih2o": {
        "upper_body": [".*hand.*link.*", ".*head.*link.*"],
    },
}

# X2T2 Robot Configuration - using URDF file
X2T2_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=UrdfFileCfg(
        asset_path="description/robots/x2_t2/urdf/x2_t2_jw_collision_kungfu.urdf",
        fix_base=False,
        self_collision=True,
        activate_contact_sensors=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,  # Will be overridden by actuators
                damping=0.0,    # Will be overridden by actuators
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            "left_hip_pitch_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.4,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.4,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "waist_roll_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_pitch_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    debug_vis=False,
)


@configclass
class NeuralWBCEnvCfgX2T2(NeuralWBCEnvCfg):
    # General parameters:
    action_space = 23
    observation_space = 1073  # Calculated: 27 tracked bodies * (3+6+3+3+3+6+3+3+3+6) + 23 actions = 1073
    state_space = 1164  # Calculated: teacher_obs (1073) + privileged_obs (91) = 1164

    # Distillation parameters:
    single_history_dim = 75  # 23 joints * 3 + 6 for base = 75
    observation_history_length = 25

    # Mask setup for an OH2O specialist policy as default:
    distill_mask_sparsity_randomization_enabled = False
    distill_mask_modes = {"omnih2o": DISTILL_MASK_MODES_ALL["omnih2o"]}

    # Robot geometry / actuation parameters:
    actuators = {
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit={
                ".*_hip_pitch_joint": 120.0,
                ".*_hip_roll_joint": 120.0,
                ".*_hip_yaw_joint": 120.0,
                ".*_knee_joint": 120.0,
                ".*_ankle_pitch_joint": 36.0,
                ".*_ankle_roll_joint": 24.0,
            },
            velocity_limit={
                ".*_hip_pitch_joint": 11.9,
                ".*_hip_roll_joint": 11.9,
                ".*_hip_yaw_joint": 11.9,
                ".*_knee_joint": 11.9,
                ".*_ankle_pitch_joint": 13.0,
                ".*_ankle_roll_joint": 19.7,
            },
            stiffness=0, # Will be overridden by cfg.stiffness
            damping=0,
        ),
        "waist": IdealPDActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"],
            effort_limit={
                "waist_yaw_joint": 120.0,
                "waist_pitch_joint": 48.0,
                "waist_roll_joint": 48.0,
            },
            velocity_limit={
                "waist_yaw_joint": 11.9,
                "waist_pitch_joint": 13.0,
                "waist_roll_joint": 13.0,
            },
            stiffness=0,
            damping=0,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_pitch_joint"],
            effort_limit={
                ".*_shoulder_pitch_joint": 36.0,
                ".*_shoulder_roll_joint": 36.0,
                ".*_shoulder_yaw_joint": 24.0,
                ".*_elbow_pitch_joint": 24.0,
            },
            velocity_limit={
                ".*_shoulder_pitch_joint": 13.0,
                ".*_shoulder_roll_joint": 13.0,
                ".*_shoulder_yaw_joint": 15.0,
                ".*_elbow_pitch_joint": 15.0,
            },
            stiffness=0,
            damping=0,
        ),
    }

    robot: ArticulationCfg = X2T2_CFG.replace(prim_path="/World/envs/env_.*/Robot", actuators=actuators)

    body_names = [
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "waist_yaw_link",
        "waist_pitch_link",
        "waist_roll_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_pitch_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_pitch_link",
    ]

    # Joint names by the order in the MJCF model.
    joint_names = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_pitch_joint",
        "waist_roll_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
    ]

    # Lower and upper body joint ids in the MJCF model.
    # X2T2 has 12 lower body joints (6 per leg) + 3 waist joints + 8 arm joints = 23 total
    lower_body_joint_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # hips, knees, ankles
    upper_body_joint_ids = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # waist, shoulders, elbows

    base_name = "pelvis"
    root_id = body_names.index(base_name)

    feet_name = ".*_ankle_roll_link"

    extend_body_parent_names = ["left_elbow_pitch_link", "right_elbow_pitch_link", "waist_roll_link"]
    extend_body_names = ["left_hand_link", "right_hand_link", "head_link"]
    extend_body_pos = torch.tensor([[0.0, 0.0, -0.21], [0.0, 0.0, -0.21], [0.0, 0.0, 0.4]])

    # These are the bodies that are tracked by the teacher. They may also contain the extended
    # bodies.
    tracked_body_names = [
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "waist_yaw_link",
        "waist_pitch_link",
        "waist_roll_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_pitch_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_pitch_link",
        "left_hand_link",
        "right_hand_link",
        "head_link",
    ]

    # control parameters
    stiffness = {
        "left_hip_pitch_joint": 160.0,
        "left_hip_roll_joint": 160.0,
        "left_hip_yaw_joint": 160.0,
        "left_knee_joint": 220.0,
        "left_ankle_pitch_joint": 80.0,
        "left_ankle_roll_joint": 80.0,
        "right_hip_pitch_joint": 160.0,
        "right_hip_roll_joint": 160.0,
        "right_hip_yaw_joint": 160.0,
        "right_knee_joint": 220.0,
        "right_ankle_pitch_joint": 80.0,
        "right_ankle_roll_joint": 80.0,
        "waist_yaw_joint": 200.0,
        "waist_pitch_joint": 120.0,
        "waist_roll_joint": 120.0,
        "left_shoulder_pitch_joint": 120.0,
        "left_shoulder_roll_joint": 120.0,
        "left_shoulder_yaw_joint": 120.0,
        "left_elbow_pitch_joint": 120.0,
        "right_shoulder_pitch_joint": 120.0,
        "right_shoulder_roll_joint": 120.0,
        "right_shoulder_yaw_joint": 120.0,
        "right_elbow_pitch_joint": 120.0,
    }

    damping = {
        "left_hip_pitch_joint": 2.0,
        "left_hip_roll_joint": 2.0,
        "left_hip_yaw_joint": 2.0,
        "left_knee_joint": 4.0,
        "left_ankle_pitch_joint": 2.0,
        "left_ankle_roll_joint": 2.0,
        "right_hip_pitch_joint": 2.0,
        "right_hip_roll_joint": 2.0,
        "right_hip_yaw_joint": 2.0,
        "right_knee_joint": 4.0,
        "right_ankle_pitch_joint": 2.0,
        "right_ankle_roll_joint": 2.0,
        "waist_yaw_joint": 5.0,
        "waist_pitch_joint": 5.0,
        "waist_roll_joint": 5.0,
        "left_shoulder_pitch_joint": 2.0,
        "left_shoulder_roll_joint": 2.0,
        "left_shoulder_yaw_joint": 2.0,
        "left_elbow_pitch_joint": 2.0,
        "right_shoulder_pitch_joint": 2.0,
        "right_shoulder_roll_joint": 2.0,
        "right_shoulder_yaw_joint": 2.0,
        "right_elbow_pitch_joint": 2.0,
    }

    mass_randomized_body_names = [
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "waist_yaw_link",
        "waist_pitch_link",
        "waist_roll_link",
    ]

    undesired_contact_body_names = [
        "pelvis",
        ".*_yaw_link",
        ".*_roll_link",
        ".*_pitch_link",
        ".*_knee_link",
    ]

    # Add a height scanner to the pelvis to detect the height of the terrain mesh
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        # Apply a grid pattern that is smaller than the resolution to only return one height value.
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.05, 0.05]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # Use X2T2-specific reward configuration
    rewards = NeuralWBCRewardCfgX2T2()

    def __post_init__(self):
        super().__post_init__()

        self.reference_motion_manager.motion_path = "neural_wbc/data/data/x2_t2_23dof_retarget_motion_dynamic_adjust" # Should be overridden by command line argument
        self.reference_motion_manager.skeleton_path = "description/robots/x2_t2/mjcf_mc/x2_mc_wts_kungfu.xml"

        if self.terrain.terrain_generator == HARD_ROUGH_TERRAINS_CFG:
            self.events.update_curriculum.params["penalty_level_up_threshold"] = 125

        if self.mode == NeuralWBCModes.TRAIN:
            self.episode_length_s = 20.0
            self.max_ref_motion_dist = 0.5
            self.events = NeuralWBCTrainEventCfg()
            self.events.reset_robot_rigid_body_mass.params["asset_cfg"].body_names = self.mass_randomized_body_names
            self.events.reset_robot_base_com.params["asset_cfg"].body_names = "pelvis"
        elif self.mode == NeuralWBCModes.DISTILL:
            self.max_ref_motion_dist = 0.5
            self.events = NeuralWBCTrainEventCfg()
            self.events.reset_robot_rigid_body_mass.params["asset_cfg"].body_names = self.mass_randomized_body_names
            self.events.reset_robot_base_com.params["asset_cfg"].body_names = "pelvis"
            self.add_policy_obs_noise = False
            self.reset_mask = True
            # Do not reset mask when there is only one mode.
            num_regions = len(self.distill_mask_modes)
            if num_regions == 1:
                region_modes = list(self.distill_mask_modes.values())[0]
                if len(region_modes) == 1:
                    self.reset_mask = False
        elif self.mode == NeuralWBCModes.TEST:
            self.terrain = flat_terrain
            self.events = NeuralWBCPlayEventCfg()
            self.ctrl_delay_step_range = (2, 2)
            self.max_ref_motion_dist = 0.5
            self.add_policy_obs_noise = False
            self.resample_motions = False
            self.distill_mask_sparsity_randomization_enabled = False
            self.distill_mask_modes = {"omnih2o": DISTILL_MASK_MODES_ALL["omnih2o"]}
        elif self.mode == NeuralWBCModes.DISTILL_TEST:
            self.terrain = flat_terrain
            self.events = NeuralWBCPlayEventCfg()
            self.distill_teleop_selected_keypoints_names = []
            self.ctrl_delay_step_range = (2, 2)
            self.max_ref_motion_dist = 0.5
            self.default_rfi_lim = 0.0
            self.add_policy_obs_noise = False
            self.resample_motions = False
            self.distill_mask_sparsity_randomization_enabled = False
            self.distill_mask_modes = {"omnih2o": DISTILL_MASK_MODES_ALL["omnih2o"]}
        else:
            raise ValueError(f"Unsupported mode {self.mode}")
