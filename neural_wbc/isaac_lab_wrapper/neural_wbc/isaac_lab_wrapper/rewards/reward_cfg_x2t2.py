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

from isaaclab.utils import configclass

from .reward_cfg import NeuralWBCRewardCfg


@configclass
class NeuralWBCRewardCfgX2T2(NeuralWBCRewardCfg):
    # Reward and penalty scales (inherited from base class)
    
    # Limits for X2T2 robot (23 joints total)
    torque_limits_scale = 0.85
    # The order here follows the order in cfg.joint_names for X2T2
    torque_limits = [
        # Left leg (6 joints)
        120.0,  # left_hip_pitch_joint
        120.0,  # left_hip_roll_joint
        120.0,  # left_hip_yaw_joint
        120.0,  # left_knee_joint
        36.0,   # left_ankle_pitch_joint
        24.0,   # left_ankle_roll_joint
        # Right leg (6 joints)
        120.0,  # right_hip_pitch_joint
        120.0,  # right_hip_roll_joint
        120.0,  # right_hip_yaw_joint
        120.0,  # right_knee_joint
        36.0,   # right_ankle_pitch_joint
        24.0,   # right_ankle_roll_joint
        # Waist (3 joints)
        120.0,  # waist_yaw_joint
        48.0,   # waist_pitch_joint
        48.0,   # waist_roll_joint
        # Left arm (4 joints)
        36.0,   # left_shoulder_pitch_joint
        36.0,   # left_shoulder_roll_joint
        24.0,   # left_shoulder_yaw_joint
        24.0,   # left_elbow_pitch_joint
        # Right arm (4 joints)
        36.0,   # right_shoulder_pitch_joint
        36.0,   # right_shoulder_roll_joint
        24.0,   # right_shoulder_yaw_joint
        24.0,   # right_elbow_pitch_joint
    ]
    
    # Joint pos limits, in the form of (lower_limit, upper_limit)
    joint_pos_limits = [
        # Left leg (6 joints)
        (-2.556, 2.556),   # left_hip_pitch_joint
        (-0.235, 2.967),  # left_hip_roll_joint
        (-1.685, 3.43),   # left_hip_yaw_joint
        (0.0, 2.12),      # left_knee_joint
        (-0.803, 0.453),  # left_ankle_pitch_joint
        (-0.262, 0.262),  # left_ankle_roll_joint
        # Right leg (6 joints)
        (-2.556, 2.556),  # right_hip_pitch_joint
        (-2.906, 0.235),  # right_hip_roll_joint
        (-3.43, 1.685),   # right_hip_yaw_joint
        (0.0, 2.12),      # right_knee_joint
        (-0.803, 0.453),  # right_ankle_pitch_joint
        (-0.262, 0.262),  # right_ankle_roll_joint
        # Waist (3 joints)
        (-3.43, 2.382),   # waist_yaw_joint
        (-0.314, 0.314),  # waist_pitch_joint
        (-0.488, 0.488),  # waist_roll_joint
        # Left arm (4 joints)
        (-2.556, 2.556),  # left_shoulder_pitch_joint
        (-0.061, 2.993),  # left_shoulder_roll_joint
        (-2.556, 2.556),  # left_shoulder_yaw_joint
        (-2.556, 0.0),    # left_elbow_pitch_joint
        # Right arm (4 joints)
        (-2.556, 2.556),  # right_shoulder_pitch_joint
        (-2.993, 0.061),  # right_shoulder_roll_joint
        (-2.556, 2.556),  # right_shoulder_yaw_joint
        (-2.556, 0.0),    # right_elbow_pitch_joint
    ]
    
    joint_vel_limits_scale = 0.85
    joint_vel_limits = [
        # Left leg (6 joints)
        11.9,    # left_hip_pitch_joint
        11.9,    # left_hip_roll_joint
        11.9,    # left_hip_yaw_joint
        11.9,    # left_knee_joint
        13.0,    # left_ankle_pitch_joint
        19.7,    # left_ankle_roll_joint
        # Right leg (6 joints)
        11.9,    # right_hip_pitch_joint
        11.9,    # right_hip_roll_joint
        11.9,    # right_hip_yaw_joint
        11.9,    # right_knee_joint
        13.0,    # right_ankle_pitch_joint
        19.7,    # right_ankle_roll_joint
        # Waist (3 joints)
        11.9,    # waist_yaw_joint
        13.0,    # waist_pitch_joint
        13.0,    # waist_roll_joint
        # Left arm (4 joints)
        13.0,    # left_shoulder_pitch_joint
        13.0,    # left_shoulder_roll_joint
        15.0,    # left_shoulder_yaw_joint
        15.0,    # left_elbow_pitch_joint
        # Right arm (4 joints)
        13.0,    # right_shoulder_pitch_joint
        13.0,    # right_shoulder_roll_joint
        15.0,    # right_shoulder_yaw_joint
        15.0,    # right_elbow_pitch_joint
    ]
