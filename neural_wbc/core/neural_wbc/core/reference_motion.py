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
import numpy as np
import torch
from dataclasses import dataclass
from typing import List

# Import motion libraries for different robot embodiments
from phc.utils.motion_lib_h1 import MotionLibH1
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree

# Import X2T2 motion library
import yaml
from pathlib import Path

from humanoidverse.utils.motion_lib.motion_lib_robot_WTS import MotionLibRobotWTS
from humanoidverse.utils.motion_lib.skeleton import SkeletonTree as HumanoidverseSkeletonTree
X2T2_AVAILABLE = True



class ReferenceMotionState:
    """Encapsulates selected fields from a reference motion."""

    def __init__(
        self,
        ref_motion: dict,
        body_ids: List[int] | None = None,
        joint_ids: List[int] | None = None,
    ):
        """Represents the state of a reference motion frame.

        Args:
            ref_motion (dict): Reference motion data at a specific point in time, loaded from the dataset.
            body_ids (List[int] | None, optional): Desired ordering of the bodies. Defaults to None.
            joint_ids (List[int] | None, optional): Desired ordering of the joints. Defaults to None.
        """

        if body_ids is None:
            num_bodies = ref_motion["rg_pos"].shape[1]
            body_ids = list(range(0, num_bodies))

        if joint_ids is None:
            num_joints = ref_motion["dof_pos"].shape[1]
            joint_ids = list(range(0, num_joints))

        # Root
        self.root_pos = ref_motion["root_pos"]
        self.root_rot = ref_motion["root_rot"]
        self.root_lin_vel = ref_motion["root_vel"]
        self.root_ang_vel = ref_motion["root_ang_vel"]

        # Links
        self.body_pos = ref_motion["rg_pos"][:, body_ids, :]
        self.body_rot = ref_motion["rb_rot"][:, body_ids, :]  # [num_envs, num_markers, 4]
        self.body_lin_vel = ref_motion["body_vel"][:, body_ids, :]  # [num_envs, num_markers, 3]
        self.body_ang_vel = ref_motion["body_ang_vel"][:, body_ids, :]  # [num_envs, num_markers, 3]

        # Extended links
        self.body_pos_extend = ref_motion["rg_pos_t"]
        self.body_rot_extend = ref_motion["rg_rot_t"]
        self.body_lin_vel_extend = ref_motion["body_vel_t"]
        self.body_ang_vel_extend = ref_motion["body_ang_vel_t"]

        self.joint_pos = ref_motion["dof_pos"][:, joint_ids]
        self.joint_vel = ref_motion["dof_vel"][:, joint_ids]


@dataclass
class ReferenceMotionManagerCfg:
    motion_path: str
    skeleton_path: str

    def __init__(self):
        pass


class ReferenceMotionManager:
    def __init__(
        self,
        cfg: ReferenceMotionManagerCfg,
        device: torch.device,
        num_envs: int,
        random_sample: bool,
        extend_head: bool,
        dt: float,
    ):
        """Initializes a reference motion manager that loads and queries a motion dataset.

        Args:
            cfg (ReferenceMotionManagerCfg): Configuration that specifies dataset and skeleton paths.
            device (torch.device): Device to host tensors on.
            num_envs (int): Number of environments/instances.
            random_sample (bool): Whether to randomly sample the dataset.
            extend_head (bool): Whether to extend the head of the body for specific robots, e.g. H1.
            dt (float): Length of a policy time step, which is the length of a physics time steps multiplied by decimation.
        """
        self._device = device
        self._num_envs = num_envs
        self._dt = dt

        # Determine which motion library to use based on skeleton path
        if "h1" in cfg.skeleton_path.lower() or "h1.xml" in cfg.skeleton_path:
            # Use H1-specific motion library for H1 robot
            self._motion_lib = MotionLibH1(
                motion_file=cfg.motion_path,
                mjcf_file=cfg.skeleton_path,
                device=self._device,
                masterfoot_conifg=None,
                fix_height=False,
                multi_thread=False,
                extend_head=extend_head,
            )
        elif "x2" in cfg.skeleton_path.lower() or "x2_mc" in cfg.skeleton_path:
            # Use X2T2-specific motion library for X2T2 robot
            if not X2T2_AVAILABLE:
                raise ImportError("X2T2 motion library not available")
            
            # Load X2T2 configuration
            config_path = Path(cfg.skeleton_path).parent.parent.parent / "cfg" / "robot" / "x2_t2_23dof.yaml"
            if not config_path.exists():
                # Fallback to default config if specific config not found
                config_path = Path("description/robots/cfg/robot/x2_t2_23dof.yaml")
            
            with open(config_path, 'r') as f:
                motion_lib_cfg = yaml.safe_load(f)
            
            # Convert to EasyDict for compatibility
            from easydict import EasyDict
            motion_lib_cfg = EasyDict(motion_lib_cfg)
            
            # Set motion file path
            motion_lib_cfg.motion_folder = cfg.motion_path
            
            self._motion_lib = MotionLibRobotWTS(
                motion_lib_cfg=motion_lib_cfg,
                num_envs=self._num_envs,
                device=self._device
            )
        else:
            raise ValueError(f"Unsupported skeleton path: {cfg.skeleton_path}. Supported robots: H1, X2T2")

        # Load skeleton trees based on robot type
        if "h1" in cfg.skeleton_path.lower() or "h1.xml" in cfg.skeleton_path:
            self._skeleton_trees = [SkeletonTree.from_mjcf(cfg.skeleton_path)] * self._num_envs
        elif "x2" in cfg.skeleton_path.lower() or "x2_mc" in cfg.skeleton_path:
            # X2T2 uses humanoidverse skeleton tree
            self._skeleton_trees = [HumanoidverseSkeletonTree.from_mjcf(cfg.skeleton_path)] * self._num_envs
        else:
            # Default to original skeleton tree
            self._skeleton_trees = [SkeletonTree.from_mjcf(cfg.skeleton_path)] * self._num_envs

        self._motion_start_times = torch.zeros(
            self._num_envs, dtype=torch.float32, device=self._device, requires_grad=False
        )

        self._motion_ids = torch.arange(self._num_envs).to(self._device)
        # Ensure motion_ids are within valid range
        self._validate_and_fix_motion_ids()

        self.load_motions(random_sample=random_sample, start_idx=0)
        

    def _validate_and_fix_motion_ids(self):
        """Validate and fix motion_ids to ensure they are within valid range."""
        num_unique_motions = self.num_unique_motions
        
        if num_unique_motions == 0:
            raise ValueError("No motions loaded in motion library")
        
        # Check if any motion_ids are out of bounds
        max_valid_id = num_unique_motions - 1
        invalid_mask = self._motion_ids > max_valid_id
        
        if invalid_mask.any():
            print(f"Warning: Some motion_ids are out of bounds. Clamping to valid range [0, {max_valid_id}]")
            # Clamp motion_ids to valid range
            self._motion_ids = torch.clamp(self._motion_ids, 0, max_valid_id)
            
            # If we have fewer unique motions than environments, we need to cycle through motions
            if num_unique_motions < self._num_envs:
                print(f"Warning: Only {num_unique_motions} unique motions available for {self._num_envs} environments. Cycling through motions.")
                self._motion_ids = torch.remainder(self._motion_ids, num_unique_motions)

    @property
    def motion_lib(self):
        """Returns the MotionLib instance that handles loading and querying the motion dataset."""
        return self._motion_lib

    @property
    def motion_start_times(self):
        """Gets the start times of the loaded motions."""
        return self._motion_start_times

    @property
    def motion_len(self):
        """Gets the lengths of the loaded motions."""
        return self._motion_len

    @property
    def num_unique_motions(self):
        """Returns the number of unique motions in the dataset."""
        return getattr(self._motion_lib, '_num_unique_motions', 0)

    @property
    def body_extended_names(self) -> list[str]:
        if isinstance(self._motion_lib, MotionLibRobotWTS):
            # X2T2 motion library uses body_names_augment attribute
            return getattr(self._motion_lib.mesh_parsers, 'body_names_augment', [])
        else:
            # H1 motion library uses model_names attribute
            return getattr(self._motion_lib.mesh_parsers, 'model_names', [])

    def get_motion_num_steps(self):
        """Gets the number of motion steps/frames of the sampled motions."""
        if hasattr(self._motion_lib, 'get_motion_num_steps'):
            return self._motion_lib.get_motion_num_steps()  # type: ignore
        else:
            # Fallback for motion libraries that don't have this method
            motion_lengths = getattr(self._motion_lib, '_motion_lengths', [])
            return len(motion_lengths)

    def load_motions(self, random_sample: bool, start_idx: int):
        """Loads motions from the motion dataset."""
        if isinstance(self._motion_lib, MotionLibRobotWTS):
            # X2T2 motion library has different interface
            self._motion_lib.load_motions(
                start_idx=start_idx,
            )
        else:
            # H1 motion library interface
            self._motion_lib.load_motions(
                skeleton_trees=self._skeleton_trees,
                gender_betas=[torch.zeros(17)] * self._num_envs,
                limb_weights=[np.zeros(10)] * self._num_envs,
                random_sample=random_sample,
                start_idx=start_idx,
            )
        
        self._motion_len = self._motion_lib.get_motion_length(self._motion_ids)
        self.reset_motion_start_times(env_ids=self._motion_ids, sample=False)

    def reset_motion_start_times(self, env_ids: torch.Tensor, sample: bool):
        """Resets the time at which the reference motions start playing."""
        if sample:
            self._motion_start_times[env_ids] = self._motion_lib.sample_time(self._motion_ids[env_ids])
        else:
            self._motion_start_times[env_ids] = 0

    def episodes_exceed_motion_length(
        self, episode_times: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Checks if the reference motion has reached to the end."""
        if env_ids is None:
            return (episode_times + self._motion_start_times) > self._motion_len
        return (episode_times[env_ids] + self._motion_start_times[env_ids]) > self._motion_len[env_ids]

    def get_state_from_motion_lib_cache(
        self,
        episode_length_buf: torch.Tensor,
        terrain_heights: torch.Tensor | None = None,
        offset: torch.Tensor | None = None,
        quaternion_is_xyzw=True,
    ) -> ReferenceMotionState:
        """Query a reference motion frame from motion lib."""
        motion_times = episode_length_buf * self._dt + self._motion_start_times
        
        if isinstance(self._motion_lib, MotionLibRobotWTS):
            # X2T2 motion library interface
            motion_res = self._motion_lib.get_motion_state(
                motion_ids=self._motion_ids, 
                motion_times=motion_times, 
                offset=offset
            )
        else:
            # H1 motion library interface
            motion_res = self._motion_lib.get_motion_state(self._motion_ids, motion_times, offset=offset)

        if terrain_heights is not None:
            delta_height = terrain_heights.clone()
            if offset is not None:
                delta_height -= offset[:, 2].unsqueeze(1)
            motion_res["root_pos"][:, 2] += delta_height.flatten()
            if "rg_pos" in motion_res:
                motion_res["rg_pos"][:, :, 2] += delta_height
            if "rg_pos_t" in motion_res:
                motion_res["rg_pos_t"][:, :, 2] += delta_height

        # Update quaternion convention
        if quaternion_is_xyzw:
            for key, value in motion_res.items():
                if value.shape[-1] == 4:
                    motion_res[key] = value[..., [3, 0, 1, 2]]

        return ReferenceMotionState(motion_res)
