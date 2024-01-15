# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Standard Library
import itertools

# Third Party
import numpy as np
import torch

# Local Folder
from .urdf import TorchURDF


# Thin wrapper around urdfpy class that adds cfg sampling
class Robot:
    def __init__(self, urdf_path, ee_name, device=None):
        if device is None:
            device = torch.device("cpu")
        self._robot = TorchURDF.load(urdf_path, device=device)
        self.dof = len(self.joint_names)
        self._ee_name = ee_name
        self.device = device
        self.min_joints = torch.tensor(
            [j.limit.lower for j in self._robot.joints if j.joint_type != "fixed"],
            device=self.device,
        ).unsqueeze_(0)
        self.max_joints = torch.tensor(
            [j.limit.upper for j in self._robot.joints if j.joint_type != "fixed"],
            device=self.device,
        ).unsqueeze_(0)

        self.link_combos = list(itertools.combinations(self.physical_link_map.keys(), 2))
        for i in range(len(self.links) - 1):
            self.set_allowed_collisions(self.links[i].name, self.links[i + 1].name)
        self.set_allowed_collisions("panda_hand", "panda_rightfinger")
        self.set_allowed_collisions("panda_link7", "panda_hand")
        link_names = list(self.physical_link_map.keys())
        self.physical_link_index = {l: i for i, l in enumerate(link_names)}
        self.link_pair_to_index = {}
        for combo in self.link_combos:
            assert self.physical_link_index[combo[0]] < self.physical_link_index[combo[1]]
            self.link_pair_to_index[
                (self.physical_link_index[combo[0]], self.physical_link_index[combo[1]])
            ] = len(self.link_pair_to_index)

    def set_allowed_collisions(self, l1, l2):
        if (l1, l2) in self.link_combos:
            self.link_combos.remove((l1, l2))
        elif (l2, l1) in self.link_combos:
            self.link_combos.remove((l2, l1))

    @property
    def joint_names(self):
        return [j.name for j in self._robot.joints if j.joint_type != "fixed"]

    @property
    def links(self):
        return self._robot.links

    @property
    def mesh_links(self):
        return [link for link in self._robot.links if link.collision_mesh is not None]

    @property
    def link_map(self):
        return self._robot.link_map

    @property
    def physical_link_map(self):
        return {k: v for k, v in self._robot.link_map.items() if v.collision_mesh is not None}

    @property
    def link_poses(self):
        return self._link_poses

    @property
    def ee_pose(self):
        return self.link_poses[self.link_map[self._ee_name]]

    @property
    def link_pose_dict(self):
        return {linkname: self.link_poses[linkmesh] for linkname, linkmesh in self.link_map.items()}

    def set_joint_cfg(self, q):
        if q is not None:
            if isinstance(q, np.ndarray):
                q = torch.from_numpy(q).float().to(self.device)
            if q.device != self.device:
                q = q.to(self.device)
            if q.ndim == 1:
                q = q.reshape(1, -1)
            if q.ndim > 2:
                raise ValueError("Tensor is wrong shape, must have 2 dims")
            link_poses = self._robot.link_fk_batch(q[:, : self.dof])
            self._link_poses = link_poses
        else:
            self._link_poses = None

    def sample_cfg(self, num=1):
        alpha = torch.rand(num, self.dof, device=self.device)
        return alpha * self.min_joints + (1 - alpha) * self.max_joints

    def set_link_combos(self, link_combos):
        self.link_combos = link_combos

    def get_link_combo_poses(self):
        poses_a = []
        poses_b = []
        indexes_a = []
        indexes_b = []
        for combo in self.link_combos:
            # print(self.link_poses.keys())
            poses_a.append(self.link_pose_dict[combo[0]][:, None])
            poses_b.append(self.link_pose_dict[combo[1]][:, None])
            batch_size = len(poses_a[-1])
            indexes_a.append(
                torch.ones((batch_size, 1), device=self.device).long()
                * self.physical_link_index[combo[0]]
            )
            indexes_b.append(
                torch.ones((batch_size, 1), device=self.device).long()
                * self.physical_link_index[combo[1]]
            )

        poses_a = torch.cat(poses_a, 1)
        poses_b = torch.cat(poses_b, 1)
        indexes_a = torch.cat(indexes_a, 1)
        indexes_b = torch.cat(indexes_b, 1)

        return indexes_a, indexes_b, poses_a, poses_b, poses_b @ torch_inverse(poses_a)


def setup_robot(robot_urdf, robot_eef_frame, device=None):
    if device is None:
        device = 0
        device = torch.device(f"cuda:{device}")

    def _setup_meshcat_robot():
        meshcat_robot = Robot(
            robot_urdf,
            robot_eef_frame,
            device=device,
        )
        return meshcat_robot

    return _setup_meshcat_robot
