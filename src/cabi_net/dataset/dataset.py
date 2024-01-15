# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Standard Library
import os
import time
from collections import OrderedDict

# Third Party
import numpy as np
import torch
import trimesh
import trimesh.transformations as tra
from autolab_core import CameraIntrinsics
from torch.autograd import Variable
from torch.utils import data
from urdfpy import URDF

# NVIDIA
from cabi_net.dataset.renderer import SceneRenderer
from cabi_net.dataset.scene import SceneManager
from cabi_net.errors import BadPointCloudRendering
from cabi_net.utils import (
    compute_camera_pose,
    convert_camera_convention
)


class CollisionDataset(data.IterableDataset):
    def __init__(
        self,
        meshes,
        batch_size,
        query_size,
        intrinsics,
        extrinsics,
        bounds,
        n_obj_points,
        n_scene_points,
        rotations=True,
        trajectories=0,
        num_eih_cameras=5,
        num_scene_cameras=2,
        obj_fully_obs=False,
        franka_target_pct=0.0,
        grasps_pct=0.0,
        env_type=0,
        batch_multiplier=1,
        test=False,
        balanced_positive_ratio=-1,
        target_query_size=2048,
        floating_mpc=False,
    ):
        self.meshes = meshes
        self.batch_size = batch_size
        self.query_size = query_size

        self._env_type = env_type
        self.cam_intr = CameraIntrinsics(**intrinsics)
        self.extrinsics = extrinsics
        self.bounds = np.array(bounds)

        self.n_obj_points = n_obj_points
        self.n_scene_points = n_scene_points
        self.grasps_pct = grasps_pct
        self._sampled_grasp = False
        self.rotations = rotations
        self.trajectories = trajectories
        self._num_eih_cameras = num_eih_cameras
        self._num_scene_cameras = num_scene_cameras
        self.obj_fully_obs = obj_fully_obs

        assert (
            franka_target_pct <= 0.0
        ), f"franka_target_pct should be 0 but {franka_target_pct} was passed in"

        self.franka_target_pct = franka_target_pct

        self.obj_info = np.load(
            os.path.join(self.meshes, "object_info.npy"),
            allow_pickle=True,
        ).item()

        # Variables for book keeping scenes
        self._cached_scene = None
        self._batch_multiplier = batch_multiplier
        self._batch_multiplier_idx = 0
        self._test = test
        self._floating_mpc = floating_mpc

        from cabi_net.utils import get_object_dataset_split
        self.obj_info_split = get_object_dataset_split(self.meshes, test=self._test)

        self._balanced_positive_ratio = balanced_positive_ratio
        self._balanced_batch_mode = True if self._balanced_positive_ratio >= 0 else False
        self._target_query_size = target_query_size
        # Oversampling is needed before rejection sampling, since free-space queries are skewed to be collision-free 
        self.query_size = 5 * self.query_size if self._balanced_batch_mode else self.query_size

    def __iter__(self):
        while True:
            # Generates object and object point cloud
            (
                obj_points,
                obj_centroid,
                obj_mesh,
                obj_pose,
                obj_type,
                camera_pose,
                _,
            ) = self.get_obj()

            # Generates scene and scene point cloud
            try:
                (
                    _,
                    scene_points,
                    scene_points_camera_idx,
                    scene_manager,
                    camera_poses,
                    _,
                    _,
                    _,
                ) = self.get_scene(camera_pose=camera_pose)

            except BadPointCloudRendering:
                continue
            np.random.seed()

            # Sends object in a linear trajectory in the scene, and annotates collision info
            trans, rots, colls = self.get_colls(
                scene_manager,
                scene_points[0],
                obj_mesh,
                obj_pose,
                obj_centroid,
                obj_type,
            )
            del scene_manager

            if trans is None:
                continue

            yield scene_points, scene_points_camera_idx, obj_points, obj_pose, trans, rots, colls, obj_type, camera_poses, None

    def get_scene(self, low=10, high=20, camera_pose=None, scene_dir=None):
        """Creates scenes and renders point clouds."""
        time_log = OrderedDict()
        t0 = time.time()
        camera_poses = []

        num_objs = np.random.randint(low, high)

        if self._batch_multiplier_idx % self._batch_multiplier == 0:
            if hasattr(self, "_cached_scene"):
                del self._cached_scene
            scene_manager, scene_renderer = self._create_scene()
            scene_data = scene_manager.arrange_scene(num_objs, scene_dir=scene_dir)
            self._cached_scene = scene_data
        else:
            if self._cached_scene is None:
                raise ValueError("Scene cache should not be empty")

            scene_data = self._cached_scene
            scene_manager, scene_renderer = self._create_scene()
            _ = scene_manager.arrange_scene(num_objs, scene_dir=scene_dir, data=scene_data)

        extrinsics = scene_data["extrinsics"]

        self._batch_multiplier_idx += 1
        time_log["get_scene:arrange_scene"] = time.time() - t0
        t0 = time.time()

        points_batch = np.zeros(
            (self.batch_size, self.n_scene_points, 3),
            dtype=np.float32,
        )

        points_batch_camera_idx = np.zeros(
            (self.batch_size, self.n_scene_points),
            dtype=np.float32,
        )

        for i in range(self.batch_size):
            if camera_pose is None:
                camera_pose = self.sample_camera_pose(extrinsics)
            scene_renderer.set_camera_pose(camera_pose)
            scene_mesh = None

            scene_mesh_wo_walls = scene_data["scene_mesh_wo_walls"]
            scene_mesh = scene_data["scene_mesh"]

            scene_metadata = scene_data["scene_metadata"]

            points = []
            points_camera_idx = []

            obj_ids = scene_metadata["obj_ids"]

            assert (self._num_eih_cameras + self._num_scene_cameras) > 0

            assert self._num_eih_cameras == 1

            if len(obj_ids) == 0:
                # Pull from support information
                obj_ids = scene_metadata["support_ids_with_placement"]
                obj_id = np.random.choice(obj_ids)
                obj_id = scene_metadata["support_ids"][obj_id]
            else:
                obj_id = np.random.choice(obj_ids)

            obj_pose = np.eye(4)
            look_at_point_eih = obj_pose[:3, 3]

            camera_poses = sample_camera_poses_batch(
                look_at_point_eih,
                extrinsics["azimuth"],
                extrinsics["elevation"],
                extrinsics["radius"],
                self._num_eih_cameras,
            )

            for j, camera_pose in enumerate(camera_poses):
                scene_renderer.set_camera_pose(camera_pose)
                az_points = scene_renderer.render_points(normals=False, segmentation=False)
                points.append(az_points)

                camera_idx = np.ones(len(az_points)) * j
                points_camera_idx.append(camera_idx)
            points = np.vstack(points)
            points_camera_idx = np.hstack(points_camera_idx)

            points_all = points.copy()
            mask = (points[:, :3] > self.bounds[0] + 1e-4).all(axis=1)

            points = points[mask]
            points_camera_idx = points_camera_idx[mask]

            mask = (points[:, :3] < self.bounds[1] - 1e-4).all(axis=1)
            points = points[mask]
            points_camera_idx = points_camera_idx[mask]

            try:
                pt_inds = np.random.choice(points.shape[0], size=self.n_scene_points)
                sample_points = points[pt_inds]
                sample_points_camera_idx = points_camera_idx[pt_inds]
            except Exception as e:
                raise BadPointCloudRendering()

            points_batch[i] = sample_points
            points_batch_camera_idx[i] = sample_points_camera_idx

        time_log["get_scene:render_scene"] = time.time() - t0

        return (
            points_all,
            points_batch,
            points_batch_camera_idx,
            scene_manager,
            camera_poses,
            scene_mesh,
            scene_mesh_wo_walls,
            time_log,
        )

    def get_obj(self):
        """Samples object model given batch."""
        time_log = OrderedDict()
        t0 = time.time()
        obj_scene, obj_renderer = self._create_scene()
        time_log["get_obj:create_scene"] = time.time() - t0

        t0 = time.time()
        camera_pose = self.sample_camera_pose(self.extrinsics)
        obj_renderer.set_camera_pose(camera_pose)
        points = np.array([])
        while not points.any():
            obj_scene.reset()

            self._sampled_grasp = np.random.random() < self.grasps_pct

            cat = (
                "franka"
                if np.random.random() < self.franka_target_pct or self._sampled_grasp
                else None
            )

            obj = "franka~gripper_1.0" if self._sampled_grasp else None
            obj_mesh, obj_info = obj_scene.sample_obj(cat=cat, obj=obj)

            time_log["get_obj:load_obj"] = time.time() - t0

            if obj_mesh is None:
                continue
            stps = obj_info["stps"]
            probs = obj_info["probs"]
            pose = stps[np.random.choice(len(stps), p=probs)].copy()
            z_rot = tra.rotation_matrix(2 * np.pi * np.random.rand(), [0, 0, 1], point=pose[:3, 3])
            pose = z_rot @ pose

            t0 = time.time()
            if self.obj_fully_obs or cat == "franka":
                mesh_tf = obj_mesh.copy()
                mesh_scale = 1.0
                mesh_tf.apply_scale(mesh_scale)
                mesh_tf.apply_transform(pose)
                samples, face_inds = trimesh.sample.sample_surface(mesh_tf, self.n_obj_points)
                points = samples.astype(np.float32)
            else:
                obj_renderer.add_object("obj", obj_mesh, pose)
                points = obj_renderer.render_points(normals=False, segmentation=False)

            time_log["get_obj:render_obj"] = time.time() - t0
        pt_inds = np.random.choice(
            points.shape[0], size=self.n_obj_points, replace=(not self.obj_fully_obs)
        )
        points_batch = np.repeat(points[None, pt_inds], self.batch_size, axis=0)
        points_center = np.mean(points_batch[0, :, :3], axis=0)
        del obj_scene

        return (
            points_batch,
            points_center,
            obj_mesh,
            pose,
            -1,
            camera_pose,
            time_log,
        )

    def get_colls(
        self,
        scene_manager,
        scene_points,
        obj,
        obj_pose,
        obj_centroid,
        obj_type,
    ):
        """Returns collision data given the scene and object."""
        if self._sampled_grasp:
            obj_names = list(scene_manager.objs)
            np.random.shuffle(obj_names)
            poses = []
            for on in obj_names:
                if "grasps" in scene_manager.objs[on]:
                    gs = scene_manager.objs[on]["grasps"]["pos"]
                    poses.append(
                        scene_manager.objs[on]["pose"] @ gs @ tra.euler_matrix(0, 0, -np.pi / 2)
                    )
            if len(poses) > 0:
                poses = np.concatenate(poses, axis=0)
                poses = poses[poses[:, 2, 2] < 0]
            if len(poses) == 0:  # no valid grasps found
                trans_start, trans_end = np.random.uniform(
                    self.bounds[0],
                    self.bounds[1],
                    size=(2, self.trajectories, len(self.bounds[0])),
                )
                trans = (
                    np.linspace(trans_start, trans_end, self.query_size // self.trajectories)
                    .transpose(1, 0, 2)
                    .reshape(self.query_size, 3)
                )
                self._sampled_grasp = False
            else:
                # Sample query size and perturb
                poses = poses[np.random.randint(len(poses), size=self.query_size)]
                poses[:, :3, 3] += np.random.normal(scale=0.0025, size=(len(poses), 3))
                trans = poses[:, :3, 3]
        elif self.trajectories == 0:
            sample_scene_pts = scene_points[
                np.random.choice(len(scene_points), size=self.query_size), :3
            ]
            trans = (
                np.random.uniform(
                    [-0.1, -0.1, -0.1],
                    [0.1, 0.1, 0.1],
                    size=(self.query_size, 3),
                ).astype(np.float32)
                + sample_scene_pts
            )
        else:
            trans_start, trans_end = np.random.uniform(
                self.bounds[0],
                self.bounds[1],
                size=(2, self.trajectories, len(self.bounds[0])),
            )

            trans = (
                np.linspace(trans_start, trans_end, self.query_size // self.trajectories)
                .transpose(1, 0, 2)
                .reshape(self.query_size, 3)
            )

        # Make sure queries are within the bounds
        mask_within_bounds = np.all((trans > self.bounds[0]), 1)
        mask_within_bounds = np.logical_and(mask_within_bounds, np.all((trans < self.bounds[1]), 1))

        mesh_trans = trans - (obj_centroid - obj_pose[:3, 3])
        mesh_tfs = np.repeat(np.eye(4)[None, ...], self.query_size, axis=0)
        mesh_tfs[:, :3, 3] = mesh_trans

        pose_mode = self._sampled_grasp
        if self.rotations:
            if pose_mode:
                rots = poses[:, :3, :2].transpose(0, 2, 1)
            elif self.trajectories == 0:
                rots = np.random.randn(self.query_size, 2, 3).astype(np.float32)
            else:
                rots = np.random.randn(2 * self.trajectories, 2, 3).astype(np.float32)
            b1 = rots[:, 0] / np.linalg.norm(rots[:, 0], axis=-1, keepdims=True)
            b2 = rots[:, 1] - np.einsum("ij,ij->i", b1, rots[:, 1])[:, None] * b1
            b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
            b3 = np.cross(b1, b2)
            rot_mats = np.stack((b1, b2, b3), axis=-1)
            if self.trajectories > 0 and not pose_mode:
                step = self.query_size // self.trajectories
                for i in range(self.trajectories):
                    quats = [tra.quaternion_from_matrix(rm) for rm in rot_mats[2 * i : 2 * (i + 1)]]
                    d = np.dot(*quats)
                    if d < 0.0:
                        d = -d
                        np.negative(quats[1], quats[1])
                    ang = np.arccos(d)
                    t = np.linspace(0, 1, step, endpoint=True)
                    quats_slerp = quats[0][None, :] * np.sin((1.0 - t) * ang)[:, None] / np.sin(
                        ang
                    ) + quats[1][None, :] * np.sin(t * ang)[:, None] / np.sin(ang)
                    mesh_tfs[i * step : (i + 1) * step, :3, :3] = tra.quaternion_matrix(
                        quats_slerp
                    )[:, :3, :3]
            else:
                mesh_tfs[:, :3, :3] = rot_mats[:, :3, :3]
            rots = mesh_tfs[:, :3, :2].transpose(0, 2, 1)
        else:
            rots = np.repeat(
                np.eye(3, dtype=np.float32)[None, :, :2], self.query_size, axis=0
            ).transpose(0, 2, 1)

        new_obj_poses = mesh_tfs @ obj_pose
        colls = np.zeros(self.query_size, dtype=np.bool)
        scene_manager._collision_manager.add_object("query_obj", obj)
        for i in range(self.query_size):
            scene_manager._collision_manager.set_transform("query_obj", new_obj_poses[i])
            colls[i] = scene_manager.collides()
        scene_manager._collision_manager.remove_object("query_obj")

        # From shape (self.query_size, 2, 3) -> (self.query_size, 6)
        rots = rots.reshape(self.query_size, -1)

        trans = trans[mask_within_bounds]
        rots = rots[mask_within_bounds]
        colls = colls[mask_within_bounds]

        if self._balanced_batch_mode:
            assert trans.shape[0] >= self._target_query_size * 2.0
            num_pos_queries = int(self._target_query_size * self._balanced_positive_ratio)
            num_neg_queries = int(self._target_query_size) - num_pos_queries
            assert num_neg_queries + num_pos_queries == self._target_query_size
            pos_idx = np.where(colls == 1)[0]
            neg_idx = np.where(colls == 0)[0]

            assert len(pos_idx) + len(neg_idx) == colls.shape[0]

            resampled_pos_idx = np.random.randint(low=0, high=len(pos_idx), size=num_pos_queries)
            pos_mask = pos_idx[resampled_pos_idx]

            resampled_neg_idx = np.random.randint(low=0, high=len(neg_idx), size=num_neg_queries)
            neg_mask = neg_idx[resampled_neg_idx]

            mask = np.append(pos_mask, neg_mask)

            trans = trans[mask]
            rots = rots[mask]
            colls = colls[mask]

        return trans, rots, colls

    def _create_scene(self):
        r = SceneRenderer()
        r.create_camera(self.cam_intr, znear=0.04, zfar=100)
        s = SceneManager(
            r,
            self.meshes,
            self._env_type,
            self.obj_info,
            self.obj_info_split,
            test=self._test,
            floating_mpc=self._floating_mpc,
        )
        return s, r

    def sample_camera_pose(self, extrinsics, mean=False):
        if mean:
            az = np.mean(extrinsics["azimuth"])
            elev = np.mean(extrinsics["elevation"])
            radius = np.mean(extrinsics["radius"])
        else:
            az = np.random.uniform(*extrinsics["azimuth"])
            elev = np.random.uniform(*extrinsics["elevation"])
            radius = np.random.uniform(*extrinsics["radius"])

        sample_pose, _ = compute_camera_pose(extrinsics["target"], radius, az, elev)

        return sample_pose


def process_batch(data):
    tmp = []
    for elem in data:
        if type(elem) == list:
            if type(elem[0]) == trimesh.Trimesh:
                tmp.append(elem)
                continue
            if type(elem[0]) != np.ndarray:
                elem = [e.numpy() for e in elem]
            camera_poses = convert_camera_convention(elem)
            tmp.append(camera_poses)
        elif type(elem) == trimesh.Trimesh:
            tmp.append(elem)
        elif type(elem) == dict:
            tmp.append(elem)
        elif elem is None:
            tmp.append(elem)
        elif type(elem) is int:
            tmp.append(elem)
        elif type(elem) is np.ndarray:
            tmp.append(elem)
        else:
            tmp.append(Variable(elem.float().cuda()))
    return tmp


def sample_camera_poses_batch(
    look_at_point_eih=[0, 0, 0],
    azimuth_range=[-0.2, 0.2],
    elevation_range=[0.6, 1.0],
    radius_range=[1.5, 2.0],
    num_eih_cameras=1,
):
    """Samples 4x4 camera poses given range of azimuth, elevation and radius."""
    camera_poses = []

    assert len(look_at_point_eih) == 3
    assert len(azimuth_range) == 2
    assert len(elevation_range) == 2
    assert len(radius_range) == 2
    assert num_eih_cameras > 0

    # Add EIH (Eye-in-hand) cameras
    for _ in range(num_eih_cameras):
        target = look_at_point_eih + np.random.uniform(-0.05, 0.05, 3)
        radius = np.random.uniform(*radius_range)
        elevation = np.random.uniform(*elevation_range)
        azimuth = np.random.uniform(*azimuth_range)
        sample_pose, _ = compute_camera_pose(target, radius, azimuth, elevation)
        camera_poses.append(sample_pose)

    return camera_poses


def sample_points(xyz, num_points):
    """Downsamples pointcloud given max points."""
    num_replica = num_points // xyz.shape[0]
    num_remain = num_points % xyz.shape[0]
    pt_idx = torch.randperm(xyz.shape[0])
    pt_idx = torch.cat([pt_idx for _ in range(num_replica)] + [pt_idx[:num_remain]])
    return pt_idx
