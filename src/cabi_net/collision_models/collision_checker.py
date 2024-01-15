# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Standard Library
import abc
import itertools
import os
import os.path as osp
from typing import Callable, Union

# Third Party
import numpy as np
import torch
import torch_scatter
import trimesh
import trimesh.transformations as tra
import open3d as o3d

# Standard Library
import multiprocessing as mp
import queue as Queue

# Third Party
import ruamel

yaml = ruamel.yaml.YAML(typ="safe", pure=True)

BATCH_SIZE_MIN_THRESH = 50
BATCH_SIZE_PADDING = 50

class CollisionChecker(abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abc.abstractmethod
    def __call__(self):
        pass


class SceneCollisionChecker(CollisionChecker):
    def __init__(self, robot, logger=None, use_knn=False, **kwargs):
        super().__init__(**kwargs)

        self.robot = robot
        self.device = self.robot.device
        self.logger = logger

        self.cur_scene_pc = None
        self.robot_to_model = None
        self.model_to_robot = None

    def clear_scene(self):
        self.cur_scene_pc = None

    def set_scene(self, obs):
        orig_scene_pc = obs["pc"]

        try:
            label_map = obs["label_map"]
            scene_labels = obs["pc_label"]

            # Remove robot points and ground points plus excluded
            self.scene_pc_mask = np.logical_and(
                scene_labels != label_map["robot"],
                scene_labels != label_map["ground"],
            )
            if "target" in label_map:
                self.scene_pc_mask = np.logical_and(
                    self.scene_pc_mask, scene_labels != label_map["target"]
                )
        except:
            print("label_map and pc_label not set. Using all points in the point cloud")
            self.scene_pc_mask = np.ones(len(orig_scene_pc))
            self.scene_pc_mask = np.array(self.scene_pc_mask, dtype=bool)

        # Transform into robot frame (z up)
        self.camera_pose = obs["camera_pose"]

        self.scene_pc = tra.transform_points(orig_scene_pc, self.camera_pose)

    def set_object(self, obs):
        self.obj_pc = obs["obj_pc"]

    def _aggregate_pc(self, cur_pc, new_pc):
        """Adds new pc observation to cur_pc."""
        # Filter tiny clusters of points and split into vis/occluding
        cam_model_tr = torch.from_numpy(self.camera_pose[:3, 3]).float().to(self.device)
        new_pc = torch.from_numpy(new_pc).float().to(self.device)
        device = new_pc.device
        vis_mask = torch.from_numpy(self.scene_pc_mask).to(self.device)
        dists = torch.norm(new_pc - cam_model_tr, dim=1) ** 2
        dists /= dists.max()
        nearest = torch.zeros_like(vis_mask)
        occ_scene_pc = new_pc[~vis_mask & (nearest < 0.1 * dists)]
        scene_pc = new_pc[vis_mask & (nearest < 0.1 * dists)]

        if cur_pc is not None:

            # Group points by rays; get mapping from points to unique rays
            cur_pc_rays = cur_pc - cam_model_tr
            cur_pc_dists = torch.norm(cur_pc_rays, dim=1, keepdim=True) + 1e-12
            cur_pc_rays /= cur_pc_dists
            occ_pc_rays = occ_scene_pc - cam_model_tr
            occ_pc_dists = torch.norm(occ_pc_rays, dim=1, keepdim=True) + 1e-12
            occ_pc_rays /= occ_pc_dists
            occ_rays = (torch.cat((cur_pc_rays, occ_pc_rays), dim=0) * 50).round().long()
            _, occ_uniq_inv, occ_uniq_counts = torch.unique(
                occ_rays, dim=0, return_inverse=True, return_counts=True
            )

            # Build new point cloud from previous now-occluded points and new pc
            cur_occ_inv = occ_uniq_inv[: len(cur_pc_rays)]
            cur_occ_counts = torch.bincount(cur_occ_inv, minlength=len(occ_uniq_counts))
            mean_occ_dists = torch_scatter.scatter_max(
                occ_pc_dists.squeeze(),
                occ_uniq_inv[-len(occ_pc_rays) :],
                dim_size=occ_uniq_inv.max() + 1,
            )[0]

            occ_mask = (occ_uniq_counts > cur_occ_counts) & (cur_occ_counts > 0)
            occ_pc = cur_pc[
                occ_mask[cur_occ_inv]
                & (cur_pc_dists.squeeze() > mean_occ_dists[cur_occ_inv] + 0.01)
            ]
            return torch.cat((occ_pc, scene_pc), dim=0)
        else:
            return scene_pc

    def _compute_model_tfs(self, obs):
        if "robot_to_model" in obs:
            self.robot_to_model = obs["robot_to_model"]
            self.model_to_robot = obs["model_to_robot"]
            assert len(self.robot_to_model) > 0
            assert len(self.model_to_robot) > 0
        else:
            scene_labels = obs["pc_label"]
            label_map = obs["label_map"]

            # Extract table transform from points
            tab_pts = self.scene_pc[scene_labels == label_map["table"]]
            if len(tab_pts) == 0:
                tab_pts = self.scene_pc[scene_labels == label_map["objs"]]
            tab_height = tab_pts.mean(axis=0)[2]
            tab_tf_2d = trimesh.bounds.oriented_bounds_2D(tab_pts[:, :2])[0]
            tab_tf = np.eye(4)
            tab_tf[:3, :2] = tab_tf_2d[:, :2]
            tab_tf[:3, 3] = np.append(tab_tf_2d[:2, 2], 0.3 - tab_height)

            # Fix "long" side of table by rotating
            self.robot_to_model = tra.euler_matrix(0, 0, np.pi / 2) @ tab_tf
            if self.robot_to_model[0, 0] < 0:
                self.robot_to_model = tra.euler_matrix(0, 0, -np.pi) @ self.robot_to_model
            self.model_to_robot = np.linalg.inv(self.robot_to_model)

        self.robot_to_model = torch.from_numpy(self.robot_to_model).float().to(self.device)
        self.model_to_robot = torch.from_numpy(self.model_to_robot).float().to(self.device)


class FCLProc(mp.Process):
    """
    Used for finding collisions in parallel using FCL.
    """

    def __init__(self, links, output_queue, use_scene_pc=True, logger=None):
        """
        Args:
        output_queue: mp.Queue, the queue that all the output data
            that is computed is added to.
        """
        super().__init__()
        self.links = links
        self.output_queue = output_queue
        self.input_queue = mp.Queue()
        self.use_scene_pc = use_scene_pc
        self.logger = logger

    def _collides(self, link_poses, inds, by_link):
        """Computes collisions."""
        coll = (
            np.zeros((len(self.links), len(inds)), dtype=np.bool)
            if by_link
            else np.zeros(len(inds), dtype=np.bool)
        )

        for k, i in enumerate(inds):
            for link in self.links:
                pose = link_poses[link.name][i].squeeze()
                self.robot_manager.set_transform(link.name, pose)

            coll_q = self.robot_manager.in_collision_other(self.scene_manager, return_names=by_link)

            if by_link:
                for j, link in enumerate(self.links):
                    coll[j, k] = np.any([link.name in pair for pair in coll_q[1]])
            else:
                coll[k] = coll_q
        return coll

    def _object_collides(self, poses, inds):
        """Computes collisions with object."""
        coll = np.zeros(len(inds), dtype=np.bool)
        for k, i in enumerate(inds):
            self.object_manager.set_transform("obj", poses[i])
            coll[k] = self.object_manager.in_collision_other(self.scene_manager)
        return coll

    def _set_scene(self, scene):
        if self.use_scene_pc:
            self.scene_manager = trimesh.collision.CollisionManager()
            self.scene_manager.add_object("scene", scene)
        else:
            if isinstance(scene, trimesh.collision.CollisionManager):
                self.scene_manager = scene
            elif isinstance(scene, trimesh.Trimesh):
                self.scene_manager = trimesh.collision.CollisionManager()
                self.scene_manager.add_object("scene", scene)
            else:
                raise TypeError(
                    f"set_scene called with unfamiliar scene instance type {type(scene)}"
                )

    def _set_object(self, obj):
        if self.use_scene_pc:
            self.object_manager = trimesh.collision.CollisionManager()
            self.object_manager.add_object("obj", obj)

    def run(self):
        self.robot_manager = trimesh.collision.CollisionManager()
        for link in self.links:
            self.robot_manager.add_object(link.name, link.collision_mesh)
        while True:
            try:
                request = self.input_queue.get(timeout=1)
            except Queue.Empty:
                continue
            if request[0] == "set_scene":
                self._set_scene(request[1]),
            elif request[0] == "set_object":
                self._set_object(request[1]),
            elif request[0] == "collides":
                self.output_queue.put(
                    (
                        request[4],
                        self._collides(*request[1:4]),
                    )
                )
            elif request[0] == "obj_collides":
                self.output_queue.put(
                    (
                        request[3],
                        self._object_collides(*request[1:3]),
                    )
                )

    def set_scene(self, scene):
        self.input_queue.put(("set_scene", scene))

    def set_object(self, obj):
        self.input_queue.put(("set_object", obj))

    def collides(self, link_poses, inds, by_link, pind=None):
        self.input_queue.put(("collides", link_poses, inds, by_link, pind))

    def object_collides(self, poses, inds, pind=None):
        self.input_queue.put(("obj_collides", poses, inds, pind))


class FCLMultiSceneCollisionChecker(SceneCollisionChecker):
    def __init__(self, robot, logger=None, n_proc=10, use_scene_pc=True):
        super().__init__(robot=robot, logger=logger, use_knn=False)
        self._n_proc = n_proc
        self._use_scene_pc = use_scene_pc

        self.output_queue = mp.Queue()
        self.coll_procs = []

        if self._n_proc > 0:
            for i in range(self._n_proc):
                self.coll_procs.append(
                    FCLProc(
                        self.robot.mesh_links,
                        self.output_queue,
                        use_scene_pc=self._use_scene_pc,
                        logger=self.logger,
                    )
                )
                self.coll_procs[-1].daemon = True
                self.coll_procs[-1].start()
        else:
            print("Not using multiprocessing for FCLMultiSceneCollisionChecker")
            self.robot_manager = trimesh.collision.CollisionManager()
            for link in self.robot.mesh_links:
                self.robot_manager.add_object(link.name, link.collision_mesh)

    def set_scene(self, obs, scene=None):
        if self._use_scene_pc:
            super().set_scene(obs)
            self.cur_scene_pc = self._aggregate_pc(self.cur_scene_pc, self.scene_pc)
            pc = trimesh.PointCloud(self.cur_scene_pc.cpu().numpy())
            self.scene_mesh = trimesh.voxel.ops.points_to_marching_cubes(pc.vertices, pitch=0.01)
            if self._n_proc > 0:
                for proc in self.coll_procs:
                    proc.set_scene(self.scene_mesh)
            else:
                self.scene_manager = trimesh.collision.CollisionManager()
                self.scene_manager.add_object("scene", self.scene_mesh)
        else:
            if self._n_proc > 0:
                for proc in self.coll_procs:
                    proc.set_scene(scene)
            else:
                self.scene_manager = trimesh.collision.CollisionManager()
                self.scene_manager.add_object("scene", scene)

    def set_object(self, obs, override=False):
        if self.robot_to_model is None:
            self._compute_model_tfs(obs)
        if self._use_scene_pc or override:
            super().set_object(obs)
            obj_pc = self.obj_pc - self.obj_pc.mean()
            pc = trimesh.PointCloud(obj_pc)
            self.obj_mesh = trimesh.voxel.ops.points_to_marching_cubes(pc.vertices, pitch=0.025)
            self.obj_mesh.vertices -= self.obj_mesh.centroid

            if self._n_proc > 0:
                for proc in self.coll_procs:
                    proc.set_object(self.obj_mesh)
            else:
                self.object_manager = trimesh.collision.CollisionManager()
                self.object_manager.add_object("obj", self.obj_mesh)

    def sample_in_bounds(self, num=20000, offset=0.0):
        return (
            torch.rand((num, 3), dtype=torch.float32, device=self.device)
            * (torch.tensor([1.0, 1.6, 0.4], device=self.device) - 2 * offset)
            + torch.tensor([-0.5, -0.8, 0.2], device=self.device)
            + offset
        )

    def check_object_collisions(self, poses, threshold=0.4, thresholded=False):
        """Checks if the object pc collides with the scene.
        Args:
            thresholded (bool): Currently NOT USED, this is just there to preserve the API
        """
        batch_size = len(poses)
        coll = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        transforms = (self.model_to_robot @ poses).cpu().numpy()

        if self._n_proc > 0:
            for i in range(self._n_proc):
                self.coll_procs[i].object_collides(
                    transforms,
                    np.arange(
                        i * batch_size // self._n_proc,
                        (i + 1) * batch_size // self._n_proc,
                    ),
                    pind=i,
                )

            # collect computed iks
            for _ in range(self._n_proc):
                i, proc_coll = self.output_queue.get(True)
                coll[i * batch_size // self._n_proc : (i + 1) * len(transforms) // self._n_proc] = (
                    torch.from_numpy(proc_coll).to(self.device)
                )
        else:
            for k in range(len(transforms)):
                self.object_manager.set_transform("obj", transforms[k])
                coll[k] = self.object_manager.in_collision_other(self.scene_manager)

        return coll

    def __call__(self, q, by_link=False, threshold=None, thresholded=True):
        if q is None or len(q) == 0:
            return torch.zeros(
                (len(self.robot.mesh_links), 0), dtype=torch.bool, device=self.device
            )

        coll = (
            torch.zeros(
                (len(self.robot.mesh_links), len(q)),
                dtype=torch.bool,
                device=self.device,
            )
            if by_link
            else torch.zeros(len(q), dtype=np.bool, device=self.device)
        )
        self.robot.set_joint_cfg(q)
        poses = {k.name: v.cpu().numpy() for k, v in self.robot.link_poses.items()}

        if self._n_proc > 0:
            for i in range(self._n_proc):
                self.coll_procs[i].collides(
                    poses,
                    np.arange(
                        i * len(q) // self._n_proc,
                        (i + 1) * len(q) // self._n_proc,
                    ),
                    by_link,
                    pind=i,
                )

            # collect computed iks
            for _ in range(self._n_proc):
                i, proc_coll = self.output_queue.get(True)
                if by_link:
                    coll[
                        :,
                        i * len(q) // self._n_proc : (i + 1) * len(q) // self._n_proc,
                    ] = torch.from_numpy(proc_coll).to(self.device)
                else:
                    coll[i * len(q) // self._n_proc : (i + 1) * len(q) // self._n_proc] = (
                        torch.from_numpy(proc_coll).to(self.device)
                    )
        else:
            inds = np.arange(len(q))
            for k, i in enumerate(inds):
                for link in self.robot.mesh_links:
                    pose = poses[link.name][i].squeeze()
                    self.robot_manager.set_transform(link.name, pose)

                coll_q = self.robot_manager.in_collision_other(
                    self.scene_manager, return_names=by_link
                )

                if by_link:
                    for j, link in enumerate(self.robot.mesh_links):
                        coll[j, k] = np.any([link.name in pair for pair in coll_q[1]])
                else:
                    coll[k] = coll_q

        return coll, None

def voxelize_pc(pc, voxel_size=None, return_raw_cubes=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    if voxel_size is None:
        voxel_size = round(max(pcd.get_max_bound() - pcd.get_min_bound()) * 0.005, 4)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

    if return_raw_cubes:
        cube_pts = []
        voxels = voxel_grid.get_voxels()
        for v in voxels:
            cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
            cube.translate(v.grid_index, relative=False)
            cube.translate([0.5, 0.5, 0.5], relative=True)
            cube.scale(voxel_size, [0, 0, 0])
            cube.translate(voxel_grid.origin, relative=True)
            f = np.asarray(cube.triangles)
            v = np.asarray(cube.vertices)
            cube = trimesh.Trimesh(v, f)
            cube_pts.append(cube.centroid)
        return cube_pts
    else:
        voxels = voxel_grid.get_voxels()
        vox_mesh = o3d.geometry.TriangleMesh()
        for v in voxels:
            cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
            cube.paint_uniform_color(v.color)
            cube.translate(v.grid_index, relative=False)
            vox_mesh += cube

        vox_mesh.translate([0.5, 0.5, 0.5], relative=True)
        vox_mesh.scale(voxel_size, [0, 0, 0])
        vox_mesh.translate(voxel_grid.origin, relative=True)

        f = np.asarray(vox_mesh.triangles)
        v = np.asarray(vox_mesh.vertices)
        mesh = trimesh.Trimesh(v, f)
        return mesh


class VoxelGridSceneCollisionChecker(FCLMultiSceneCollisionChecker):
    def __init__(self, robot, logger=None, n_proc=1, use_scene_pc=True):
        super().__init__(robot=robot, logger=logger, use_scene_pc=use_scene_pc, n_proc=0)

    def set_scene(self, obs, scene=None):
        assert self._use_scene_pc
        super(FCLMultiSceneCollisionChecker, self).set_scene(obs)
        self.cur_scene_pc = self._aggregate_pc(self.cur_scene_pc, self.scene_pc)

        self.scene_mesh = voxelize_pc(self.cur_scene_pc.cpu().numpy(), voxel_size=0.01)

        if self._n_proc > 0:
            for proc in self.coll_procs:
                proc.set_scene(self.scene_mesh)
        else:
            self.scene_manager = trimesh.collision.CollisionManager()
            self.scene_manager.add_object("scene", self.scene_mesh)

    def set_object(self, obs, override=False):
        if self.robot_to_model is None:
            self._compute_model_tfs(obs)

        assert self._use_scene_pc

        super(FCLMultiSceneCollisionChecker, self).set_object(obs)
        self.obj_mesh = voxelize_pc(self.obj_pc, voxel_size=0.01)

        if self._n_proc > 0:
            for proc in self.coll_procs:
                proc.set_object(self.obj_mesh)
        else:
            self.object_manager = trimesh.collision.CollisionManager()
            self.object_manager.add_object("obj", self.obj_mesh)


class NNCollisionChecker(CollisionChecker):
    def __init__(self, model_path, device=torch.device("cuda:0"), **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.device = device

        with open(osp.join(self.model_path, "train.yaml")) as f:
            self.cfg = yaml.load(f)

    def _load_model(self):
        chk = torch.load(
            os.path.join(self.model_path, "model_best.pth.tar"),
            map_location=self.device,
        )

        self.model.load_state_dict(chk["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

    @abc.abstractmethod
    def __call__(self):
        pass


class NNSceneCollisionChecker(NNCollisionChecker, SceneCollisionChecker):
    def __init__(self, model_path, robot, logger=None, ignore_collision_with_base=False, **kwargs):
        super().__init__(model_path=model_path, robot=robot, logger=logger, **kwargs)
        self.normals = self.cfg["dataset"].get("pc_normals", True)
        self._ignore_collision_with_base = ignore_collision_with_base

        self._setup_model()
        self._load_model()

    def _setup_model(self):
        self.normals = False
        # NVIDIA
        from cabi_net.model.cabinet import CabiNetCollision

        self.model = CabiNetCollision(
            config=self.cfg,
            device=torch.device("cuda"),
        )

    def _setup_robot(self):
        # Get features for robot links
        mesh_links = self.robot.mesh_links
        n_pts = self.cfg["dataset"]["n_obj_points"]
        self.link_pts = np.zeros((len(mesh_links), n_pts, 3 + 3 * self.normals), dtype=np.float32)
        for i, link in enumerate(mesh_links):
            pts, face_inds = link.collision_mesh.sample(n_pts, return_index=True)
            if self.normals:
                l_pts = np.concatenate((pts, link.collision_mesh.face_normals[face_inds]), axis=-1)[
                    None, ...
                ]
            else:
                l_pts = pts[None, ...]
            self.link_pts[i] = l_pts
        with torch.no_grad():
            self.link_features = self.model.get_obj_features(
                torch.from_numpy(self.link_pts).to(self.device)
            )

    def set_scene(self, obs, scene=None, offset=None):
        # Always update the robot_to_model and model_to_robot transforms to
        # being the latest values in the obs.
        self._compute_model_tfs(obs)
        assert offset is None
        super().set_scene(obs)

        if self.cur_scene_pc is not None:
            self.cur_scene_pc = self._aggregate_pc(self.cur_scene_pc, self.scene_pc)
        else:
            self.cur_scene_pc = self._aggregate_pc(None, self.scene_pc)
        model_scene_pc = (
            self.robot_to_model
            @ torch.cat(
                (
                    self.cur_scene_pc,
                    torch.ones((len(self.cur_scene_pc), 1), device=self.device),
                ),
                dim=1,
            ).T
        )
        model_scene_pc = model_scene_pc[:3].T

        if self.model.bounds[0].device != self.device:
            self.model.bounds = [b.to(self.device) for b in self.model.bounds]
            self.model.vox_size = self.model.vox_size.to(self.device)
            self.model.num_voxels = self.model.num_voxels.to(self.device)

        # Clip points to model bounds and feed in for features
        in_bounds = (model_scene_pc[..., :3] > self.model.bounds[0] + 1e-5).all(dim=-1)
        in_bounds &= (model_scene_pc[..., :3] < self.model.bounds[1] - 1e-5).all(dim=-1)

        # This is used in mppi_planner to visualize the exact point cloud input to CollisionNet
        self.model_scene_pc = model_scene_pc[in_bounds]

        self.scene_features = self.model.get_scene_features(
            model_scene_pc[in_bounds].unsqueeze(0)
        ).squeeze(0)

    def set_object(self, obs):
        super().set_object(obs)
        if self.robot_to_model is None:
            self._compute_model_tfs(obs)

        obj_pc = tra.transform_points(
            self.obj_pc,
            self.robot_to_model.cpu().numpy(),
        )

        obj_tensor = torch.from_numpy(obj_pc.astype(np.float32)).to(self.device)
        obj_tensor -= obj_tensor.mean(dim=0)

        self.obj_features = self.model.get_obj_features(obj_tensor.unsqueeze(0)).squeeze(0)

    def sample_in_bounds(self, num=20000, offset=0.0):
        return (
            torch.rand((num, 3), dtype=torch.float32, device=self.device)
            * (-torch.sub(*self.model.bounds) - 2 * offset)
            + self.model.bounds[0]
            + offset
        )

    def validate_obj_pose_goals(
        self, poses_in_model_frame: Union[np.ndarray, torch.Tensor], threshold=0.45
    ):
        """
        Checks whether a set of object poses are collision free. Used for checking
        placements locations.

        Args:
            threshold, bool: Lower the value, the more aggresive you are in
              rejecting queries based on collision score.
        """

        if isinstance(poses_in_model_frame, torch.Tensor):
            poses = poses_in_model_frame
        else:
            poses = torch.from_numpy(poses_in_model_frame).float().to(self.device)

        print(f"VALIDATE obj pose goals, poses {poses.shape}", "red")

        batch_size = poses.shape[0]

        # The following is done to handle some edge cases e.g. only 1 available grasp
        if batch_size <= BATCH_SIZE_MIN_THRESH:
            print("VALIDATE, adding some padding", "red")
            poses = torch.tile(
                poses,
                [
                    BATCH_SIZE_PADDING,
                    1,
                    1,
                ],
            )
        score = self.check_object_collisions(poses, thresholded=False)
        score = 1 - score  # More collision-free is better, so should be higher

        # mask = ~(score > threshold)
        mask = score > threshold

        if batch_size <= BATCH_SIZE_MIN_THRESH:
            mask = mask[:batch_size]

        return mask, score

    def check_object_collisions(self, poses, threshold=0.45, thresholded=True):
        if not thresholded:
            dtype = torch.float32
        else:
            dtype = torch.bool

        translations = poses[:, :3, 3]
        res = torch.ones(len(poses), dtype=dtype, device=self.device)
        in_bounds = (translations > self.model.bounds[0] + 1e-5).all(dim=-1)
        in_bounds &= (translations < self.model.bounds[1] - 1e-5).all(dim=-1)
        if in_bounds.any():
            poses_in = poses[in_bounds]
            tr_in = poses_in[:, :3, 3]
            rots_in = poses_in[:, :3, :2].reshape([len(tr_in), 6])
            # rots = np.repeat(np.eye(4)[:3, :2].flatten()[None, :], len(tr_in), axis=0)
            with torch.no_grad():
                out = self.model.classify_tfs(
                    self.obj_features[None, :],
                    self.scene_features[None, ...],
                    tr_in,
                    rots_in,
                )
                if thresholded:
                    res[in_bounds] = torch.sigmoid(out).squeeze() > threshold
                else:
                    res[in_bounds] = torch.sigmoid(out).squeeze()
        return res

    def __call__(
        self,
        qs,
        by_link=False,
        thresholded=True,
        threshold=0.4,
    ):
        self.robot.set_joint_cfg(qs)
        colls = torch.ones(
            (len(self.link_features), len(qs)),
            dtype=torch.bool if thresholded else torch.float32,
            device=self.device,
        )
        trans = torch.empty(
            (len(self.link_features), len(qs), 3),
            dtype=torch.float32,
            device=self.device,
        )
        rots = torch.empty(
            (len(self.link_features), len(qs), 6),
            dtype=torch.float32,
            device=self.device,
        )

        for i, link in enumerate(self.robot.mesh_links):
            poses_tf = self.robot_to_model @ self.robot.link_poses[link]
            trans[i] = poses_tf[:, :3, 3]
            rots[i] = poses_tf[:, :3, :2].reshape(len(qs), -1)

        # filter translations that are out of bounds
        in_bounds = (trans > self.model.bounds[0] + 1e-5).all(dim=-1)
        in_bounds &= (trans < self.model.bounds[1] - 1e-5).all(dim=-1)

        # The raw collision net predictions (probs) is returned for
        # debugging and visualization purposes
        probs = None
        if in_bounds.any():
            trans[~in_bounds] = 0.0  # Make inputs safe
            with torch.no_grad():
                out = self.model.classify_multi_obj_tfs(
                    self.link_features,
                    self.scene_features,
                    trans,
                    rots,
                )
                probs = torch.sigmoid(out).squeeze(-1)
                res = probs > threshold if thresholded else probs
                colls = res * in_bounds
                probs = probs * in_bounds

                if self._ignore_collision_with_base:
                    colls[0, :] = False

        if thresholded:
            return colls if by_link else colls.any(dim=0), probs
        else:
            return colls if by_link else colls.max(dim=0)[0], probs
