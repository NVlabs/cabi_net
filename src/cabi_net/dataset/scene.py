# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Standard Library
import json
import os

# Third Party
import numpy as np
import trimesh
import trimesh.transformations as tra

# NVIDIA
from cabi_net.geometry_utils import as_trimesh_scene
from cabi_net.utils import get_cnet_model_frame

try:
    # NVIDIA
    from cabi_net.dataset.envs import (
        EnvID,
        convert_env_type_to_id,
        generate_clutter_scene,
        get_camera_distribution,
    )
except:
    pass
# NVIDIA
from cabi_net.errors import InfeasibleSceneError, SceneGenerationError


class SceneManager:
    def __init__(
        self,
        renderer,
        dataset_folder,
        env_type,
        obj_info,
        obj_info_split,
        test=False,
        floating_mpc=False,
    ):
        self._dataset_path = dataset_folder
        self.mesh_info = obj_info["meshes"]
        self.categories = obj_info_split["categories"]

        self._collision_manager = trimesh.collision.CollisionManager()
        self._renderer = renderer

        self.objs = {}

        self._gravity_axis = 2
        self._table_dims = np.array([1.0, 1.6, 0.6])
        self._table_pose = np.eye(4)

        self._env_type = env_type

        self._env_id = convert_env_type_to_id(self._env_type)

        if self._env_id == EnvID.Universal:
            self._env_id = np.random.choice(
                [EnvID.Shelf, EnvID.Cubby, EnvID.Drawer, EnvID.Cabinet, EnvID.Table]
            )

        self.extrinsics = get_camera_distribution(self._env_id)

        self._test = test
        self.floating_mpc = floating_mpc

        if self._env_id == EnvID.Shelf:
            asset_file = (
                "labels/shelf/test_assets.json" if self._test else "labels/shelf/train_assets.json"
            )

        num_robot_poses = 2
        randomize_robot_planar_rotation = False
        limit_robot_placement = True
        distance_above_support = 0.002
        num_objects = np.random.choice([0, 2, 4, 9], p=[0.30, 0.31, 0.30, 0.09])

        use_stable_pose = True if np.random.random() < 0.80 else False
        position_iterator = 0
        robot_urdf = "data/franka_description/franka_panda.urdf"

        obj_asset_bank = []
        mesh_info = obj_info["meshes"]

        for obj in list(mesh_info):
            info = mesh_info[obj]

            obj_dict = {
                "fname": os.path.join(self._dataset_path, info["path"]),
                "scale": (
                    1.0
                ),  # everything in this dataset should be pre-scaled during the pre-processing, hence scale is 1.0
                "stable_poses": info["stps"],
                "stable_pose_probs": info["probs"],
            }
            obj_asset_bank.append(obj_dict)

        self._scene_generation_kwargs = dict(
            obj_asset_bank=obj_asset_bank,
            num_objs=num_objects,
            robot_urdf=robot_urdf,
            position_iterator=position_iterator,
            use_stable_pose=use_stable_pose,
            num_robot_poses=num_robot_poses,
            randomize_robot_planar_rotation=randomize_robot_planar_rotation,
            limit_robot_placement=limit_robot_placement,
            distance_above_support=distance_above_support,
        )

        self._scene_generation_kwargs["env_id"] = self._env_id
        if self._env_id == EnvID.Kitchen:
            self._scene_generation_kwargs["robot_urdf"] = None
            self._scene_generation_kwargs["include_floor"] = False
        elif self._env_id in [EnvID.Drawer, EnvID.Cabinet, EnvID.Cubby, EnvID.Table]:
            pass
        elif self._env_id == EnvID.Shelf:
            asset_list = json.load(open(asset_file, "rb"))
            self.shelf_filename = np.random.choice(asset_list)
            self._scene_generation_kwargs["shelf_filename"] = self.shelf_filename
        else:
            raise NotImplementedError(f"Invalid env {self._env_id}")

        self.create_env_func = generate_clutter_scene

    def collides(self):
        return self._collision_manager.in_collision_internal()

    def min_distance(self, obj_manager):
        return self._collision_manager.min_distance_other(obj_manager)

    def add_object(self, name, mesh, info={}, pose=None, color=None):
        if name in self.objs:
            raise ValueError("Duplicate name: object {} already exists".format(name))

        if pose is None:
            pose = np.eye(4, dtype=np.float32)

        color = np.asarray((0.7, 0.7, 0.7)) if color is None else np.asarray(color)
        mesh.visual.face_colors = np.tile(np.reshape(color, [1, 3]), [mesh.faces.shape[0], 1])
        self.objs[name] = {"mesh": mesh, "pose": pose}
        if "grasps" in info:
            self.objs[name]["grasps"] = info["grasps"]
        self._collision_manager.add_object(
            name,
            mesh,
            transform=pose,
        )
        self._renderer.add_object(name, mesh, pose)

        return True

    def sample_obj(self, cat=None, obj=None):
        if cat is None:
            cat = np.random.choice(list(self.categories.keys()))
        if obj is None:
            obj = np.random.choice(list(self.categories[cat]))
        try:
            mesh_path = os.path.join(self._dataset_path, self.mesh_info[obj]["path"])
            mesh = trimesh.load(mesh_path)
            mesh.metadata["key"] = obj
            info = self.mesh_info[obj]
        except ValueError:
            mesh = None
            info = None
        info["cat"] = cat
        info["obj_name"] = obj
        return mesh, info

    def place_obj(self, obj_id, mesh, info, max_attempts=10):
        lbs, ubs = self.get_table_xy_bounds()
        self.add_object(obj_id, mesh, info=info)
        for _ in range(max_attempts):
            rand_stp = self.random_object_pose(info, lbs, ubs)
            self.set_object_pose(obj_id, rand_stp)
            if not self.collides():
                return True

        self.remove_object(obj_id)
        return False

    def sample_and_place_obj(self, obj_id, max_attempts=10):
        for _ in range(max_attempts):
            obj_mesh, obj_info = self.sample_obj()
            if not obj_mesh:
                continue
            if self.place_obj(obj_id, obj_mesh, obj_info):
                break
            else:
                continue

    def arrange_scene(self, num_objects, max_attempts=10, scene_dir=None, data=None):
        if data is None:
            data = {}

        if data is not None and len(data) != 0:
            scene_mesh = data["scene_mesh"]

            self.add_object(
                name="scene",
                mesh=scene_mesh,
                pose=self._table_pose,
            )
        else:
            try:
                my_scene, metadata = self.create_env_func(**self._scene_generation_kwargs)
            except Exception as e:
                if type(e) == InfeasibleSceneError:
                    raise InfeasibleSceneError()
                raise SceneGenerationError()
            data["scene_metadata"] = metadata

            mesh = my_scene._scene.dump().sum()

            T_robot_to_world = np.array(metadata["robot_poses"][0])
            random_support_id = np.random.choice(metadata["support_ids_with_placement"])
            T_model_to_robot, _ = get_cnet_model_frame(
                metadata,
                T_robot_to_world,
                random_support_id,
                crop_id=2,
                randomize=True,
                randomize_aggressive=self.floating_mpc,
            )

            model_pose = T_robot_to_world @ T_model_to_robot

            T_model_to_world = model_pose
            T_world_to_model = np.linalg.inv(T_model_to_world)
            metadata["T_world_to_model"] = T_world_to_model

            scene_mesh_wo_walls = mesh

            meshes = [
                scene_mesh_wo_walls,
            ]
            transforms = [
                np.eye(4),
            ]

            if "primitives" in metadata:
                for _, info in metadata["primitives"].items():
                    mesh = trimesh.primitives.Box(extents=info["extents"])
                    meshes.append(mesh)
                    transforms.append(info["pose"])

            whole_scene = as_trimesh_scene(meshes, transforms)
            scene_mesh = whole_scene.dump().sum()

            scene_mesh.apply_transform(T_world_to_model)
            scene_mesh_wo_walls.apply_transform(T_world_to_model)

            data["scene_mesh"] = scene_mesh
            data["scene_mesh_wo_walls"] = scene_mesh_wo_walls

            self.add_object(
                name="scene",
                mesh=scene_mesh,
                pose=self._table_pose,
            )

            data["extrinsics"] = self.extrinsics

        data["scene_mesh"] = scene_mesh
        data["extrinsics"] = self.extrinsics
        return data

    def set_object_pose(self, name, pose):
        if name not in self.objs:
            raise ValueError("object {} needs to be added first".format(name))
        self.objs[name]["pose"] = pose
        self._collision_manager.set_transform(
            name,
            pose,
        )
        self._renderer.set_object_pose(name, pose)

    def get_object_pose(self, name):
        if name not in self.objs:
            raise ValueError("object {} needs to be added first".format(name))
        return self.objs[name]["pose"]

    def random_object_pose(self, obj_info, lbs, ubs):
        stps, probs = obj_info["stps"], obj_info["probs"]
        pose = stps[np.random.choice(len(stps), p=probs)].copy()
        pose[:3, 3] += np.random.uniform(lbs, ubs)
        z_rot = tra.rotation_matrix(2 * np.pi * np.random.rand(), [0, 0, 1], point=pose[:3, 3])
        return z_rot @ pose

    def get_table_xy_bounds(self):
        lbs = self._table_pose[:3, 3] - 0.5 * self._table_dims
        ubs = self._table_pose[:3, 3] + 0.5 * self._table_dims
        ubs[self._gravity_axis] = (
            self._table_pose[self._gravity_axis, 3]
            + 0.5 * self._table_dims[self._gravity_axis]
            + 0.001
        )
        lbs[self._gravity_axis] = ubs[self._gravity_axis]
        return lbs, ubs

    def remove_object(self, name):
        if name not in self.objs:
            raise ValueError("object {} needs to be added first".format(name))
        self._collision_manager.remove_object(name)
        self._renderer.remove_object(name)
        del self.objs[name]

    def reset(self):
        self._renderer.reset()

        for name in self.objs:
            self._collision_manager.remove_object(name)

        self.objs = {}
