# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Standard Library
import itertools
import logging
import os
import time
import datetime
from enum import Enum

# Third Party
import numpy as np
import trimesh
import trimesh.transformations as tra

# NVIDIA
from cabi_net.meshcat_utils import (
    create_visualizer,
    get_color_from_score,
    make_frame,
    visualize_bbox,
    visualize_mesh,
    visualize_pointcloud,
    visualize_robot,
    visualize_triad,
)

_VALID_TYPES = {tuple, list, str, int, float, bool}

log = logging.getLogger("CabiNet")


def convert_to_dict(cfg_node, key_list=[]):
    # Third Party
    from yacs.config import CfgNode

    """Convert a config node to dictionary"""
    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print(
                "Key {} with value {} is not a valid type; valid types: {}".format(
                    ".".join(key_list), type(cfg_node), _VALID_TYPES
                ),
            )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict


def list_iterator(list_of_stuff):
    while True:
        return itertools.cycle(list_of_stuff)


def get_timestamp():
    now = datetime.datetime.now()
    year = "{:02d}".format(now.year)
    month = "{:02d}".format(now.month)
    day = "{:02d}".format(now.day)
    hour = "{:02d}".format(now.hour)
    minute = "{:02d}".format(now.minute)
    day_month_year = "{}-{}-{}-{}-{}".format(year, month, day, hour, minute)
    return day_month_year


def mkdir(dir):
    """
    Creates folder if it doesn't exist
    """
    if not os.path.isdir(dir):
        os.makedirs(dir)

class CropID(Enum):
    """
    Description of each Crop Id mode

    CropNone: No crop mode (default)
    CropWorkspaceRotationAligned: The frame of the model workspace (the rotation that is) is aligned to that of the robot frame
    CropWorkspaceRotated: The frame of the workspace has a planar rotation along the robot frame z-axis.
    """

    CropNone = 0
    CropWorkspaceRotationAligned = 2
    CropWorkspaceRotated = 3

def convert_camera_convention(camera_poses):
    if type(camera_poses) == list:
        camera_poses_tmp = []
        for camera_pose in camera_poses:
            camera_pose_tmp = camera_pose.copy()
            camera_pose_tmp[:, 1:3] *= -1.0
            camera_poses_tmp.append(camera_pose_tmp)
        return camera_poses_tmp
    else:
        if type(camera_poses) != np.ndarray:
            raise ValueError("Incorrect format for camera pose")
        camera_pose_tmp = camera_poses.copy()
        camera_pose_tmp[:, 1:3] *= -1.0
        return camera_pose_tmp



def crop(image, x_offset, y_offset):
    """Crops image to pixel offsets."""
    h, w, _ = image.shape
    return image[y_offset : h - y_offset, x_offset : w - x_offset, :]


def plot_scene_data(
    meshcat_vis,
    pc,
    obj_trans,
    obj_rot,
    obj_coll,
    bounds,
    scene_mesh,
    camera_poses,
    meshcat_robot=None,
    obj_mesh=None,
    plot_queries=False,
    plot_robot=False,
    plot_workspace_bounds=True,
    plot_object_mesh=False,
):
    """
    Plots the scene data in meshcat visualizater.
    """

    pc_color = np.tile([1, 0.706, 0], [pc.shape[0], 1]) * 255.0

    pc = np.hstack([pc, pc_color])

    if plot_queries:
        obj_pc = obj_trans
        obj_pc_color = obj_coll

        obj_pc_color = np.stack(
            [
                obj_pc_color,
                np.ones(obj_pc_color.shape[0]) - obj_pc_color,
                np.zeros(obj_pc_color.shape[0]),
            ],
            axis=1,
        )
        obj_pc_color = obj_pc_color * 255.0
        obj_pc = np.hstack([obj_pc, obj_pc_color])

        visualize_pointcloud(
            meshcat_vis,
            "collision_queries",
            pc=obj_pc[:, :3],
            color=obj_pc[:, 3:],
            size=0.01,
        )

    if plot_object_mesh:
        assert obj_mesh is not None

        visualize_mesh(
            meshcat_vis,
            "obj_mesh",
            obj_mesh,
            color=[255, 127, 80],
            transform=tra.translation_matrix([-2.0, 0, 1.0]),
        )

    if scene_mesh is not None:
        visualize_mesh(
            meshcat_vis,
            "scene",
            scene_mesh,
            color=[255, 127, 80],
            transform=np.eye(4),
        )

    visualize_pointcloud(meshcat_vis, "point_cloud", pc=pc[:, :3], color=pc[:, 3:], size=0.01)

    if plot_workspace_bounds:
        dims = bounds[1] - bounds[0]
        center = bounds[0] + dims / 2
        box_pose = tra.translation_matrix(center)
        dims = dims.tolist()
        visualize_bbox(meshcat_vis, "bounds", dims, T=box_pose)

    make_frame(meshcat_vis, f"robot_pose", T=np.eye(4), radius=0.02, h=0.25)

    for i, camera_pose in enumerate(camera_poses):
        make_frame(meshcat_vis, f"camera_{i}", T=camera_pose, radius=0.01)

    if plot_robot:
        visualize_robot(meshcat_vis, meshcat_robot, q=np.zeros(9), color=[125, 0, 125])


def plot_object_animation(
    meshcat_vis, obj_mesh, trans, rots, coll, obj_centroid, obj_pose, max_animate=200
):
    """Plots animation of object in collision trajectory path."""
    counter = 0
    poses = []
    color_trues = []

    for i, tr, rot, gt in zip(np.arange(len(trans)), trans, rots, coll):
        mesh_trans = tr - (obj_centroid - obj_pose[:3, 3])
        mesh_tf = np.eye(4)
        mesh_tf[:3, 3] = mesh_trans
        if (rot != 0).any():
            b1 = rot[:3]
            b2 = rot[3:] - b1.dot(rot[3:]) * b1
            b2 /= np.linalg.norm(b2)
            b3 = np.cross(b1, b2)
            mesh_tf[:3, :3] = np.stack((b1, b2, b3), axis=-1)
        new_pose = mesh_tf @ obj_pose

        poses.append(new_pose)

        color = get_color_from_score(1 - float(gt), use_255_scale=True)
        color = color.astype(np.int)
        color_trues.append(color)

        counter += 1
        if counter > max_animate:
            break

    for color, pose in zip(color_trues, poses):
        visualize_mesh(
            meshcat_vis,
            "obj",
            obj_mesh,
            color=color,
            transform=pose,
        )

        time.sleep(0.13)


def plot_trimesh_scene(names, meshes, poses, namespace="world"):
    """Constructs trimesh scene."""
    assert len(names) == len(meshes) == len(poses)

    s = trimesh.Scene(base_frame=namespace)
    for name, mesh, pose in zip(names, meshes, poses):
        s.add_geometry(
            node_name=f"{namespace}/{name}",
            geom_name=f"{namespace}/{name}",
            parent_node_name=s.graph.base_frame,
            geometry=mesh,
            transform=pose,
        )

    return s


def compute_camera_pose(center, distance, azimuth, elevation):
    """Construct 4x4 homogenous matrix, given camera center, distance, azimuth and elevation."""
    cam_tf = tra.euler_matrix(np.pi / 2, 0, 0).dot(tra.euler_matrix(0, np.pi / 2, 0))

    extrinsics = np.eye(4)
    extrinsics[0, 3] += distance
    extrinsics = tra.euler_matrix(0, -elevation, azimuth).dot(extrinsics)
    extrinsics[:3, 3] += center

    cam_pose = extrinsics.dot(cam_tf)
    frame_pose = cam_pose.copy()
    frame_pose[:, 1:3] *= -1.0
    return cam_pose, frame_pose


def get_cnet_model_frame(
    metadata,
    robot_pose,
    support_id,
    crop_id,
    randomize=True,
):
    """
    Returns the frame for CabiNet model inference.

    metadata, dict: scene information output from generate_clutter_scene function
    robot_pose, np.array: 4x4 transform of robot pose in world frame
    support_id, int: id of the object
    randomize, bool (optional): Add some noise to the selected support pose
    """

    T_robot_to_world = robot_pose

    support_ids = metadata["support_ids_with_placement"]
    assert support_id in support_ids
    support_name = metadata["support_ids"][support_id]

    poses = metadata["pose"]
    T_support_to_world = np.array(poses[support_name])
    if randomize:
        lim_xyz = [-0.05, 0.05]
        T_support_to_world[:3, 3] += np.random.uniform(low=lim_xyz[0], high=lim_xyz[1], size=3)
    T_world_to_support = np.linalg.inv(T_support_to_world)
    T_robot_to_model = T_world_to_support @ T_robot_to_world

    T_model_to_robot = np.linalg.inv(T_robot_to_model)
    anchor_point = tra.translation_from_matrix(T_model_to_robot)

    if randomize:
        lim_xyz = [-0.10, 0.10]
        anchor_point += np.random.uniform(lim_xyz[0], lim_xyz[1], 3)

    T_model_to_robot = tra.translation_matrix(anchor_point)

    if crop_id == 3:
        angle_z = np.random.uniform(low=-np.pi / 6, high=np.pi / 6)
        T_model_to_robot = T_model_to_robot @ tra.euler_matrix(0, 0, angle_z)

    T_robot_to_model = np.linalg.inv(T_model_to_robot)
    return T_model_to_robot, T_robot_to_model
