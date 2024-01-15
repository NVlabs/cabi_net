# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Example script for running CabiNet waypoint prediction in a unknown kitchen environment for transitioning between a pick and place pose."""

# Standard Library
import argparse
import os

# Third Party
import numpy as np
import torch
import trimesh.transformations as tra
from demo_placement import setup_dataset, setup_robot_model
from matplotlib.cm import get_cmap

# NVIDIA
from cabi_net.gripper import PandaGripper
from cabi_net.meshcat_utils import (
    create_visualizer,
    make_frame,
    visualize_mesh,
    visualize_pointcloud,
    visualize_robot,
)
from cabi_net.model.waypoint import load_cabinet_model_for_inference


def get_set_colors():
    """Precomputes the fixed, color pallete."""
    n_colors = [9, 8, 12]
    colors = []
    for i, n in enumerate(n_colors):
        cmap = get_cmap(f"Set{i+1}")
        for j in range(n):
            colors.append([int(c * 255) for c in cmap(j / n)[:3]])
    return colors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file")
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--point-size", type=float, default=0.01)
    parser.add_argument("--max-poses-sampled", type=float, default=0.01)
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/cabinet_waypoint",
        help="Checkpoints for the CabiNet collision model",
    )
    parser.add_argument(
        "--robot-model-path",
        type=str,
        default="assets/robot/franka/franka_panda.urdf",
        help="Path to robot URDF file",
    )
    parser.add_argument(
        "--robot-eef-frame",
        type=str,
        default="right_gripper",
        help="Frame name of the robot end effector",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Visualize intermediate placement predictions",
        default=False,
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not start meshcat",
        default=False,
    )
    args = parser.parse_args()

    # Misc setup
    colors = get_set_colors()

    vis = None if args.headless else create_visualizer()
    device = torch.device("cuda")

    # Load robot model
    robot = setup_robot_model(
        device, urdf_path=args.robot_model_path, eef_frame=args.robot_eef_frame
    )

    # Load CabiNet waypoint model
    model, _ = load_cabinet_model_for_inference(
        os.path.join(args.model_path, "weights", "last.ckpt"),
        os.path.join(args.model_path, "inference.yaml"),
    )

    # Load dataset
    data = setup_dataset(
        args.data_file,
    )

    pc = data["pc"]
    pc_color = data["pc_color"]
    camera_pose = data["camera_pose"]
    placement_mask = data["placement_mask"]

    if vis is not None:
        make_frame(vis, "frames/camera", T=camera_pose.astype("float64"))

        make_frame(
            vis,
            "frames/origin",
            T=np.eye(4),
            h=0.40,
            radius=0.01,
        )

        make_frame(
            vis,
            "frames/cabinet_model_frame",
            T=data["model_to_robot"],
            h=0.40,
            radius=0.01,
        )

        # Note - this is the original high res point cloud used for visualization,
        # It is not the one used for inference.
        visualize_pointcloud(
            vis,
            "scene_pc",
            data["pc_highres"],
            data["pc_highres_color"],
            size=args.point_size,
        )

        visualize_robot(
            vis,
            "robot",
            robot,
            q=data["robot_q"],
            color=[169, 169, 169],
        )

        visualize_pointcloud(
            vis,
            "placement_mask",
            tra.transform_points(data["pc"][data["placement_mask"] == 1], data["camera_pose"]),
            color=[0, 255, 0],
            size=args.point_size,
        )

    T_robot_to_model = data["robot_to_model"]
    T_model_to_robot = tra.translation_matrix([0.35, 0, 0.76])
    T_robot_to_model = tra.inverse_matrix(T_model_to_robot)
    T_query_to_robot = tra.translation_matrix(robot.ee_pose.cpu().numpy()[0, :3, 3])

    make_frame(
        vis,
        "waypointnet/query",
        T=T_query_to_robot,
        radius=0.003,
        h=0.05,
    )

    waypoints, _ = model.run_inference(
        tra.transform_points(pc, data["camera_pose"]),
        T_robot_to_model,
        T_query_to_robot,
    )

    gripper = PandaGripper().hand
    ee_pose = robot.ee_pose.cpu().numpy().copy()[0]

    for i in range(waypoints.shape[0]):
        waypoint_pose = ee_pose.copy()
        waypoint_pose[:3, 3] = waypoints[i]
        make_frame(
            vis,
            f"waypoints/{i}",
            T=tra.translation_matrix(waypoints[i]),
            radius=0.003,
            h=0.05,
        )
        visualize_mesh(vis, f"gripper/{i}", gripper, transform=waypoint_pose.astype("float64"))
