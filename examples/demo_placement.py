# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Example script for running CabiNet collision checking for the task of placing a novel object in a unknown kitchen environment."""

# Standard Library
import argparse
from time import time

# Third Party
import numpy as np
import scipy
import torch
import trimesh.transformations as tra

# NVIDIA
from cabi_net.dataset.dataset import sample_points
from cabi_net.meshcat_utils import (
    create_visualizer,
    make_frame,
    visualize_pointcloud,
    visualize_bbox,
    visualize_robot,
)
from cabi_net.robot.robot import Robot
from cabi_net.collision_models.collision_checker import NNSceneCollisionChecker

NUM_POINTS_SCENE_PC = 29387
NUM_POINTS_OBJECT_PC = 1024
NUM_POINTS_SEED = 4000


def setup_robot_model(
    device: torch.device,
    urdf_path: str = "assets/franka/franka_panda.urdf",
    eef_frame: str = "right_gripper",
) -> Robot:
    """Initializes a robot object given URDF path and tool frame."""
    robot = Robot(
        urdf_path,
        eef_frame,
        device=device,
    )
    return robot


def setup_model(model_path: str, device: torch.device, robot: Robot) -> NNSceneCollisionChecker:
    """Initializes a CabiNet collision model given checkpoint path."""
    model = NNSceneCollisionChecker(
        model_path,
        robot,
        device=device,
        use_knn=False,
        logger=None,
        ignore_collision_with_base=True,
    )
    return model


get_model_bounds = lambda model: [
    model.model.bounds[0].cpu().numpy(),
    model.model.bounds[1].cpu().numpy(),
]


def setup_dataset(
    data_file_path: str,
) -> dict:
    """Loads and processes the scene data given file path."""
    obs = {}
    if data_file_path.endswith(".npy"):
        data = np.load(data_file_path, allow_pickle=True).item()

        pc = data["pc"]
        pc_color = data["pc_color"]
        pc_label = data["pc_label"]

        pt_idx = sample_points(pc, NUM_POINTS_SCENE_PC)
        print(
            "Point cloud loaded from",
            data_file_path,
            pc.shape[0],
            "points downsampled to",
            pt_idx.shape[0],
        )

        pc = pc[pt_idx]
        pc_color = pc_color[pt_idx]
        pc_label = pc_label[pt_idx]

        pts = tra.transform_points(
            data["pc"][data["placement_mask"] == 1],
            data["camera_pose"],
        )
        model_to_robot = np.eye(4)
        model_to_robot[:3, 3] = pts.mean(axis=0)
        model_to_robot[2, 3] += -0.05
        robot_to_model = tra.inverse_matrix(model_to_robot)

        try:
            obj_xyz = data["pc"][data["pc_label"] == data["grasp_target_id"]]
            obj_rgb = data["pc_color"][data["pc_label"] == data["grasp_target_id"]]
            obj_xyz = tra.transform_points(obj_xyz, data["camera_pose"])
        except:
            obj_xyz = None

        if obj_xyz is not None:
            obj_pt_idx = sample_points(obj_xyz, NUM_POINTS_OBJECT_PC)
            print(
                "Object point cloud loaded, ",
                obj_xyz.shape[0],
                "points downsampled to",
                obj_pt_idx.shape[0],
            )

        # Set the following, needed for placement later
        obs["pc"] = pc
        obs["pc_color"] = pc_color
        obs["pc_label"] = pc_label
        obs["camera_pose"] = data["camera_pose"]
        obs["label_map"] = data["label_map"]
        obs["model_to_robot"] = model_to_robot
        obs["robot_to_model"] = robot_to_model
        if obj_xyz is not None:
            obs["obj_pc"] = obj_xyz[obj_pt_idx]
            obs["obj_pc_color"] = obj_rgb[obj_pt_idx]
        obs["target_to_ee_transform"] = np.eye(4)
        obs["placement_mask"] = data["placement_mask"][pt_idx]
        obs["robot_q"] = data["latest_robot_q"]

        # Filter out points outside the bounds of the scene
        pc_highres = data["external_xyz_image"].reshape(-1, 3)
        pc_highres = tra.transform_points(pc_highres, data["camera_pose"])
        pc_higher_color = data["rgb_image"][1].reshape(-1, 3)
        bounds = [[-0.30, -1.5, -0.10], [2.0, 1.5, 2.0]]
        mask_within_bounds = np.all((pc_highres > bounds[0]), 1)
        mask_within_bounds = np.logical_and(mask_within_bounds, np.all((pc_highres < bounds[1]), 1))
        pc_highres = pc_highres[mask_within_bounds]
        pc_higher_color = pc_higher_color[mask_within_bounds]
        obs["pc_highres"] = pc_highres
        obs["pc_highres_color"] = pc_higher_color
    else:
        print(f"Invalid object dataset type {args.data_file}")

    return obs


def compute_placement(
    model: NNSceneCollisionChecker, data: dict, max_poses_sampled: int = 100000
) -> dict:
    """Complutes the object placement in the scene with CabiNet.

    Args:
        model: CabiNet model object.
        data: Dict with point clouds and necessary transforms
        max_poses_sampled: Limits on total placement pose candidates. Randomly
            subsamples if limit is exceeded.
    """
    output = {}
    pc = data["pc"]
    placement_mask = data["placement_mask"]
    camera_to_robot = data["camera_pose"]
    robot_to_model = data["robot_to_model"]
    model_to_robot = data["model_to_robot"]
    model_to_robot_tf = torch.from_numpy(model_to_robot).float().to(device)

    # Load placement points box
    pts_masked = placement_mask > 0
    pts = tra.transform_points(
        pc[pts_masked],
        camera_to_robot,
    )

    cvx_hull = scipy.spatial.Delaunay(pts[:, :2], qhull_options="QbB Pp")
    in_placement_box = lambda p: torch.from_numpy(
        cvx_hull.find_simplex(p.cpu().numpy()) >= 0
    ).cuda()

    pc_placement = pc[placement_mask > 0]

    pc_placement = tra.transform_points(pc_placement, camera_to_robot)
    pc_placement = tra.transform_points(pc_placement, robot_to_model)

    bounds = get_model_bounds(model)

    mask = np.all((pc_placement > bounds[0]), 1)
    mask = np.logical_and(mask, np.all((pc_placement < bounds[1]), 1))

    pc_placement = pc_placement[mask]

    placement_pts = pc_placement[
        np.random.randint(
            0,
            pc_placement.shape[0],
            min(NUM_POINTS_SEED, pc_placement.shape[0]),
        ),
        :,
    ]

    object_rotation = np.eye(4)

    placement_poses = (
        torch.from_numpy(object_rotation).float().to(device).repeat(len(placement_pts), 1, 1)
    )

    placement_scores = np.ones((placement_pts.shape[0],), dtype=np.float32)
    placement_pts = torch.from_numpy(placement_pts).to(device)
    placement_poses[:, :3, 3] = placement_pts

    if isinstance(placement_poses, np.ndarray):
        placement_poses = torch.from_numpy(placement_poses).to(device)
    placement_scores = torch.from_numpy(placement_scores).to(device)

    orig_placement_poses = placement_poses.clone()
    orig_placement_scores = placement_scores.clone()

    # Params taken from CabiNet paper experiments
    num_height_discretization = 8
    max_height = 0.20
    min_height = 0.05
    step = (max_height - min_height) / num_height_discretization
    model_offset = 0

    all_placement_poses = []
    all_placement_scores = []
    all_placement_poses_sampled = []  # This is stored just for debugging later
    all_placement_poses_sampled_beforebox = []  # This is stored just for debugging later

    for discretization_iter in range(num_height_discretization):
        placement_poses = orig_placement_poses.clone()
        placement_scores = orig_placement_scores.clone()

        lb = discretization_iter * step + min_height
        ub = lb + step

        placement_poses[:, 2, 3] = (
            torch.empty(len(placement_poses), device=device, dtype=torch.float32).uniform_(lb, ub)
            + model_offset
        )

        # Keep the points that are in placement boxes.
        placement_poses_in_robot_frame = model_to_robot_tf @ placement_poses
        mask = in_placement_box(placement_poses_in_robot_frame[:, :2, 3])
        placement_poses_beforebox = placement_poses.clone().detach()
        placement_poses = placement_poses[mask]
        placement_scores = placement_scores[mask]

        all_placement_poses_sampled.append(model_to_robot_tf @ placement_poses)
        all_placement_poses_sampled_beforebox.append(model_to_robot_tf @ placement_poses_beforebox)

        all_placement_poses.append(placement_poses.cpu().numpy())
        all_placement_scores.append(placement_scores.cpu().numpy())

    all_placement_poses = np.concatenate(all_placement_poses, 0)
    all_placement_scores = np.concatenate(all_placement_scores, 0)

    # Limit the number of placement poses sampled
    if len(all_placement_poses) > max_poses_sampled:
        mask = np.random.randint(0, len(all_placement_poses), max_poses_sampled)
        all_placement_poses = all_placement_poses[mask]
        all_placement_scores = all_placement_scores[mask]

    all_placement_poses = torch.from_numpy(all_placement_poses).to(device)
    all_placement_scores = torch.from_numpy(all_placement_scores).to(device)

    _, score = model.validate_obj_pose_goals(all_placement_poses)
    all_placement_scores = score

    all_placement_poses = model_to_robot_tf @ all_placement_poses
    all_placement_poses = all_placement_poses.cpu().numpy() @ data["target_to_ee_transform"]
    all_placement_scores = all_placement_scores.cpu().numpy()

    # this is for debugging placement for the shelf/cubby scenes
    all_placement_poses_sampled = torch.cat(all_placement_poses_sampled)
    all_placement_poses_sampled = all_placement_poses_sampled.cpu().numpy()
    all_placement_pts_sampled = all_placement_poses_sampled[:, :3, 3]

    # this is for debugging placement, the bounding box stuff
    all_placement_poses_sampled_beforebox = torch.cat(all_placement_poses_sampled_beforebox)
    all_placement_poses_sampled_beforebox = all_placement_poses_sampled_beforebox.cpu().numpy()
    all_placement_pts_sampled_beforebox = all_placement_poses_sampled_beforebox[:, :3, 3]

    placement_seed_pts = tra.transform_points(
        placement_pts.cpu().numpy(),
        model_to_robot,
    )

    output["all_placement_pts_sampled_beforebox"] = all_placement_pts_sampled_beforebox
    output["all_placement_pts_sampled"] = all_placement_pts_sampled
    output["placement_seed_pts"] = placement_seed_pts
    output["all_placement_poses_sampled"] = all_placement_poses_sampled
    output["all_placement_poses"] = all_placement_poses
    output["all_placement_scores"] = all_placement_scores
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file")
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--point-size", type=float, default=0.01)
    parser.add_argument("--max-poses-sampled", type=float, default=0.01)
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/cabinet_collision",
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
    vis = None if args.headless else create_visualizer()
    device = torch.device("cuda")

    # Load robot model
    robot = setup_robot_model(
        device, urdf_path=args.robot_model_path, eef_frame=args.robot_eef_frame
    )

    # Load CabiNet model
    model = setup_model(args.model_path, device, robot)

    # Load dataset
    data = setup_dataset(
        args.data_file,
    )

    pc = data["pc"]
    pc_color = data["pc_color"]
    camera_pose = data["camera_pose"]
    placement_mask = data["placement_mask"]

    # Update the model with the dataset
    model.set_scene(obs=data)
    model.set_object(obs=data)

    # Visualize the scene and object point clouds
    obj_pc = model.obj_pc.copy()
    obj_pc_color = data["obj_pc_color"]
    obj_pc = obj_pc - obj_pc.mean(axis=0)
    obj_pc_view_frame = tra.translation_matrix([0, -0.40, 0.20])
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

        # Note - this is the original high res point cloud used for visualization, it is not the one used for inference.
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

        # Plot the CabiNet workspace bounding box
        model_to_robot = data["model_to_robot"]
        bounds = get_model_bounds(model)
        bounds = tra.transform_points(bounds, model_to_robot)
        # Assumes bbox orientation is aligned with the robot frame
        dims = bounds[1] - bounds[0]
        center = bounds[0] + dims / 2
        box_pose = tra.translation_matrix(center)
        dims = dims.tolist()
        visualize_bbox(vis, "placement/bounds", dims, T=box_pose)

    # Compute the placement
    start_time = time()
    output = compute_placement(model, data)
    placement_time = time() - start_time
    all_placement_pts_sampled_beforebox = output["all_placement_pts_sampled_beforebox"]
    all_placement_pts_sampled = output["all_placement_pts_sampled"]
    placement_seed_pts = output["placement_seed_pts"]
    all_placement_poses = output["all_placement_poses"]
    all_placement_scores = output["all_placement_scores"]
    placement_mask_pts = tra.transform_points(
        data["pc"][data["placement_mask"] > 0],
        data["camera_pose"],
    )

    targets = all_placement_poses[:, :3, 3]
    confidence = all_placement_scores

    # The best placement pose is the one which is the most collision free, at least that's the heuristic we use here.
    best_placement_pose = all_placement_poses[np.argmax(confidence)]

    print(f"CabiNet computed {targets.shape[0]} placements in the scene, in {placement_time}s")

    mask = confidence > args.threshold
    confidence = confidence[mask]
    targets = targets[mask]
    target_colors = (
        np.stack([1 - confidence, confidence, np.zeros_like(confidence)], axis=1) * 255
    ).astype("uint8")

    if vis is not None:
        visualize_pointcloud(vis, "placement_targets", targets, target_colors, size=args.point_size)

        visualize_pointcloud(
            vis,
            "obj_pc",
            tra.transform_points(obj_pc, best_placement_pose),
            obj_pc_color,
            size=0.01,
        )

        # Extra debugging visualization
        if args.debug:
            # Entire placement mask
            visualize_pointcloud(
                vis,
                "placement/placement_mask_points",
                placement_mask_pts,
                color=[0, 255, 0],
                size=0.01,
            )

            # The seed points, computed from the scene point cloud, within the CabiNet bounds
            visualize_pointcloud(
                vis,
                "placement/placement_seed_pts_on_pc",
                placement_seed_pts,
                color=[0, 125, 0],
                size=0.02,
            )

            # Entire set of placement poses sampled
            visualize_pointcloud(
                vis,
                "placement/placement_positions_on_pc_sampled",
                all_placement_pts_sampled,
                color=[125, 0, 0],
                size=0.02,
            )

            placement_positions_feasible = all_placement_poses @ tra.inverse_matrix(
                data["target_to_ee_transform"]
            )

            # Final set of placement poses, that are collision free
            visualize_pointcloud(
                vis,
                "placement/placement_positions_feasible",
                placement_positions_feasible[:, :3, 3],
                color=[0, 255, 0],
                size=0.02,
            )
