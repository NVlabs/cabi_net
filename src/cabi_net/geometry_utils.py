# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Third Party
import numpy as np
import trimesh


def convert_pc_to_mesh(pointcloud):
    """
    Runs mesh reconstruction using marching cubes/poisson
    """
    # Third Party
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
    alpha = 0.03
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    mesh = trimesh.Trimesh(
        np.asarray(mesh.vertices),
        np.asarray(mesh.triangles),
        vertex_normals=np.asarray(mesh.vertex_normals),
    )
    return mesh


def get_mesh_mesh_intersection(mesh1, mesh2):
    """
    This function needs scad installed to work
    """
    try:
        return True, trimesh.boolean.intersection([mesh1, mesh2], engine="scad")
    except:
        return False, None


def get_mesh_mesh_intersection_with_pose(mesh1, pose1, mesh2, pose2):
    """
    Use this when get_mesh_mesh_intersection does not work (due to backend issues)
    """
    assert type(mesh1) == trimesh.Trimesh
    assert type(mesh2) == trimesh.Trimesh
    assert type(pose1) == np.ndarray
    assert type(pose2) == np.ndarray

    mesh1 = mesh1.copy()
    mesh2 = mesh2.copy()

    scene = trimesh.collision.CollisionManager()
    scene.add_object("mesh1", mesh1, pose1)

    return scene.in_collision_single(mesh2, pose2)


def as_trimesh_scene(meshes, transforms, namespace="world"):
    s = trimesh.Scene(base_frame=namespace)

    for mesh, transform in zip(meshes, transforms):
        s.add_geometry(
            parent_node_name=s.graph.base_frame,
            geometry=mesh,
            transform=transform,
        )

    return s


def sample_mesh_volume(mesh, count):
    """Samples <count> numer of points within mesh"""
    pts = trimesh.sample.volume_mesh(mesh, count=count)
    while pts.shape[0] <= count:
        pts = np.vstack([pts, trimesh.sample.volume_mesh(mesh, count=count)])
    pts = pts[:count, :]
    return pts


def get_cubes_from_pc(pc, num_cubes=2000, cube_size=0.01):
    """Simple collision representation from point clouds. Cubes placed at point locations."""
    num_cubes = min(pc.shape[0], num_cubes)
    cube_pts = pc[np.random.randint(pc.shape[0], size=num_cubes), :]
    meshes = []
    transforms = []

    for cube_pt in cube_pts:
        cube = trimesh.primitives.Box(extents=[cube_size] * 3)
        meshes.append(cube)
        transforms.append(trimesh.transformations.translation_matrix(cube_pt))

    scene = as_trimesh_scene(meshes, transforms)
    mesh = scene.dump().sum()
    return mesh, cube_pts
