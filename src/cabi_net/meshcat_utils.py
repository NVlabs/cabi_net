# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Third Party
import meshcat
import meshcat.geometry as g
import meshcat.transformations as mtf
import numpy as np
import trimesh.transformations as tra



def trimesh_to_meshcat_geometry(mesh):
    """
    Args:
        mesh: trimesh.TriMesh object
    """

    return meshcat.geometry.TriangularMeshGeometry(mesh.vertices, mesh.faces)

def is_rotation_matrix(M, tol=1e-4):
    tag = False
    I = np.identity(M.shape[0])

    if (np.linalg.norm((np.matmul(M, M.T) - I)) < tol) and (np.abs(np.linalg.det(M) - 1) < tol):
        tag = True

    if tag is False:
        print("M @ M.T:\n", np.matmul(M, M.T))
        print("det:", np.linalg.det(M))

    return tag

def visualize_mesh(vis, name, mesh, color=None, transform=None):
    """Visualize a mesh in meshcat"""

    if color is None:
        color = np.random.randint(low=0, high=256, size=3)

    mesh_vis = trimesh_to_meshcat_geometry(mesh)
    color_hex = rgb2hex(tuple(color))
    material = meshcat.geometry.MeshPhongMaterial(color=color_hex)
    vis[name].set_object(mesh_vis, material)

    if transform is not None:
        vis[name].set_transform(transform)


def rgb2hex(rgb):
    """
    Converts rgb color to hex

    Args:
        rgb: color in rgb, e.g. (255,0,0)
    """
    return "0x%02x%02x%02x" % (rgb)


def visualize_scene(vis, object_dict, randomize_color=True, visualize_transforms=False):
    for name, data in object_dict.items():
        # try assigning a random color
        if randomize_color:
            if "color" in data:
                color = data["color"]

                # if it's not an integer, convert it to [0,255]
                if not np.issubdtype(color.dtype, np.int):
                    color = (color * 255).astype(np.int32)
            else:
                color = np.random.randint(low=0, high=256, size=3)
                data["color"] = color
        else:
            color = [0, 255, 0]

        # mesh_vis = trimesh_to_meshcat_geometry(data['mesh_transformed'])
        mesh_vis = trimesh_to_meshcat_geometry(data["mesh"])
        color_hex = rgb2hex(tuple(color))
        material = meshcat.geometry.MeshPhongMaterial(color=color_hex)

        mesh_name = f"{name}/mesh"
        vis[mesh_name].set_object(mesh_vis, material)
        vis[mesh_name].set_transform(data["T_world_object"])

        if visualize_transforms:
            frame_name = f"{name}/transform"
            make_frame(vis, frame_name, T=data["T_world_object"])


def create_visualizer(clear=True):
    print("Waiting for meshcat server to start")
    vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    if clear:
        vis.delete()
    return vis


def make_frame(vis, name, h=0.15, radius=0.001, o=1.0, T=None):
    """Add a red-green-blue triad to the Meschat visualizer.
    Args:
      vis (MeshCat Visualizer): the visualizer
      name (string): name for this frame (should be unique)
      h (float): height of frame visualization
      radius (float): radius of frame visualization
      o (float): opacity
    """
    vis[name]["x"].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0xFF0000, reflectivity=0.8, opacity=o),
    )
    rotate_x = mtf.rotation_matrix(np.pi / 2.0, [0, 0, 1])
    rotate_x[0, 3] = h / 2
    vis[name]["x"].set_transform(rotate_x)

    vis[name]["y"].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0x00FF00, reflectivity=0.8, opacity=o),
    )
    rotate_y = mtf.rotation_matrix(np.pi / 2.0, [0, 1, 0])
    rotate_y[1, 3] = h / 2
    vis[name]["y"].set_transform(rotate_y)

    vis[name]["z"].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0x0000FF, reflectivity=0.8, opacity=o),
    )
    rotate_z = mtf.rotation_matrix(np.pi / 2.0, [1, 0, 0])
    rotate_z[2, 3] = h / 2
    vis[name]["z"].set_transform(rotate_z)

    if T is not None:
        is_valid = is_rotation_matrix(T[:3, :3])

        if not is_valid:
            raise ValueError("meshcat_utils:attempted to visualize invalid transform T")

        vis[name].set_transform(T)


def visualize_bbox(vis, name, dims, T=None, color=[255, 0, 0]):
    """Visualize a bounding box using a wireframe.

    Args:
        vis (MeshCat Visualizer): the visualizer
        name (string): name for this frame (should be unique)
        dims (array-like): shape (3,), dimensions of the bounding box
        T (4x4 numpy.array): (optional) transform to apply to this geometry

    """
    color_hex = rgb2hex(tuple(color))
    material = meshcat.geometry.MeshBasicMaterial(wireframe=True, color=color_hex)
    bbox = meshcat.geometry.Box(dims)
    vis[name].set_object(bbox, material)

    if T is not None:
        vis[name].set_transform(T)


def visualize_pointcloud(vis, name, pc, color=None, transform=None, **kwargs):
    """
    Args:
        vis: meshcat visualizer object
        name: str
        pc: Nx3 or HxWx3
        color: (optional) same shape as pc[0 - 255] scale or just rgb tuple
        transform: (optional) 4x4 homogeneous transform
    """
    if pc.ndim == 3:
        pc = pc.reshape(-1, pc.shape[-1])

    if color is not None:
        if isinstance(color, list):
            color = np.array(color)
        color = np.array(color)
        # Resize the color np array if needed.
        if color.ndim == 3:
            color = color.reshape(-1, color.shape[-1])
        if color.ndim == 1:
            color = np.ones_like(pc) * np.array(color)

        # Divide it by 255 to make sure the range is between 0 and 1,
        color = color.astype(np.float32) / 255
    else:
        color = np.ones_like(pc)

    vis[name].set_object(meshcat.geometry.PointCloud(position=pc.T, color=color.T, **kwargs))

    if transform is not None:
        vis[name].set_transform(transform)



def visualize_robot(vis, name, robot, q=None, color=None):
    if q is not None:
        robot.set_joint_cfg(q)
    robot_link_poses = {
        linkname: robot.link_poses[linkmesh][0].cpu().numpy()
        for linkname, linkmesh in robot.link_map.items()
    }
    link_id = -1
    for link_name in robot.physical_link_map:
        link_id += 1
        coll_mesh = robot.link_map[link_name].collision_mesh
        assert coll_mesh is not None
        link_color = None
        if color is not None and not isinstance(color, np.ndarray):
            color = np.asarray(color)
        if color.ndim == 1:
            link_color = color
        else:
            link_color = color[link_id]
        if coll_mesh is not None:
            visualize_mesh(
                vis["robot"],
                f"{link_name}_{robot}",
                coll_mesh,
                color=link_color,
                transform=robot_link_poses[link_name].astype(np.float),
            )


def get_color_from_score(labels, use_255_scale=False):
    scale = 255.0 if use_255_scale else 1.0
    if type(labels) in [np.float32, float]:
        return scale * np.array([1 - labels, labels, 0])
    else:
        scale = 255.0 if use_255_scale else 1.0
        return scale * np.stack(
            [np.ones(labels.shape[0]) - labels, labels, np.zeros(labels.shape[0])],
            axis=1,
        )


def visualize_triad(vis, name, color=[0, 255, 0], T=np.eye(4), radius=0.05):
    color_hex = rgb2hex(tuple(color))
    material = meshcat.geometry.MeshBasicMaterial(wireframe=False, color=color_hex)
    sphere = meshcat.geometry.Sphere(radius=radius)
    vis[name].set_object(sphere, material)

    if T is not None:
        vis[name].set_transform(T)
