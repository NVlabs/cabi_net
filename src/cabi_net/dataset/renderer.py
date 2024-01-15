# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Third Party
import numpy as np
import pyrender
import trimesh
import trimesh.transformations as tra
from autolab_core import ColorImage, DepthImage


class SceneRenderer:
    def __init__(self):
        self._scene = pyrender.Scene()
        self._node_dict = {}
        self._camera_intr = None
        self._camera_node = None
        self._light_node = None
        self._renderer = None

    def create_camera(self, intr, znear, zfar):
        cam = pyrender.IntrinsicsCamera(intr.fx, intr.fy, intr.cx, intr.cy, znear, zfar)
        self._camera_intr = intr
        self._camera_node = pyrender.Node(camera=cam, matrix=np.eye(4))
        self._scene.add_node(self._camera_node)
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=4.0)
        self._light_node = pyrender.Node(light=light, matrix=np.eye(4))
        self._scene.add_node(self._light_node)
        self._renderer = pyrender.OffscreenRenderer(
            viewport_width=intr.width,
            viewport_height=intr.height,
            point_size=5.0,
        )

    def set_camera_pose(self, cam_pose):
        if self._camera_node is None:
            raise ValueError("call create camera first!")
        self._scene.set_pose(self._camera_node, cam_pose)
        self._scene.set_pose(self._light_node, cam_pose)

    def get_camera_pose(self):
        if self._camera_node is None:
            raise ValueError("call create camera first!")
        return self._camera_node.matrix

    def render_rgbd(self, depth_only=False):
        if depth_only:
            depth = self._renderer.render(self._scene, pyrender.RenderFlags.DEPTH_ONLY)
            color = None
            depth = DepthImage(depth, frame="camera")
        else:
            color, depth = self._renderer.render(self._scene)
            color = ColorImage(color, frame="camera")
            depth = DepthImage(depth, frame="camera")
        return color, depth

    def render_segmentation(self, full_depth=None):
        if full_depth is None:
            _, full_depth = self.render_rgbd(depth_only=True)

        self.hide_objects()
        output = np.zeros(full_depth.data.shape, dtype=np.uint8)
        for i, obj_name in enumerate(self._node_dict):
            self._node_dict[obj_name].mesh.is_visible = True
            _, depth = self.render_rgbd(depth_only=True)
            mask = np.logical_and(
                (np.abs(depth.data - full_depth.data) < 1e-6),
                np.abs(full_depth.data) > 0,
            )
            if np.any(output[mask] != 0):
                raise ValueError("wrong label")
            output[mask] = i + 1
            self._node_dict[obj_name].mesh.is_visible = False
        self.show_objects()

        return output, ["BACKGROUND"] + list(self._node_dict.keys())

    def render_all(self):
        rgb, depth = self.render_rgbd()
        seg, labels = self.render_segmentation(depth)

        return rgb, depth, seg, labels

    def render_points(self, normals=True, segmentation=False):
        _, depth = self.render_rgbd(depth_only=True)
        point_norm_cloud = depth.point_normal_cloud(self._camera_intr)

        pts = point_norm_cloud.points.data.T.reshape(depth.height, depth.width, 3)
        norms = point_norm_cloud.normals.data.T.reshape(depth.height, depth.width, 3)
        cp = self.get_camera_pose()
        cp[:, 1:3] *= -1

        if segmentation:
            seg, labels = self.render_segmentation(depth)
            tab_pts = tra.transform_points(pts[seg == 1], cp)
            obj_pts = tra.transform_points(pts[seg > 1], cp)
            tab_norms = (cp[:3, :3] @ norms[seg == 1].T).T
            obj_norms = (cp[:3, :3] @ norms[seg > 1].T).T
            if normals:
                return (
                    np.concatenate((tab_pts, tab_norms), axis=1).astype(np.float32),
                    np.concatenate((obj_pts, obj_norms), axis=1).astype(np.float32),
                )
            else:
                return (
                    tab_pts.astype(np.float32),
                    obj_pts.astype(np.float32),
                )
        else:
            pt_mask = np.logical_and(
                np.linalg.norm(pts, axis=-1) != 0.0,
                np.linalg.norm(norms, axis=-1) != 0.0,
            )
            pts = tra.transform_points(pts[pt_mask], cp)
            norms = (cp[:3, :3] @ norms[pt_mask].T).T
            if normals:
                return np.concatenate((pts, norms), axis=1).astype(np.float32)
            else:
                return pts.astype(np.float32)

    def add_object(self, name, mesh, pose=None):
        if pose is None:
            pose = np.eye(4, dtype=np.float32)

        node = pyrender.Node(
            name=name,
            mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False),
            matrix=pose,
        )
        self._node_dict[name] = node
        self._scene.add_node(node)

    def add_points(self, points, name, pose=None, color=None, radius=0.005):
        points = np.asanyarray(points)
        if points.ndim == 1:
            points = np.array([points])

        if pose is None:
            pose = np.eye(4)
        else:
            pose = pose.matrix

        color = np.asanyarray(color, dtype=np.float) if color is not None else None

        # If color specified per point, use sprites
        if color is not None and color.ndim > 1:
            self._renderer.point_size = 1000 * radius
            m = pyrender.Mesh.from_points(points, colors=color)
        # otherwise, we can make pretty spheres
        else:
            mesh = trimesh.creation.uv_sphere(radius, [20, 20])
            if color is not None:
                mesh.visual.vertex_colors = color
            poses = None
            poses = np.tile(np.eye(4), (len(points), 1)).reshape(len(points), 4, 4)
            poses[:, :3, 3::4] = points[:, :, None]
            m = pyrender.Mesh.from_trimesh(mesh, poses=poses)

        node = pyrender.Node(mesh=m, name=name, matrix=pose)
        self._node_dict[name] = node
        self._scene.add_node(node)

    def set_object_pose(self, name, pose):
        self._scene.set_pose(self._node_dict[name], pose)

    def has_object(self, name):
        return name in self._node_dict

    def remove_object(self, name):
        self._scene.remove_node(self._node_dict[name])
        del self._node_dict[name]

    def show_objects(self, names=None):
        for name, node in self._node_dict.items():
            if names is None or name in names:
                node.mesh.is_visible = True

    def toggle_wireframe(self, names=None):
        for name, node in self._node_dict.items():
            if names is None or name in names:
                node.mesh.primitives[0].material.wireframe ^= True

    def hide_objects(self, names=None):
        for name, node in self._node_dict.items():
            if names is None or name in names:
                node.mesh.is_visible = False

    def reset(self):
        for name in self._node_dict:
            self._scene.remove_node(self._node_dict[name])
        self._node_dict = {}
