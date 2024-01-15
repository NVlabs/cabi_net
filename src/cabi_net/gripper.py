# Standard Library
from pathlib import Path

# Third Party
import numpy as np
import torch
import trimesh
import trimesh.transformations as tra


def repeat_new_axis(tensor, rep, dim):
    reps = [1] * len(tensor.shape)
    reps.insert(dim, rep)
    return tensor.unsqueeze(dim).repeat(*reps)


def load_control_points(path=None):
    if path is None:
        path = f"{Path(__file__).parent.parent}/assets/robot/panda/panda.npy"
    control_points = torch.from_numpy(np.load(path))
    control_points = torch.cat(
        [
            control_points[[-2, 0]].float(),
            torch.tensor([[0, 0, 0, 1]]).float(),
            control_points[[1, -1]].float(),
        ]
    ).T  # 4x5
    return control_points


class PandaGripper(object):
    """An object representing a Franka Panda gripper."""

    def __init__(self, data_root_dir=None, q=None, num_contact_points_per_finger=10):
        """Create a Franka Panda parallel-yaw gripper object.
        Keyword Arguments:
            q {list of int} -- configuration (default: {None})
            num_contact_points_per_finger {int} -- contact points per finger (default: {10})
            data_root_dir {str} -- base folder for model files (default: {''})
        """
        self.joint_limits = [0.0, 0.04]
        self.default_pregrasp_configuration = 0.04

        if q is None:
            q = self.default_pregrasp_configuration

        self.q = q
        if data_root_dir is None:
            data_root_dir = f"{Path(__file__).parent.parent}/../assets/robot/panda"
        fn_base = data_root_dir + "/panda_gripper/hand.stl"
        fn_finger = data_root_dir + "/panda_gripper/finger.stl"
        self.base = trimesh.load(fn_base)
        self.finger_l = trimesh.load(fn_finger)
        self.finger_r = self.finger_l.copy()

        # transform fingers relative to the base
        self.finger_l.apply_transform(tra.euler_matrix(0, 0, np.pi))
        self.finger_l.apply_translation([+q, 0, 0.0584])
        self.finger_r.apply_translation([-q, 0, 0.0584])

        self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r])
        self.hand = trimesh.util.concatenate([self.fingers, self.base])

        self.ray_origins = []
        self.ray_directions = []
        for i in np.linspace(-0.01, 0.02, num_contact_points_per_finger):
            self.ray_origins.append(np.r_[self.finger_l.bounding_box.centroid + [0, 0, i], 1])
            self.ray_origins.append(np.r_[self.finger_r.bounding_box.centroid + [0, 0, i], 1])
            self.ray_directions.append(
                np.r_[-self.finger_l.bounding_box.primitive.transform[:3, 0]]
            )
            self.ray_directions.append(
                np.r_[+self.finger_r.bounding_box.primitive.transform[:3, 0]]
            )

        self.ray_origins = np.array(self.ray_origins)
        self.ray_directions = np.array(self.ray_directions)

        self.standoff_range = np.array(
            [
                max(
                    self.finger_l.bounding_box.bounds[0, 2],
                    self.base.bounding_box.bounds[1, 2],
                ),
                self.finger_l.bounding_box.bounds[1, 2],
            ]
        )
        self.standoff_range[0] += 0.001

    def get_obbs(self):
        """Get list of obstacle meshes.
        Returns:
            list of trimesh -- bounding boxes used for collision checking
        """
        return [
            self.finger_l.bounding_box,
            self.finger_r.bounding_box,
            self.base.bounding_box,
        ]

    def get_meshes(self):
        """Get list of meshes that this gripper consists of.
        Returns:
            list of trimesh -- visual meshes
        """
        return [self.finger_l, self.finger_r, self.base]

    def get_closing_rays(self, transform):
        """Get an array of rays defining the contact locations and directions on the hand.
        Arguments:
            transform {[nump.array]} -- a 4x4 homogeneous matrix
        Returns:
            numpy.array -- transformed rays (origin and direction)
        """
        return (
            transform[:3, :].dot(self.ray_origins.T).T,
            transform[:3, :3].dot(self.ray_directions.T).T,
        )
