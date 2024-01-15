# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Standard Library
import multiprocessing as mp
import queue
import time

# Third Party
import numpy as np
import torch
import trimesh.transformations as tra

# NVIDIA
from cabi_net.robot.robot import Robot


class IK_TYPES:
    def __init__(self):
        self.TRACIKPY = "tracikpy"  # https://github.com/mjd3/tracikpy
        self.PYBULLET = "pybullet" #https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_kinematics.py
        self.TRACIK = "trac_ik"  # https://bitbucket.org/clemi/trac_ik/src/devel/


ik_types = IK_TYPES()


class IKEngine:
    def __init__(
        self,
        num_ik_procs: int,
        ik_backend: str = "trac_ik",
        robot_urdf: str = "assets/robot/franka/franka_panda.urdf",
        robot_eef_frame: str = "right_gripper",
        robot_base_frame: str = "panda_link0",
        device: int = 0,
    ):
        self.num_ik_procs = num_ik_procs
        self.ik_backend = ik_backend
        self._ik_output_queue = mp.Queue()
        self.ik_procs = []
        self.device = torch.device(f"cuda:{device}")
        self.robot_urdf = robot_urdf
        self.robot_eef_frame = robot_eef_frame
        self.robot_base_frame = robot_base_frame
        for _ in range(self.num_ik_procs):
            self.ik_procs.append(
                IKProc(
                    self._ik_output_queue,
                    ik=self.ik_backend,
                    robot_urdf=self.robot_urdf,
                    robot_eef_frame=self.robot_eef_frame,
                    robot_base_frame=self.robot_base_frame,
                )
            )
            self.ik_procs[-1].daemon = True
            self.ik_procs[-1].start()

        self.robot = Robot(
            self.robot_urdf,
            self.robot_eef_frame,
            device=self.device,
        )

    def ik(self, ee_poses: np.ndarray, init_q: np.ndarray = None, num_random_init: int = 4):
        """Computes ik."""
        print(f"entering ik, num_poses {len(ee_poses)} random_inits {num_random_init}")
        ik_time = time.time()
        num_ee_poses = len(ee_poses)
        for i, g in enumerate(ee_poses):
            if init_q is None or len(init_q.shape) == 1:
                q0 = init_q
            else:
                q0 = init_q[i]
            self.ik_procs[i % self.num_ik_procs].ik(g, q0, ind=i)

            for _ in range(num_random_init):
                rand_q = self.robot.sample_cfg()[0].cpu().numpy()
                self.ik_procs[i % self.num_ik_procs].ik(g, rand_q, ind=i)

        # collect computed iks
        inds = []
        iks = []
        for _ in range(num_ee_poses * (1 + num_random_init)):
            output = self._ik_output_queue.get(True)
            assert isinstance(output, tuple)
            assert len(output) == 2
            if output[1] is None:
                continue
            assert (
                output[0] < num_ee_poses
            ), f"index {output[0]} is out of bound for input_ee_poses of length {num_ee_poses}"
            inds.append(output[0])
            iks.append(output[1])

        assert self._ik_output_queue.qsize() == 0, self._ik_output_queue.qsize
        ik_time = time.time() - ik_time
        print(f"ik took {ik_time} seconds")

        return np.asarray(iks), np.asarray(inds)


class IKProc(mp.Process):
    """
    Used for finding ik in parallel.
    """

    def __init__(
        self,
        output_queue,
        ik="tracikpy",
        robot_urdf: str = "assets/robot/franka/franka_panda.urdf",
        robot_eef_frame: str = "right_gripper",
        robot_base_frame: str = "panda_link0",
    ):
        """
        Args:
        output_queue: mp.Queue, the queue that all the output data
            that is computed is added to.
        """
        super().__init__()
        self.output_queue = output_queue
        self.input_queue = mp.Queue()
        self.ik_mode = ik

        # Initialize IK Solver
        if self.ik_mode == ik_types.TRACIKPY:
            # Third Party
            from tracikpy import TracIKSolver

            self.ik_solver = TracIKSolver(
                robot_urdf,
                robot_base_frame,
                robot_eef_frame,
                timeout=0.04,
            )
        elif self.ik_mode == ik_types.TRACIK:
            # Third Party
            from trac_ik_python.trac_ik import IK

            urdfstring = "".join(open(robot_urdf, "r").readlines())
            self.ik_solver = IK(robot_base_frame, robot_eef_frame, urdf_string=urdfstring)
        elif self.ik_mode == ik_types.PYBULLET:
            self.ik_solver = None
            raise ValueError(f"IK solver {self.ik} is not supported by MPPI")
        else:
            raise ValueError(f"Provided invalid IK solver {self.ik}")

    def _ik(self, ee_pose, init_q):
        """computes collision free ik."""
        if init_q is None:
            init_q = [0.0] * 7

        if self.ik_mode == ik_types.TRACIKPY:
            q = self.ik_solver.ik(ee_pose, init_q[:7])
            if q is None:
                return None
            q = np.append(q, init_q[-2:])
            return q
        else:
            assert self.ik_mode == ik_types.TRACIK
            [x, y, z] = tra.translation_from_matrix(ee_pose).tolist()
            [rw, rx, ry, rz] = tra.quaternion_from_matrix(ee_pose).tolist()

            # Sometimes trac_ik only returns the solution after multiple initializations
            for _ in range(5):
                q = None
                q = self.ik_solver.get_ik(init_q[:7].tolist(), x, y, z, rx, ry, rz, rw)
                if q is not None:
                    break
            if q is None:
                return None
            q = np.append(q, [0, 0])
            return q

    def run(self):
        """
        the main function of each path collector process.
        """
        while True:
            try:
                request = self.input_queue.get(timeout=1)
            except queue.Empty:
                continue
            if request[0] == "ik":
                # ik grasp1 grasp2 init_q
                self.output_queue.put(
                    (
                        request[3],
                        self._ik(request[1], request[2]),
                    )
                )

    def ik(self, grasp, init_q, ind=None):
        self.input_queue.put(("ik", grasp, init_q, ind))
