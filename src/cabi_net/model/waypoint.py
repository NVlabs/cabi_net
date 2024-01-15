# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Standard Library
import collections
import os
import time

# Third Party
import numpy as np
import pytorch_lightning as pl
import ruamel
import torch
import trimesh.transformations as tra
from torch import nn
from torch.utils.data import DataLoader

# NVIDIA
from cabi_net.dataset.dataset import CollisionDataset
from cabi_net.errors import BadPointCloudRendering, VoxelOverflowError
from cabi_net.geometry_utils import sample_mesh_volume
from cabi_net.model.cabinet import CabiNetCollision
from cabi_net.utils import convert_to_dict
from pointnet2.pointnet2_modules import PointnetSAModule

yaml = ruamel.yaml.YAML(typ="safe", pure=True)

os.environ["PYOPENGL_PLATFORM"] = "egl"

# Standard Library
import logging

logging.getLogger("yourdfpy").setLevel(logging.ERROR)
logging.getLogger("cabinet").setLevel(logging.ERROR)

log = logging.getLogger("cabinet")


def loss_implicit_maximum_likelihood_for_3dpoints(x_gt, x_pred):
    """
    x_gt is torch tensor sized [N_gt, 3]
    x_pred is torch tensor sized [N_pred, 3]
    """
    N_gt = x_gt.shape[0]
    N_pred = x_pred.shape[0]
    assert x_gt.shape[1] == 3
    assert x_pred.shape[1] == 3

    x_gt = torch.tile(x_gt.unsqueeze(0), [N_pred, 1, 1])  # N_pred x N_gt x 3
    x_pred = torch.tile(x_pred.unsqueeze(1), [1, N_gt, 1])  # N_pred x N_gt x 3
    error = x_gt - x_pred  # N_pred x N_gt x 3
    error = torch.abs(error)  # N_pred x N_gt x 3
    error = error.sum(dim=2)  # Take the sum across all the 3 dims, result is # N_pred x N_gt

    error_gt, _ = torch.min(error, dim=0)  # take the min distance for each gt pt, result is N_gt
    assert error_gt.shape[0] == N_gt
    error_gt = error_gt.mean()

    error_pred, _ = torch.min(
        error, dim=1
    )  # take the min distance for each pred pt, result is N_pred
    assert error_pred.shape[0] == N_pred

    error_pred = error_pred.mean()

    return error_gt, error_pred


class WaypointPrediction(pl.LightningModule):
    def __init__(self, cfg, device):
        super().__init__()

        self.cfg = cfg
        self.cfg_dict = convert_to_dict(self.cfg)
        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.05,
                nsample=128,
                mlp=[2, 64, 64, 64],
                use_xyz=True,
                bn=False,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.3,
                nsample=128,
                mlp=[64, 128, 128, 256],
                bn=False,
            )
        )
        self.SA_modules.append(PointnetSAModule(mlp=[256, 512, 512, 1024], bn=False))

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 3),
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        """Forward pass of WaypointPrediction"""

        latents = torch.rand(self.cfg.model.num_waypoints, self.cfg.model.latent_dim)
        latents = latents.unsqueeze(1)
        latents = torch.tile(latents, [1, self.cfg.dataset.n_scene_points, 1])
        latents = latents.to(self.device)

        pointcloud = torch.tile(pointcloud, [self.cfg.model.num_waypoints, 1, 1])
        pointcloud = pointcloud.to(self.device)

        assert pointcloud.shape[0] == latents.shape[0]
        assert pointcloud.shape[1] == self.cfg.dataset.n_scene_points
        assert pointcloud.shape[2] == 3

        pointcloud = pointcloud.type(torch.cuda.FloatTensor)
        latents = latents.type(torch.cuda.FloatTensor)

        pointcloud = torch.cat([pointcloud, latents], axis=2)

        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        waypoints = self.fc_layer(features.squeeze(-1))
        return waypoints

    def loss(self, trues, pred, return_all_losses=True):
        loss_gt, loss_pred = loss_implicit_maximum_likelihood_for_3dpoints(trues, pred)
        loss = (
            (loss_gt + self.cfg.trainer.robust_loss_prec_weight * loss_pred)
            if self.cfg.trainer.robust_loss
            else loss_gt
        )

        if return_all_losses:
            return [loss_gt, loss_pred, loss]

        return loss

    def training_step(self, batch):
        (pc, waypoints_gt) = batch
        pc = pc.squeeze(0)
        waypoints_gt = waypoints_gt.squeeze(0)

        waypoints_pred = self.forward(pc)

        loss_gt, loss_pred, loss = self.loss(waypoints_gt, waypoints_pred)

        log = dict(train_loss=loss)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "train_loss_gt",
            loss_gt,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_loss_pred",
            loss_pred,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return dict(loss=loss, log=log, progress_bar=log)

    def validation_step(self, batch, batch_idx):
        (pc, waypoints_gt) = batch
        pc = pc.squeeze(0)
        waypoints_gt = waypoints_gt.squeeze(0)

        waypoints_pred = self.forward(pc)

        loss_gt, loss_pred, loss = self.loss(waypoints_gt, waypoints_pred)

        log = dict(val_loss=loss)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "val_loss_gt",
            loss_gt,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_loss_pred",
            loss_pred,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return dict(loss=loss, log=log, progress_bar=log)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.cfg.trainer.lr,
            momentum=self.cfg.trainer.momentum,
        )

        return optimizer

    def setup_data_loader(self, name="Train", num_workers=1, joint_dataloader=False):
        print(f"Setting up {name} DataLoader with {num_workers} workers")

        dataset = WaypointDataset(
            **self.cfg_dict["dataset"],
            **self.cfg_dict["camera"],
            bounds=self.cfg_dict["model"]["bounds"],
            cfg=self.cfg,
            joint_dataloader=joint_dataloader,
        )

        kwargs = {
            "num_workers": num_workers,
            "pin_memory": False,
            "worker_init_fn": lambda _: np.random.seed(),
        }

        return DataLoader(dataset, batch_size=1, **kwargs)

    def train_dataloader(self):
        return self.setup_data_loader(name="Train", num_workers=self.cfg.trainer.num_workers_train)

    def val_dataloader(self):
        return self.setup_data_loader(
            name="Validation", num_workers=self.cfg.trainer.num_workers_val
        )


class CabiNetWaypoint(WaypointPrediction):
    def __init__(
        self,
        cfg,
        device,
    ):
        self.device_arg = device
        super().__init__(cfg, device=device)

    def _build_model(self):
        cnet_model_path = self.cfg.model.cnet_model_path

        with open(os.path.join(cnet_model_path, "train.yaml")) as f:
            cfg_cnet = yaml.load(f)

        model = CabiNetCollision(
            config=cfg_cnet,
            device=self.device_arg,
        )
        self.model_collision = model.scene_encoder.cuda()

        checkpoint_fname = os.path.join(cnet_model_path, "model_best.pth.tar")
        if not os.path.exists(checkpoint_fname):
            checkpoint_fname = os.path.join(cnet_model_path, "checkpoint.pth.tar")
            if not os.path.exists(checkpoint_fname):
                raise FileNotFoundError(
                    f"Specified checkpoint file {checkpoint_fname} does not exist!"
                )

        checkpoint = torch.load(checkpoint_fname, map_location=self.device_arg)
        print(f"Loading CabiNetWaypoint backbone checkpoint from file {checkpoint_fname}")

        # Extract keys just for scene encoder
        model_state_dict_scene_encoder = collections.OrderedDict()
        for key in checkpoint["model_state_dict"]:
            if key.find("scene_encoder") >= 0:
                key_modified = key[key.find(".") + 1 :]
                model_state_dict_scene_encoder[key_modified] = checkpoint["model_state_dict"][key]

        self.model_collision.load_state_dict(model_state_dict_scene_encoder)

        if self.cfg.model.freeze_cnet:
            # No fine-tuning, use scene features as-is
            for param in self.model_collision.parameters():
                param.requires_grad = False

        num_feature_dim = 1024 + self.cfg.model.latent_dim + 3
        self.fc_layer = nn.Sequential(
            nn.Linear(num_feature_dim, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 3),
        )

    def forward(self, pc, query):
        """
        pointcloud pc (tensor): of size (1, N, 3)
        query (tensor): of size (1, 3)

        """
        """Forward pass of CabiNetWaypoint"""

        scene_features = self.model_collision(pc)

        # Get voxel indices for translations
        trans_inds = self.model_collision.voxel_inds(query, scale=2).long()
        if trans_inds.max() >= scene_features.shape[2]:
            print(query[trans_inds.argmax()], trans_inds.max(), scene_features.shape)

        if trans_inds.max() > scene_features.shape[2] or trans_inds.min() < 0:
            raise VoxelOverflowError()

        vox_trans_features = scene_features[..., trans_inds].transpose(2, 1)

        # Calculate translation offsets from centers of voxels
        T_voxel_to_model = (
            self.model_collision._inds_from_flat(trans_inds, scale=2)
            * self.model_collision.vox_size
            * 2
            + self.model_collision.vox_size / 2
            + self.model_collision.bounds[0]
        )
        T_query_to_voxel = query - T_voxel_to_model.float()

        pc = pc.squeeze(0)

        scene_feature = vox_trans_features.squeeze(0)

        latents = torch.rand(self.cfg.model.num_waypoints, self.cfg.model.latent_dim)
        latents = latents.to(self.device)

        scene_feature = torch.tile(scene_feature, [self.cfg.model.num_waypoints, 1])

        query = torch.tile(T_query_to_voxel, [self.cfg.model.num_waypoints, 1])

        latents = torch.cat((scene_feature, latents, query), dim=1)
        latents = latents.float()

        waypoints = self.fc_layer(latents)
        waypoints = waypoints + T_voxel_to_model.float()

        return waypoints

    def training_step(self, batch):
        (pc, waypoints_gt, T_query_to_model) = batch
        waypoints_gt = waypoints_gt.squeeze(0)
        query = T_query_to_model[:, :3, 3]

        waypoints_pred = self.forward(pc, query)

        loss_gt, loss_pred, loss = self.loss(waypoints_gt, waypoints_pred)

        log = dict(train_loss=loss)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "train_loss_gt",
            loss_gt,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_loss_pred",
            loss_pred,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return dict(loss=loss, log=log, progress_bar=log)

    def validation_step(self, batch):
        (pc, waypoints_gt, T_query_to_model) = batch
        waypoints_gt = waypoints_gt.squeeze(0)
        query = T_query_to_model[:, :3, 3]

        waypoints_pred = self.forward(pc, query)

        loss_gt, loss_pred, loss = self.loss(waypoints_gt, waypoints_pred)

        log = dict(val_loss=loss)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "val_loss_gt",
            loss_gt,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_loss_pred",
            loss_pred,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return dict(loss=loss, log=log, progress_bar=log)

    def run_inference(
        self, point_cloud: np.ndarray, T_robot_to_model: np.ndarray, T_query_to_robot: np.ndarray
    ) -> np.ndarray:
        """
        Runs inference on a single point cloud, given the robot's end-effector pose and model pose.
        """

        T_query_to_robot[0, 3] -= 0.15  # offset to account for the robot's gripper length
        T_model_to_robot = np.linalg.inv(T_robot_to_model)
        T_robot_to_query = np.linalg.inv(T_query_to_robot)
        T_model_to_query = T_model_to_robot @ T_robot_to_query
        T_query_to_model = np.linalg.inv(T_model_to_query)

        point_cloud = tra.transform_points(point_cloud, T_robot_to_model)

        b = self.cfg.dataset.num_waypoints

        bounds = np.array(self.cfg_dict["model"]["bounds"])

        point_cloud_masked = point_cloud.copy()
        mask = (point_cloud_masked[:, :3] > bounds[0] + 1e-4).all(axis=1)

        point_cloud_masked = point_cloud_masked[mask]

        mask = (point_cloud_masked[:, :3] < bounds[1] - 1e-4).all(axis=1)
        point_cloud_masked = point_cloud_masked[mask]

        point_cloud_masked = point_cloud_masked[
            np.random.randint(
                point_cloud_masked.shape[0],
                size=self.cfg.dataset.n_scene_points,
            ),
            :,
        ]

        point_cloud = point_cloud_masked
        point_cloud = torch.from_numpy(point_cloud).to(self.device)
        point_cloud = point_cloud.unsqueeze(0)

        T_query_to_model = torch.from_numpy(T_query_to_model).to(self.device)
        query = T_query_to_model[:3, 3]

        point_cloud = point_cloud.float()
        query = query.float()
        query = query.unsqueeze(0)

        with torch.no_grad():
            waypoints = self.forward(point_cloud, query)

        waypoints = waypoints.cpu().numpy()
        waypoints = tra.transform_points(waypoints, T_model_to_robot)
        return waypoints, point_cloud_masked

    def setup_data_loader(self, name="Train", num_workers=1):
        return super().setup_data_loader(name, num_workers, joint_dataloader=True)


def get_waypoints_from_mesh(
    scene_mesh,
    num_init_samples=75000,
    lim_dist=[-0.20, -0.30],
    lim_donut=[0.20, 0.30],
    eef_position=None,
):
    """
    Function to generate waypoints from mesh
    """
    # Third Party
    from pysdf import SDF

    extents = [1.1, 1.1, 1.1]

    waypoint_volume_mesh = trimesh.primitives.Box(extents=extents)
    scene_pose = tra.translation_matrix(eef_position)
    waypoint_volume_mesh.apply_transform(scene_pose)

    sdf_func = SDF(scene_mesh.vertices, scene_mesh.faces)

    waypoints_init = sample_mesh_volume(waypoint_volume_mesh, num_init_samples)
    val = sdf_func(waypoints_init)
    val_norm = (val - val.min()) / (val.max() - val.min())
    # NVIDIA
    from cabi_net.meshcat_utils import get_color_from_score

    color_init = get_color_from_score(val_norm, use_255_scale=True)

    mask = np.logical_and(val <= lim_dist[0], val >= lim_dist[1])

    waypoints_bounded = waypoints_init[mask]
    color_bounded = color_init[mask]

    center = eef_position
    dist = np.linalg.norm(
        waypoints_bounded - np.tile(center, [waypoints_bounded.shape[0], 1]),
        axis=1,
    )
    mask_1 = dist < lim_donut[0]
    mask_2 = dist < lim_donut[1]
    mask_within_valid = mask_1 != mask_2
    waypoints_final = waypoints_bounded[mask_within_valid]
    color_final = color_bounded[mask_within_valid]

    if len(waypoints_final) == 0:
        return None

    results = {
        "waypoints_init": waypoints_init,
        "color_init": color_init,
        "waypoints_bounded": waypoints_bounded,
        "color_bounded": color_bounded,
        "waypoints_final": waypoints_final,
        "color_final": color_final,
    }
    return results


class WaypointDataset(CollisionDataset):
    def __init__(
        self,
        cfg=None,
        joint_dataloader=False,
        num_waypoints=100,  # Just to preserve API
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task_id = 4
        self.cfg = cfg
        self.cfg_dict = convert_to_dict(cfg)
        self.joint_dataloader = joint_dataloader
        self._caching = self.cfg.trainer.caching >= 0
        self._caching_idx = 0
        self._caching_rate = self.cfg.trainer.caching

    # Generator that yields batches of training tuples
    def __iter__(self):
        while True:
            if (self._caching_idx % self._caching_rate == 0) or not self._caching:
                if hasattr(self, "_cached_data"):
                    del self._cached_data
                try:
                    (
                        point_cloud_uncropped,
                        point_cloud,
                        _,
                        _,
                        _,
                        _,  # this is scene mesh with walls, etc.
                        scene_mesh,
                        _,
                    ) = self.get_scene()
                except BadPointCloudRendering:
                    continue

                t0 = time.time()
                waypoints, scene_data = self.get_waypoints(scene_mesh)

                if waypoints is None:
                    continue
                log.debug(f"Getting waypoints took {time.time() - t0}s")

                b = self.cfg.dataset.num_waypoints
                waypoints_final = waypoints["waypoints_final"]
                eef_position = waypoints["eef_position"]
                T_query_to_model = tra.translation_matrix(eef_position)
                T_model_to_query = np.linalg.inv(T_query_to_model)

                if waypoints_final.shape[0] <= 10:
                    continue

                if not self.joint_dataloader:
                    waypoints_final = tra.transform_points(waypoints_final, T_model_to_query)

                waypoints_gt = waypoints_final[
                    np.random.randint(waypoints_final.shape[0], size=b), :
                ]

                bounds = self.cfg_dict["model"]["bounds"]
                bounds = np.array(bounds)

                points_new = point_cloud_uncropped.copy()
                mask = (points_new[:, :3] > bounds[0] + 1e-4).all(axis=1)

                points_new = points_new[mask]

                mask = (points_new[:, :3] < bounds[1] - 1e-4).all(axis=1)
                points_new = points_new[mask]

                if not self.joint_dataloader:
                    points_new = tra.transform_points(points_new, T_model_to_query)

                point_cloud = points_new
                if self._caching:
                    self._cached_data = (
                        point_cloud,
                        waypoints_final,
                        T_query_to_model,
                    )

                pt_inds = np.random.choice(
                    point_cloud.shape[0], size=self.cfg.dataset.n_scene_points
                )
                point_cloud = point_cloud[pt_inds]

                log.debug(f"Found {waypoints_final.shape[0]} waypoints. Choosing {b} out of them.")
                yield point_cloud, waypoints_gt, T_query_to_model

            else:
                if self._cached_data is None:
                    raise ValueError("Cached data should not be empty")

                (point_cloud, waypoints_final, T_query_to_model) = self._cached_data

                pt_inds = np.random.choice(
                    point_cloud.shape[0], size=self.cfg.dataset.n_scene_points
                )
                point_cloud = point_cloud[pt_inds]

                waypoints_gt = waypoints_final[
                    np.random.randint(waypoints_final.shape[0], size=b), :
                ]
                yield point_cloud, waypoints_gt, T_query_to_model
            self._caching_idx += 1

    def get_waypoints(
        self,
        scene_mesh,
    ):
        scene_data = self._cached_scene

        scene_metadata = scene_data["scene_metadata"]
        T_world_to_model = scene_metadata["T_world_to_model"]

        obj_poses = scene_metadata["pose"]
        obj_ids = scene_metadata["obj_ids"]

        if len(obj_ids) == 0:
            return None, None
        else:
            obj_id = np.random.choice(obj_ids)
            obj_pose = obj_poses[obj_id]
            obj_pose = T_world_to_model @ obj_pose
            eef_position = tra.translation_from_matrix(obj_pose)
            eef_position[0] += -np.random.uniform(0.10, 0.15)
            eef_position[2] += np.random.uniform(0.10, 0.12)

        results = get_waypoints_from_mesh(
            scene_mesh,
            lim_dist=[-0.42, -0.45],
            lim_donut=[0.42, 0.45],
            eef_position=eef_position,
        )

        if results is None:
            return None, None

        results["eef_position"] = eef_position
        results["obj_pose"] = obj_pose
        results["scene_metadata"] = scene_metadata

        return results, scene_data

    def filter_waypoints(self):
        pass


def load_cabinet_model_for_inference(ckpt_path, cfg_file):
    # NVIDIA
    from cabi_net.config.waypoint import get_cfg_defaults

    cfg = get_cfg_defaults()

    if cfg_file != "":
        if os.path.exists(cfg_file):
            cfg.merge_from_file(cfg_file)
        else:
            raise FileNotFoundError(cfg_file)

    if ckpt_path != "":
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)

    if torch.cuda.is_available():
        print("CUDA is available, make sure you are training or running inference on the GPU")

    cfg.freeze()
    print(cfg)

    # Setup Model
    kwargs = {"cfg": cfg, "device": torch.device("cuda")}
    # NVIDIA
    from cabi_net.model.waypoint import CabiNetWaypoint, WaypointPrediction

    if cfg.model.model_arch == "WaypointPrediction":
        print(f"Loading checkpoint from {ckpt_path}")
        model = WaypointPrediction.load_from_checkpoint(ckpt_path, **kwargs).cuda()
    elif cfg.model.model_arch == "CabiNetWaypoint":
        print(f"Loading checkpoint from {ckpt_path}")
        model = CabiNetWaypoint.load_from_checkpoint(ckpt_path, **kwargs).cuda()
    else:
        raise NotImplementedError(f"Unknown architecture {cfg.model.model_arch}")
    model.eval()

    return model, cfg
