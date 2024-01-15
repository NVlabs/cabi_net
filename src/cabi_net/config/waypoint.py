# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Third Party
from yacs.config import CfgNode as CN

# model
_C = CN()
_C.model = CN()
_C.model.name = "CabiNetWaypoint"
_C.model.path = "weights_cabinet_waypoint"
_C.model.bounds = [[-0.5, -0.8, -0.06], [0.5, 0.8, 0.44]]
_C.model.vox_size = [0.125, 0.1, 0.125]
_C.model.activation = "relu"
_C.model.bn = 0
_C.model.latent_dim = 2
_C.model.num_waypoints = 70
_C.model.model_arch = "CabiNetWaypoint"
_C.model.freeze_cnet = True
_C.model.cnet_model_path = ""

# Training
_C.trainer = CN()
_C.trainer.accelerator = "gpu"
_C.trainer.devices = [0]
_C.trainer.num_workers_train = 8
_C.trainer.num_workers_val = 2
_C.trainer.lr = 0.001
_C.trainer.momentum = 0.9
_C.trainer.debug = False
_C.trainer.every_n_epochs = 10  # Save checkpoint every 4 epochs
_C.trainer.max_epochs = 10000000000  # Train to infinity and beyond!
_C.trainer.val_check_interval = 500  # Checking validation every K training iterations!
_C.trainer.limit_train_batches = 1000  # Iterations/batches per epoch
_C.trainer.limit_val_batches = 6  # Iterations/batches per epoch
_C.trainer.log_every_n_steps = 10  # Log metrics every K iterations
_C.trainer.robust_loss = True
_C.trainer.robust_loss_prec_weight = 1.0
_C.trainer.caching = 3

# Dataset
_C.dataset = CN()
_C.dataset.query_size = 32768  # this is not actually needed
_C.dataset.n_obj_points = 1024  # this is not actually needed
_C.dataset.meshes = ""
_C.dataset.batch_size = 1
_C.dataset.n_scene_points = 1024
_C.dataset.num_scene_cameras = 0
_C.dataset.num_eih_cameras = 1
_C.dataset.env_type = "cubby"
_C.dataset.batch_multiplier = 1
_C.dataset.test = False
_C.dataset.floating_mpc = False
_C.dataset.num_waypoints = 70

# Camera - intrinsics
_C.camera = CN()
_C.camera.intrinsics = CN()
_C.camera.intrinsics.frame = "camera"
_C.camera.intrinsics.fx = 616.36529541
_C.camera.intrinsics.fy = 616.20294189
_C.camera.intrinsics.cx = 310.25881958
_C.camera.intrinsics.cy = 236.59980774
_C.camera.intrinsics.skew = 0.0
_C.camera.intrinsics.width = 640
_C.camera.intrinsics.height = 480

# Camera - extrinsics
_C.camera.extrinsics = CN()
_C.camera.extrinsics.azimuth = [2.54, 3.74]
_C.camera.extrinsics.elevation = [0.3, 0.5]
_C.camera.extrinsics.radius = [1.3, 2.0]
_C.camera.extrinsics.target = [0, 0, 0]


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
