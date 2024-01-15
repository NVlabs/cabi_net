# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Standard Library
import argparse
import os
import sys
from typing import List

# Third Party
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

# NVIDIA
from cabi_net.config.waypoint import get_cfg_defaults
from cabi_net.utils import get_timestamp

os.environ["PYOPENGL_PLATFORM"] = "egl"


def setup_checkpointing(checkpoint_path, metric_to_monitor="train_loss", every_n_epochs=1):
    callbacks: List[pl.callbacks.Callback] = []
    every_n_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor=metric_to_monitor,
        save_last=True,
        dirpath=checkpoint_path,
        every_n_epochs=every_n_epochs,
    )
    epoch_end_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor=metric_to_monitor,
        save_last=True,
        dirpath=checkpoint_path,
        save_on_train_epoch_end=True,
    )
    epoch_end_checkpoint.CHECKPOINT_NAME_LAST = "epoch-end"
    callbacks.extend([every_n_checkpoint, epoch_end_checkpoint])
    pl.utilities.rank_zero_info(f"Saving checkpoints to {checkpoint_path}")
    return callbacks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CabiNet Waypoint training")
    parser.add_argument(
        "--cfg-file",
        help="yaml file in YACS config format to override default configs",
        default="",
        type=str,
    )
    parser.add_argument(
        "--resume-ckpt-path",
        help="Checkpoint to resume training from",
        default="",
        type=str,
    )
    args = parser.parse_args()

    cfg = get_cfg_defaults()

    if args.cfg_file != "":
        if os.path.exists(args.cfg_file):
            cfg.merge_from_file(args.cfg_file)
        else:
            raise FileNotFoundError(args.cfg_file)

    resume_training_from_checkpoint = False
    if args.resume_ckpt_path != "":
        if not os.path.exists(args.resume_ckpt_path):
            raise FileNotFoundError(args.resume_ckpt_path)
        print(f"CAUTION!! Resuming checkpoint training from {args.resume_ckpt_path}")
        resume_training_from_checkpoint = True

    if torch.cuda.is_available():
        print("Cuda is available, make sure you are training on GPU")

    cfg.freeze()
    print(cfg)

    exp_name = f"{cfg.model.name}_{get_timestamp()}"
    log_dir = os.path.join(cfg.model.path, exp_name)
    checkpoint_path = os.path.join(log_dir, "weights")

    # Setup checkpointing
    callbacks = setup_checkpointing(checkpoint_path, every_n_epochs=cfg.trainer.every_n_epochs)

    device = torch.device(f"cuda:{0}")

    # Setup Model
    ModelCls = getattr(sys.modules[__name__], cfg.model.model_arch)
    model = ModelCls(cfg, device=device)

    # Setup Logging
    logger = TensorBoardLogger(cfg.model.path, exp_name)

    # Setup trainer
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        enable_checkpointing=True,
        max_epochs=cfg.trainer.max_epochs,
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        limit_val_batches=cfg.trainer.limit_val_batches,
        logger=logger,
        limit_train_batches=cfg.trainer.limit_train_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        precision=16,
    )

    if resume_training_from_checkpoint:
        trainer.fit(model, ckpt_path=args.resume_ckpt_path)
    else:
        trainer.fit(model)
