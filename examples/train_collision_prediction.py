# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Standard Library
import argparse
import logging
import os
import os.path as osp
import shutil
import sys

# Third Party
import numpy as np
import torch
import torch.nn as nn
import tqdm
from autolab_core import BinaryClassificationResult, YamlConfig, keyboard_input
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# NVIDIA
from cabi_net.dataset.dataset import CollisionDataset, process_batch
from cabi_net.errors import ColllisionRatioError
from cabi_net.meshcat_utils import create_visualizer
from cabi_net.model.cabinet import CabiNetCollision
from cabi_net.robot.robot import Robot
from cabi_net.utils import plot_scene_data

os.environ["PYOPENGL_PLATFORM"] = "egl"
logging.getLogger("yourdfpy").setLevel(logging.ERROR)


class Trainer(object):
    def __init__(
        self,
        model,
        optimizer,
        scaler,
        criterion,
        train_loader,
        train_iterations,
        val_iterations,
        loss_pct,
        out,
        camera_info=None,
    ):
        self.model = model
        self.optim = optimizer
        self.scaler = scaler
        self.criterion = criterion

        self.data_loader = iter(train_loader)
        self.train_iterations = train_iterations
        self.val_iterations = val_iterations
        self.loss_pct = loss_pct

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = 1000000
        self.best_ap = 0.0
        self.writer = SummaryWriter(log_dir=self.out)
        self.camera_info = camera_info
        self.max_resample_attempts = 20
        self.model_error_counter = 0
        self.model_error_max = 100

    def get_data(self):
        counter = 0
        while True:
            counter += 1
            try:
                data = next(self.data_loader)
            except Exception as e:
                print(f"Caught error {counter} in data loader: {e}")
                data = None
            if data is not None:
                return data

            if counter >= self.max_resample_attempts:
                print("Exceeded the maximum number of data resampling attempts...")
                sys.exit(1)

    def validate(self):
        self.model.eval()

        val_preds, val_trues, val_type, val_loss = (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

        for _ in tqdm.trange(
            self.val_iterations,
            desc="Valid iteration={:d}".format(self.iteration),
            leave=False,
        ):
            data = self.get_data()
            img, _, obj, _, trans, rots, coll, obj_type, _, _ = process_batch(data)

            result, out = self.forward(img, obj, trans, rots, train=False)

            if not result:
                continue

            coll = coll.unsqueeze(0).expand(out.shape)
            losses = self.criterion(out, coll.float())

            batch_preds = torch.sigmoid(out)

            batch_size = batch_preds.cpu().numpy().flatten().shape[0]
            val_preds = np.append(val_preds, batch_preds.cpu().numpy().flatten())
            val_trues = np.append(val_trues, coll.float().cpu().numpy().flatten())

            val_type = np.append(val_type, np.repeat(obj_type, batch_size))
            val_loss = np.append(val_loss, losses.cpu().numpy().flatten())

        val_loss_mean = np.mean(val_loss)
        bcr = BinaryClassificationResult(val_preds, val_trues)

        self.writer.add_scalar("Val/loss", val_loss_mean, self.epoch)
        self.writer.add_scalar("Val/accuracy", bcr.accuracy, self.epoch)
        self.writer.add_scalar("Val/ap", bcr.ap_score, self.epoch)

        # Aggregate object results
        indices = np.where(val_type == -1)[0]
        mesh_name = "overall_objects"
        if len(indices) > 0:
            bcr = BinaryClassificationResult(val_preds[indices], val_trues[indices])
            self.tensorboard_log(
                f"Val/{mesh_name}",
                np.mean(val_loss[indices]),
                bcr.accuracy,
                bcr.ap_score,
                self.epoch,
            )

        is_best = bcr.ap_score > self.best_ap
        save_dict = {
            "epoch": self.epoch + 1,
            "iteration": self.iteration,
            "arch": self.model.__class__.__name__,
            "optim_state_dict": self.optim.state_dict(),
            "model_state_dict": self.model.state_dict(),
            "best_ap": self.best_ap,
        }
        torch.save(save_dict, osp.join(self.out, "checkpoint.pth.tar"))

        if is_best:
            shutil.copy(
                osp.join(self.out, "checkpoint.pth.tar"),
                osp.join(self.out, "model_best.pth.tar"),
            )

    def forward(self, img, obj, trans, rots, train=True):
        if train:
            with autocast():
                out = self.model(img, obj, trans, rots).squeeze(dim=-1)
        else:
            with torch.no_grad():
                out = self.model(img, obj, trans, rots).squeeze(dim=-1)
        return True, out

    def train_epoch(self):
        self.model.train()

        train_bar = tqdm.trange(
            self.train_iterations,
            desc="Train epoch={:d}".format(self.epoch),
            leave=False,
        )

        for batch_idx in train_bar:
            iteration = batch_idx + self.epoch * self.train_iterations
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            assert self.model.training

            data = self.get_data()

            (
                img,
                _,
                obj,
                _,
                trans,
                rots,
                coll,
                obj_type,
                camera_poses,
                _,
            ) = process_batch(data)

            self.optim.zero_grad()

            result, out = self.forward(img, obj, trans, rots, train=True)

            if not result:
                continue

            batch_size = trans.shape[0]
            collision_ratio = float(coll.cpu().numpy().sum()) / batch_size
            coll = coll.unsqueeze(0).expand(out.shape)
            losses = self.criterion(out, coll.float())
            top_losses, _ = torch.topk(losses, int(losses.size(1) * self.loss_pct), sorted=False)
            rand_losses = losses[
                :, torch.randint(losses.size(1), (int(losses.size(1) * self.loss_pct),))
            ]

            loss = 0.5 * (top_losses.mean() + rand_losses.mean())
            loss_data = loss.item()
            if torch.isnan(loss.data):
                raise ValueError("loss is nan while training")
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            lr = self.optim.param_groups[0]["lr"]

            with torch.no_grad():
                batch_preds = torch.sigmoid(out)
                batch_trues = coll

                bcr = BinaryClassificationResult(
                    batch_preds.data.cpu().numpy().flatten(),
                    batch_trues.data.cpu().numpy().flatten(),
                )
            train_bar.set_postfix_str("Loss: {:.5f}".format(loss_data))

            self.writer.add_scalar("Train/lr", lr, iteration)
            self.writer.add_scalar("Train/collision_ratio", collision_ratio, iteration)
            self.writer.add_scalar("Train/loss_batch_mean", losses.mean().item(), iteration)
            self.tensorboard_log(
                f"Train",
                loss_data,
                bcr.accuracy,
                bcr.ap_score,
                iteration,
            )

            if obj_type == -1:
                mesh_name = "overall_objects"
                self.tensorboard_log(
                    f"Train/{mesh_name}",
                    loss_data,
                    bcr.accuracy,
                    bcr.ap_score,
                    iteration,
                )

    def tensorboard_log(self, prefix, loss, accuracy, ap, time_step):
        self.writer.add_scalar(f"{prefix}/loss", loss, time_step)
        self.writer.add_scalar(f"{prefix}/accuracy", accuracy, time_step)
        self.writer.add_scalar(f"{prefix}/ap", ap, time_step)

    def train(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch, desc="Train"):
            self.epoch = epoch
            self.train_epoch()
            self.validate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CabiNet Collision module")
    parser.add_argument(
        "--cfg",
        type=str,
        default="cfg/train.yaml",
        help="config file with training params",
    )
    parser.add_argument("--resume", action="store_true", help="resume training")
    args = parser.parse_args()

    # Replace config with args
    config = YamlConfig(args.cfg)

    resume = args.resume

    # Create output directory for model
    out = osp.join(config["model"]["path"], config["model"]["name"])

    if osp.exists(out) and not resume:
        response = keyboard_input(
            "A model exists at {}. Would you like to overwrite?".format(out),
            yesno=True,
        )
        if response.lower() == "n":
            sys.exit(0)
    elif osp.exists(out) and resume:
        print(f"Resuming training from checkpoint at {out}")
        resume = resume and osp.exists(osp.join(out, "checkpoint.pth.tar"))
    else:
        resume = False
        os.makedirs(out)
    config.save(osp.join(out, "train.yaml"))

    # 1. Model
    start_epoch = 0
    start_iteration = 0

    device = torch.device(f"cuda:{0}")
    model = CabiNetCollision(
        config=config,
        device=device,
    )

    if resume:
        checkpoint = torch.load(osp.join(out, "checkpoint.pth.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        start_iteration = checkpoint["iteration"]
        print(
            f"Loaded checkpoint and opt vals from {out}. Start epoch {start_epoch}, start iteration"
            f" {start_iteration}"
        )
    model = model.cuda()

    num_workers = config["trainer"]["num_workers"]
    num_workers = os.cpu_count() if num_workers == -1 else num_workers
    print(f"DataLoader is using {num_workers} workers")

    # 2. Dataset
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "worker_init_fn": lambda _: np.random.seed(),
    }

    assert not config["dataset"]["test"]

    train_set = CollisionDataset(
        **config["dataset"],
        **config["camera"],
        bounds=config["model"]["bounds"],
    )
    train_loader = DataLoader(train_set, batch_size=None, **kwargs)

    # 3. Optimizer and loss
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["trainer"]["lr"],
        momentum=config["trainer"]["momentum"],
    )
    if resume:
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
    scaler = GradScaler()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        criterion=criterion,
        train_loader=train_loader,
        train_iterations=config["trainer"]["train_iterations"],
        val_iterations=config["trainer"]["val_iterations"],
        loss_pct=config["trainer"]["loss_pct"],
        out=out,
        camera_info=config["camera"],
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()
