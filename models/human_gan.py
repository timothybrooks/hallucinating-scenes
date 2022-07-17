from __future__ import annotations

__all__ = ["HumanGAN"]

import copy
import os
from typing import Any, Iterator, Optional

import data
import einops
from pytorch_lightning import LightningModule
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import utils

from . import networks
from .layers import nvidia_ops


class HumanGAN(LightningModule):
    def __init__(
        self,
        resolution: int,
        num_keypoints: int,
        num_frames: int,
        learning_rate: float = 0.0025,
        mixing: float = 0.0,
        fake_pair: bool = True,
        ema_beta: float = 0.995,
        ema_warmup: float = 0.05,
        lambda_r1: float = 0.01,
        lambda_path: float = 2.0,
        batch_shrink_path: int = 2,
        interval_r1: Optional[int] = 16,
        interval_path: Optional[int] = 4,
        interval_log_frames: int = 5000,
        kwargs_a: dict[str, Any] = {},
        kwargs_d: dict[str, Any] = {},
        kwargs_g: dict[str, Any] = {},
    ):
        assert interval_r1 is None or interval_r1 > 0
        assert interval_path is None or interval_path > 0

        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.mixing = mixing
        self.fake_pair = fake_pair
        self.ema_beta = ema_beta
        self.ema_warmup = ema_warmup
        self.lambda_r1 = lambda_r1
        self.lambda_path = lambda_path
        self.batch_shrink_path = batch_shrink_path
        self.interval_r1 = interval_r1
        self.interval_path = interval_path
        self.interval_log_frames = interval_log_frames

        self.augment = data.AugmentImageAndKeypoints(**kwargs_a)

        self.discriminator = networks.KeypointDiscriminator(
            resolution, num_keypoints, num_frames, **kwargs_d
        )

        self.generator = networks.Generator(
            resolution, num_keypoints, num_frames, **kwargs_g
        )
        self.generator_ema = copy.deepcopy(self.generator)
        self.generator_ema.eval().requires_grad_(False)

        self._loss_r1 = 0
        self._loss_path = 0

        self.register_buffer("_path_length_ema", torch.zeros(()))
        self._path_length_ema: torch.Tensor

        self.logger: utils.Logger

    @classmethod
    def load_from_id(cls, id: str, step: Optional[int] = None) -> HumanGAN:
        checkpoint_path = cls._get_checkpoint_path(id, step)
        print(f"Loading checkpoint: {checkpoint_path}")
        model = cls.load_from_checkpoint(checkpoint_path, map_location="cpu")
        return model

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        optimizer_idx: int,
    ):
        real_frames, keypoints = batch

        self._discriminator_step(real_frames, keypoints)
        self._generator_step(keypoints)

        if self.global_step % self.interval_log_frames == 0:
            self._log_frames(real_frames, keypoints)

    def configure_optimizers(self) -> tuple[optim.Adam, optim.Adam]:
        optimizer_d = self._get_optimizer(
            self.discriminator.parameters(), self.interval_r1
        )
        optimizer_g = self._get_optimizer(
            self.generator.parameters(), self.interval_path
        )
        return optimizer_d, optimizer_g

    @property
    def automatic_optimization(self) -> bool:
        return False

    # ==========================================================================
    # Discriminator step.
    # ==========================================================================

    def _discriminator_step(
        self, real_frames: torch.Tensor, keypoints: torch.Tensor
    ):
        optimizer_idx = 0
        optimizer: optim.Adam = self.optimizers()[optimizer_idx]  # type: ignore
        self.toggle_optimizer(optimizer, optimizer_idx)

        with torch.no_grad():
            time_indices = self._sample_indices(keypoints)

            real_frames_subset = self._index_time(real_frames, time_indices)
            frame_keypoints = self._index_time(keypoints, time_indices)

            fake_frames = self.generator(
                keypoints, frame_keypoints, self.mixing
            )

            real_frames_augment, real_keypoints_augment = self.augment(
                real_frames_subset, keypoints
            )
            fake_frames_augment, fake_keypoints_augment = self.augment(
                fake_frames, keypoints
            )

        real_frame_keypoints_augment = self._index_time(
            real_keypoints_augment, time_indices
        )
        loss_real = self.discriminator.loss_real(
            real_frames_augment,
            real_keypoints_augment,
            real_frame_keypoints_augment,
        )

        fake_frame_keypoints_augment = self._index_time(
            fake_keypoints_augment, time_indices
        )
        loss_fake = self.discriminator.loss_fake(
            fake_frames_augment,
            fake_keypoints_augment,
            fake_frame_keypoints_augment,
        )

        loss = loss_real + loss_fake

        if self.fake_pair:

            loss_fake_pair = self.discriminator.loss_fake(
                real_frames_augment,
                real_keypoints_augment.flip(dims=(0,)),
                real_frame_keypoints_augment.flip(dims=(0,)),
            )
            loss = loss + loss_fake_pair

        optimizer.zero_grad()
        self.manual_backward(loss, optimizer)
        optimizer.step()

        self.log("loss_d", loss, prog_bar=True)
        self.log("loss_d_real", loss_real)
        self.log("loss_d_fake", loss_fake)

        if self.fake_pair:
            self.log("loss_d_fake_pair", loss_fake_pair)

        if self.interval_r1 is not None:

            if self.global_step % self.interval_r1 == 0:

                loss_r1_weighted = self._discriminator_reg(
                    real_frames, keypoints
                )

                optimizer.zero_grad()
                self.manual_backward(loss_r1_weighted, optimizer)
                optimizer.step()

            self.log("loss_r1", self._loss_r1)

        self.untoggle_optimizer(optimizer_idx)

    def _discriminator_reg(
        self, real_frames: torch.Tensor, keypoints: torch.Tensor
    ) -> torch.Tensor:

        assert self.interval_r1 is not None

        time_indices = self._sample_indices(keypoints)
        real_frames = self._index_time(real_frames, time_indices)

        real_frames.requires_grad = True

        real_frames_augment, real_keypoints_augment = self.augment(
            real_frames, keypoints
        )

        real_frame_keypoints_augment = self._index_time(
            real_keypoints_augment, time_indices
        )
        real_prediction = self.discriminator(
            real_frames_augment,
            real_keypoints_augment,
            real_frame_keypoints_augment,
        )

        with nvidia_ops.no_weight_gradients():
            (grads_r1,) = autograd.grad(
                outputs=real_prediction.sum(),
                inputs=real_frames,
                create_graph=True,
            )

        loss_r1 = grads_r1.square().sum(dim=(1, 2, 3, 4)).mean()
        self._loss_r1 = loss_r1.detach()

        loss_r1 = loss_r1 * (self.lambda_r1 / 2) * self.interval_r1
        loss_r1 = loss_r1 + 0 * real_prediction[0]

        return loss_r1

    # ==========================================================================
    # Generator step.
    # ==========================================================================

    def _generator_step(self, keypoints: torch.Tensor):
        optimizer_idx = 1
        optimizer: optim.Adam = self.optimizers()[optimizer_idx]  # type: ignore
        self.toggle_optimizer(optimizer, optimizer_idx)

        time_indices = self._sample_indices(keypoints)
        frame_keypoints = self._index_time(keypoints, time_indices)
        fake_frames = self.generator(keypoints, frame_keypoints, self.mixing)

        fake_frames_augment, fake_keypoints_augment = self.augment(
            fake_frames, keypoints
        )

        fake_frame_keypoints_augment = self._index_time(
            fake_keypoints_augment, time_indices
        )
        loss = self.discriminator.loss_real(
            fake_frames_augment,
            fake_keypoints_augment,
            fake_frame_keypoints_augment,
        )

        optimizer.zero_grad()
        self.manual_backward(loss, optimizer)
        optimizer.step()

        self.log("loss_g", loss, prog_bar=True)

        if self.interval_path is not None:

            if self.global_step % self.interval_path == 0:

                loss_path = self._generator_reg_step(keypoints)

                optimizer.zero_grad()
                self.manual_backward(loss_path, optimizer)
                optimizer.step()

            self.log("loss_path", self._loss_path)

        self._generator_ema_step()
        self.untoggle_optimizer(optimizer_idx)

    def _generator_reg_step(self, keypoints: torch.Tensor) -> torch.Tensor:
        assert self.interval_path is not None

        batch_size_path = max(1, keypoints.size(0) // self.batch_shrink_path)
        keypoints = keypoints[:batch_size_path]

        styles = self.generator.mapping_network(keypoints, self.mixing)

        time_indices = self._sample_indices(keypoints)
        frame_keypoints = self._index_time(keypoints, time_indices)

        num_frames = frame_keypoints.size(1)
        styles = einops.repeat(styles, "n s c -> (n t) s c", t=num_frames)

        frame_keypoints = einops.rearrange(
            frame_keypoints, "n t k c -> (n t) k c"
        )

        fake_frames = self.generator.synthesis_network(styles, frame_keypoints)

        noise = torch.randn_like(fake_frames) / self.generator.resolution

        with nvidia_ops.no_weight_gradients():
            (grads_path,) = autograd.grad(
                outputs=(fake_frames * noise).sum(),
                inputs=styles,
                create_graph=True,
            )

        path_length = grads_path.square().sum(dim=2).mean(dim=1).sqrt()

        decay = 0.01
        path_length_ema = self._path_length_ema.lerp(path_length.mean(), decay)
        self._path_length_ema = path_length_ema.detach()

        loss_path = (path_length - path_length_ema).square().mean()
        self._loss_path = loss_path.detach()

        loss_path = loss_path * self.lambda_path * self.interval_path
        loss_path = loss_path + 0 * fake_frames[0, 0, 0, 0]

        return loss_path

    @torch.no_grad()
    def _generator_ema_step(self):
        if self.global_step == 0:
            ema_beta = 0.0

        else:
            ema_beta_warmup = 0.5 ** (1 / (self.global_step * self.ema_warmup))
            ema_beta = min(self.ema_beta, ema_beta_warmup)

        def _ema(tensors_ema, tensors):
            for tensor_ema, tensor in zip(tensors_ema, tensors):
                tensor_ema.copy_(tensor.lerp(tensor_ema, ema_beta))

        _ema(self.generator_ema.parameters(), self.generator.parameters())
        _ema(self.generator_ema.buffers(), self.generator.buffers())

    # ==========================================================================
    # Utilities.
    # ==========================================================================

    def _get_optimizer(
        self,
        parameters: Iterator[nn.Parameter],
        interval_reg: Optional[int],
        beta_1: float = 0,
        beta_2: float = 0.99,
    ) -> optim.Adam:

        learning_rate = self.learning_rate

        if interval_reg is not None:
            ratio_reg = interval_reg / (interval_reg + 1)
            learning_rate *= ratio_reg
            beta_1 **= ratio_reg
            beta_2 **= ratio_reg

        optimizer = optim.Adam(
            parameters, lr=learning_rate, betas=(beta_1, beta_2)
        )
        return optimizer

    def _sample_indices(self, input: torch.Tensor) -> torch.Tensor:
        weights = torch.ones(*input.shape[:2], device=self.device)
        time_indices = torch.multinomial(weights, num_samples=1)
        return time_indices

    def _index_time(
        self, input: torch.Tensor, time_indices: torch.Tensor
    ) -> torch.Tensor:

        assert time_indices.dim() == 2
        assert time_indices.size(0) == input.size(0)

        output = []
        for i in range(input.size(0)):
            output.append(input[i, time_indices[i]])

        output = torch.stack(output)
        return output

    @staticmethod
    def _get_checkpoint_path(id: str, step: Optional[int] = None) -> str:
        checkpoint_dir = os.path.join("checkpoints/gan", id)

        if os.path.exists(checkpoint_dir):
            checkpoint_filenames = os.listdir(checkpoint_dir)

            if len(checkpoint_filenames) > 0:

                checkpoint_filename = None

                if step is not None:
                    for filename in checkpoint_filenames:
                        if filename == f"step={step - 1:08d}.ckpt":
                            checkpoint_filename = filename
                            break

                    if checkpoint_filename is None:
                        raise ValueError(
                            f"No checkpoint exists for step {step}: {id}"
                        )

                else:
                    checkpoint_filename = sorted(checkpoint_filenames)[-1]

                checkpoint_path = os.path.join(
                    checkpoint_dir, checkpoint_filename  # type: ignore
                )
                return checkpoint_path

        raise ValueError(f"No checkpoints exist: {id}")

    @torch.no_grad()
    def _log_frames(self, real_frames: torch.Tensor, keypoints: torch.Tensor):

        real_frames = real_frames[:4]
        keypoints = keypoints[:4]

        num_frames = real_frames.size(1)

        self.generator_ema.eval()
        fake_frames = self.generator_ema(keypoints, keypoints)

        real_frames = einops.rearrange(real_frames, "n t c h w -> (n t) c h w")
        fake_frames = einops.rearrange(fake_frames, "n t c h w -> (n t) c h w")
        keypoints = einops.rearrange(keypoints, "n t k c -> (n t) k c")

        real_frames_pose = utils.draw_skeleton(real_frames, keypoints)
        fake_frames_pose = utils.draw_skeleton(fake_frames, keypoints)

        frames = {
            "real_frames": real_frames,
            "fake_frames": fake_frames,
            "real_frames_pose": real_frames_pose,
            "fake_frames_pose": fake_frames_pose,
        }

        for name, frames in frames.items():
            frames = self.all_gather(frames)
            frames = einops.rearrange(frames, "g n c h w -> (g n) c h w")
            self.logger.log_image(name, frames, nrow=num_frames)
