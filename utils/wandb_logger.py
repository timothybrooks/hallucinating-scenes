from __future__ import annotations

__all__ = ["Logger"]

import logging
import os
import pathlib
from typing import Optional

import data
import omegaconf
import torch
import torchvision.utils as utils
import wandb
from pytorch_lightning import LightningModule
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


def is_rank_zero():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank == 0


class Logger(WandbLogger):
    def __init__(
        self,
        *,
        name: str,
        project: str,
        entity: str,
        group: Optional[str] = None,
        offline: bool = False,
    ):
        logging.getLogger("lightning").handlers = []

        formatter = logging.Formatter(
            fmt="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d,%H:%M:%S",
        )
        logging.getLogger().handlers[0].setFormatter(formatter)

        # Automatically increments run index.
        wandb_runs = wandb.Api().runs(
            path=f"{entity}/{project}", order="-created_at"
        )

        try:
            prev_name = wandb_runs[0].name
            run_index = int(prev_name.split("_")[0]) + 1
        except:
            run_index = 0

        name = f"{run_index:04d}_{name}"
        save_dir = str(pathlib.Path(__file__).parents[1])

        super().__init__(
            name=name,
            save_dir=save_dir,
            offline=offline,
            project=project,
            log_model=False,
            entity=entity,
            group=group,
        )

    def log_hyperparams(self, *args, **kwargs) -> None:
        pass

    @rank_zero_only
    def log_config(self, config: omegaconf.DictConfig):

        if is_rank_zero():
            hydra_config = omegaconf.OmegaConf.to_yaml(config)

            filename = "hydra_config.yaml"
            self.experiment.save(filename)

            path = os.path.join(self.experiment.dir, filename)
            with open(path, "w") as file:
                print(hydra_config, file=file)

            params = omegaconf.OmegaConf.to_container(config)
            assert isinstance(params, dict)
            params.pop("wandb", None)

            self.experiment.config.update(params)

    @rank_zero_only
    def log_model_summary(self, model: LightningModule):

        if is_rank_zero():
            summary = ModelSummary(model, mode=ModelSummary.MODE_FULL)

            filename = "model_summary.txt"
            self.experiment.save(filename)

            path = os.path.join(self.experiment.dir, filename)
            with open(path, "w") as file:
                print(summary, file=file)

    @torch.no_grad()
    @rank_zero_only
    def log_image(self, name: str, image: torch.Tensor, **kwargs):

        if is_rank_zero():
            image_grid = utils.make_grid(
                image, normalize=True, value_range=(-1, 1), **kwargs
            )
            wandb_image = wandb.Image(image_grid.cpu())
            self.experiment.log({name: wandb_image}, commit=False)
