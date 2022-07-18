import pathlib

import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import data
import models
import utils


@hydra.main(config_path="configs/run_train", config_name="ours")
def train(config: omegaconf.DictConfig):

    logger = utils.Logger(**config.wandb)
    logger.log_config(config)

    dataset = data.HumansDataset(**config.dataset)
    data_loader = DataLoader(dataset, **config.data_loader)

    if config.resume.id is not None:
        checkpoint = models.HumanGAN._get_checkpoint_path(**config.resume)
        model = models.HumanGAN.load_from_checkpoint(checkpoint)

    else:
        checkpoint = None
        model = models.HumanGAN(
            dataset.resolution,
            dataset.num_keypoints,
            dataset.num_frames,
            kwargs_a=config.augmentation,
            kwargs_d=config.discriminator,
            kwargs_g=config.generator,
            **config.model,
        )

    logger.log_model_summary(model)

    dirpath = pathlib.Path("checkpoints/gan", str(logger.version))
    checkpoint_callback = ModelCheckpoint(dirpath, filename="{step:08d}", save_top_k=-1, every_n_train_steps=10000)

    trainer = pl.Trainer(
        resume_from_checkpoint=checkpoint, logger=logger, callbacks=[checkpoint_callback], **config.trainer,
    )
    trainer.fit(model, data_loader)


if __name__ == "__main__":
    train()
