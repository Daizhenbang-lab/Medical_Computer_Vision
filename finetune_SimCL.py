import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from lightly.data import collate, LightlyDataset
from PIL import Image
import numpy as np
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss
import sys
from pytorch_lightning.strategies.ddp import DDPStrategy
#from stainlib.augmentation.augmenter import HedColorAugmenter1
import torchvision.transforms as T
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from lightly.transforms import RandomRotate
from utils.model import ImageCollateFunction, SimCLRModel

path_to_data = 'sample_56_overlap0.6/sample'
subfolders = ['scan17', 'scan62', 'scan121']
path_to_model = 'tenpercent_resnet18.ckpt'

num_workers = 4
batch_size = 128
seed = 1
input_size = 56
max_epochs = 25
pl.seed_everything(seed)

imagenet_normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

if __name__ == '__main__':
    #train_dirs = [os.path.join(path_to_data, subfolder) for subfolder in subfolders]

    collate_fn = ImageCollateFunction(input_size=input_size,
                                      min_scale=0.25,
                                      vf_prob=0.5,
                                      hf_prob=0.5,
                                      rr_prob=0.5,
                                      hed_thresh=0.3)

    gpus = torch.cuda.device_count()
    print(gpus)

    model = SimCLRModel()

    for name, p in model.named_parameters():
        if "backbone" in name:
            p.requires_grad = False


    dataset_train_simclr = LightlyDataset(input_dir=path_to_data)
    dataloader_train_simclr = torch.utils.data.DataLoader(
        dataset_train_simclr,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=True
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss_ssl",
        dirpath="checkpoints_size56",
        filename="simclr-{epoch:02d}-{train_loss_ssl:.2f}",
        save_top_k=3,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=gpus,
        num_nodes=1,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, dataloader_train_simclr)

