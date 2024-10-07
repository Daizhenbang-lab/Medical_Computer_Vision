import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from lightly.data import collate, LightlyDataset
from PIL import Image
import numpy as np
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from lightly.transforms import RandomRotate
import torchvision.transforms as T
from stainlib.augmentation.augmenter import HedColorAugmenter1

path_to_data = 'training_set'  # 替换为实际的 ImageNet 数据集路径
num_workers = 4
batch_size = 256  # 使用更大的批次大小以适应 ImageNet 数据集
seed = 1
max_epochs = 90
input_size = 224

pl.seed_everything(seed)

class HedColorAug:
    def __init__(self, hed_thresh=0.03):
        self.hed_thresh = hed_thresh

    def __call__(self, image):
        dab_lighter_aug = HedColorAugmenter1(self.hed_thresh)
        dab_lighter_aug.randomize()
        return Image.fromarray(dab_lighter_aug.transform(np.array(image)))

imagenet_normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

class ImageCollateFunction(collate.BaseCollateFunction):
    def __init__(self,
                 input_size: int = 224,
                 min_scale: float = 0.08,
                 vf_prob: float = 0.0,
                 hf_prob: float = 0.5,
                 rr_prob: float = 0.0,
                 hed_thresh: float = 0.3,
                 normalize: dict = imagenet_normalize):

        transform = [T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
                     RandomRotate(prob=rr_prob),
                     T.RandomHorizontalFlip(p=hf_prob),
                     T.RandomVerticalFlip(p=vf_prob),
                     HedColorAug(hed_thresh=hed_thresh),
                     T.ToTensor(),
                     T.Normalize(mean=normalize['mean'], std=normalize['std'])
                     ]

        transform = T.Compose(transform)
        super(ImageCollateFunction, self).__init__(transform)

class SimCLRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()


        resnet = torchvision.models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)
        self.criterion = NTXentLoss(gather_distributed=True)

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

if __name__ == '__main__':
    collate_fn = ImageCollateFunction(input_size=input_size, min_scale=0.2, vf_prob=0.5, hf_prob=0.5, rr_prob=0.5, hed_thresh=0.3)

    gpus = torch.cuda.device_count()
    print(f'Number of GPUs: {gpus}')

    model = SimCLRModel()

    # Freeze backbone layers
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False

    # Define the ImageNet dataset and data loader
    dataset_train_simclr = torchvision.datasets.ImageFolder(root=os.path.join(path_to_data, 'train'),
                                                            transform=collate_fn)
    dataloader_train_simclr = torch.utils.data.DataLoader(dataset_train_simclr,
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          num_workers=num_workers,
                                                          drop_last=True)

    checkpoint_callback = ModelCheckpoint(monitor="train_loss_ssl",
                                          dirpath="checkpoints",
                                          filename="imagenet-{epoch:02d}-{train_loss_ssl:.2f}",
                                          save_top_k=3,
                                          mode="min")

    trainer = pl.Trainer(max_epochs=max_epochs,
                         accelerator='gpu',
                         devices=gpus,
                         num_nodes=1,
                         callbacks=[checkpoint_callback])

    trainer.fit(model, dataloader_train_simclr)
