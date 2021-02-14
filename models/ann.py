import argparse
import pathlib
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import pytorch_lightning as pl
import torchvision
import tqdm

import norse


class DVSModel(pl.LightningModule):
    def __init__(
        self,
        n_class,
        n_channels,
        height,
        width,
        iter_per_frame,
        class_weights=None,
        method="super",
        alpha=100,
        dt=0.001,
    ):
        super(DVSModel, self).__init__()
        self.n_class = n_class
        self.height = height
        self.width = width
        self.iter_per_frame = iter_per_frame
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        # block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/2
            nn.BatchNorm2d(64),
        )

        # block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.BatchNorm2d(128),
        )

        # block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.BatchNorm2d(256),
        )

        # dense
        self.dense = nn.Sequential(
            nn.Conv2d(256, 256, 7, padding=3, bias=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        self.score_block2 = nn.Conv2d(128, n_class, 1, bias=False)
        self.deconv_block2 = nn.ConvTranspose2d(
            n_class, n_class, 8, stride=4, padding=2, bias=False
        )

        self.score_dense = nn.Conv2d(256, n_class, 1, bias=False)
        self.deconv_dense = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, padding=4, bias=False
        )

        self.final = nn.ReLU()

    def forward(self, x):
        # output              batch,      class,        frame,      height,     width
        output = torch.empty(
            (x.shape[0], self.n_class, x.shape[1], x.shape[3], x.shape[4]),
            device=self.device,
        )

        # for each frame
        for i in range(x.shape[1]):
            frame = x[:, i, :, :, :]

            for j in range(self.iter_per_frame):

                out_block1 = self.block1(frame)  # 1/2
                out_block2 = self.block2(out_block1)  # 1/4
                out_block3 = self.block3(out_block2)  # 1/8
                out_dense = self.dense(out_block3)

                ####### WITH FEATURE FUSION
                out_score_block2 = self.score_block2(out_block2)
                out_deconv_block2 = self.deconv_block2(out_score_block2)

                # out_score_dense = checkpoint(self.score_dense, out_dense)
                out_score_dense = self.score_dense(out_dense)
                out_deconv_dense = self.deconv_dense(out_score_dense)

                out_deconv = out_deconv_block2 + out_deconv_dense
                #######

                out_final = self.final(out_deconv)

            output[:, :, i, :, :] = out_final

        return output

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        x, y = batch
        z = self.forward(x)

        loss = self.loss_fn(z, y)
        # Log the loss
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-03, weight_decay=1e-5)
