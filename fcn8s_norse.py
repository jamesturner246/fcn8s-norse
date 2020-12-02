import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.checkpoint import checkpoint

from norse.torch.module.sequential import SequentialState
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.lif import LIFFeedForwardCell
from norse.torch.functional.leaky_integrator import LIParameters
from norse.torch.module.leaky_integrator import LIFeedForwardCell
from norse.torch.module.leaky_integrator import LICell

import pytorch_lightning as pl


class FCN8s(pl.LightningModule):
    def __init__(
        self,
        n_class,
        height,
        width,
        timesteps,
        encoder,
        loss_fn,
        dt=0.001,
        method="super",
        alpha=100.0,
        checkpoint=True,
    ):
        super(FCN8s, self).__init__()
        self.n_class = n_class
        self.height = height
        self.width = width
        self.dt = dt
        self.method = method
        self.alpha = alpha
        self.timesteps = timesteps
        self.encoder = encoder
        self.loss_fn = loss_fn

        p_lif = LIFParameters(

            # tau_syn_inv=torch.tensor(200.0),
            # tau_mem_inv=torch.tensor(200.0),

            tau_syn_inv=torch.tensor(12.5),
            tau_mem_inv=torch.tensor(6.25),

            v_leak=torch.tensor(0.0),
            v_th=torch.tensor(1.0),
            v_reset=torch.tensor(0.0),

            method = 'super',
            #alpha = 0.0,
            alpha = 100.0,

        )

        p_li = LIParameters(

            # tau_syn_inv=torch.tensor(200.0),
            # tau_mem_inv=torch.tensor(100.0),

            tau_syn_inv=torch.tensor(12.5),
            tau_mem_inv=torch.tensor(6.25),

            v_leak=torch.tensor(0.0),

        )

        # block 1
        self.block1 = SequentialState(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=p_lif, dt=dt),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/2
        )

        # block 2
        self.block2 = SequentialState(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=p_lif, dt=dt),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/4
        )

        # block 3
        self.block3 = SequentialState(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=p_lif, dt=dt),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=p_lif, dt=dt),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/8
        )

        # block 4
        self.block4 = SequentialState(
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=p_lif, dt=dt),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=p_lif, dt=dt),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.BatchNorm2d(512),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/16
        )

        # block 5
        self.block5 = SequentialState(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=p_lif, dt=dt),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=p_lif, dt=dt),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.BatchNorm2d(512),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/32
        )

        # dense
        self.dense = SequentialState(
            nn.Conv2d(512, 1024, 7, padding=3, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            # nn.Dropout2d(),
            nn.Conv2d(1024, 1024, 1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.BatchNorm2d(1024),
            # nn.Dropout2d(),
        )

        self.score_block3 = nn.Conv2d(256, n_class, 1, bias=False)
        self.score_block4 = nn.Conv2d(512, n_class, 1, bias=False)
        self.score_dense = nn.Conv2d(1024, n_class, 1, bias=False)

        self.upscore_2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, padding=1, bias=False
        )
        self.upscore_block4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, padding=1, bias=False
        )
        self.upscore_8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, padding=4, bias=False
        )

        # self.final = LICell(n_class, n_class, dt=dt)
        self.final = LIFeedForwardCell(dt=dt)
        # self.final = LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-04)

    def forward(self, x):

        state_block1 = (
            state_block2
        ) = (
            state_block3
        ) = state_block4 = state_block5 = state_dense = state_final = None

        for ts in range(self.timesteps):
            encoded = self.encoder(
                x
            ).squeeze()  # Encode a single timestep and remove that dimension

            self.log("input_mean", encoded.mean())
            out_block1, state_block1 = self.block1(encoded, state_block1)  # 1/2
            out_block2, state_block2 = self.block2(out_block1, state_block2)  # 1/4
            out_block3, state_block3 = self.block3(out_block2, state_block3)  # 1/8
            out_block4, state_block4 = self.block4(out_block3, state_block4)  # 1/16
            out_block5, state_block5 = self.block5(out_block4, state_block5)  # 1/32
            out_dense, state_dense = self.dense(out_block5, state_dense)

            self.log("out_block1_mean", out_block1.mean())
            self.log("out_block2_mean", out_block2.mean())
            self.log("out_block3_mean", out_block3.mean())
            self.log("out_block4_mean", out_block4.mean())
            self.log("out_block5_mean", out_block5.mean())
            self.log("out_dense_mean", out_dense.mean())

            ####### WITH FEATURE FUSION
            out_score_block3 = self.score_block3(out_block3)  # 1/8
            out_score_block4 = self.score_block4(out_block4)  # 1/16
            out_score_dense = self.score_dense(out_dense)  # 1/32

            out_upscore_2 = self.upscore_2(out_score_dense)  # 1/16
            out_upscore_block4 = self.upscore_block4(
                out_score_block4 + out_upscore_2
            )  # 1/8
            out_upscore_8 = checkpoint(
                self.upscore_8, out_score_block3 + out_upscore_block4
            )  # 1/1
            #######

            # ####### WITHOUT FEATURE FUSION
            # out_score_dense = self.score_dense(out_dense)  # 1/32

            # out_upscore_2 = self.upscore_2(out_score_dense)  # 1/16
            # out_upscore_block4 = self.upscore_block4(out_upscore_2)  # 1/8
            # out_upscore_8 = self.upscore_8(out_upscore_block4)  # 1/1
            # #######

            # out = out_dense
            # print(out[out != 0].shape)
            # print(out[out.isnan() + out.isinf()].shape)
            # if (out[out.isnan() + out.isinf()] != 0).any():
            #     print(out[out.isnan() + out.isinf()])
            #     print(np.unique(out[out.isnan() + out.isinf()].cpu().detach()))
            # print()

            out_final, state_final = self.final(out_upscore_8, state_final)

        return out_final.squeeze().softmax(1)  # Remove time dimension

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = checkpoint(self.forward, x)
        loss = self.loss_fn(y_pred, y)
        self.log("loss", loss)
        return loss
