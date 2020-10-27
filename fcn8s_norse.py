import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from norse.torch.module.sequential import SequentialState
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.lif import LIFFeedForwardCell
from norse.torch.module.leaky_integrator import LIFeedForwardCell
from norse.torch.module.leaky_integrator import LICell

import pytorch_lightning as pl

class FCN8s(pl.LightningModule):

    def __init__(self, n_class, height, width, timesteps, encoder, dt=0.001, method='super', alpha=100.0):
        super(FCN8s, self).__init__()
        self.n_class = n_class
        self.height = height
        self.width = width
        self.dt = dt
        self.method = method
        self.alpha = alpha
        self.timesteps = timesteps
        self.encoder = encoder(1, dt=dt) # Use single steps for each image encoding to reduce memory overhead

        # block 1
        self.block1 = SequentialState(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/2
        )

        # block 2
        self.block2 = SequentialState(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/4
        )

        # block 3
        self.block3 = SequentialState(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/8
        )

        # block 4
        self.block4 = SequentialState(
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/16
        )

        # block 5
        self.block5 = SequentialState(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/32
        )

        # dense
        self.dense = SequentialState(
            nn.Conv2d(512, 4096, 7, padding=3, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            #nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            #nn.Dropout2d(),
        )

        self.score_block3 = nn.Conv2d(256, n_class, 1, bias=False)
        self.score_block4 = nn.Conv2d(512, n_class, 1, bias=False)
        self.score_dense = nn.Conv2d(4096, n_class, 1, bias=False)

        self.upscore_2 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, padding=1, bias=False)
        self.upscore_block4 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, padding=1, bias=False)
        self.upscore_8 = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, padding=4, bias=False)

        #self.final = LICell(n_class, n_class, dt=dt)
        self.final = LIFeedForwardCell(dt=dt)
        #self.final = LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-04)

    def forward(self, x):

        state_block1 = state_block2 = state_block3 = state_block4 = state_block5 = state_dense = state_final = None

        outputs = []
        for ts in range(self.timesteps):
            encoded = self.encoder(x).squeeze() # Encode a single timestep and remove that dimension
            #inp = x[ts]
            #print(inp.shape)
            #print(inp[inp > 0].shape)
            #print(inp[0].unique())
            #print(inp[0])
            #print(inp[0][inp[0] > 0])
            #print(inp[0][inp[0].isnan() + inp[0].isinf()])

            out_block1, state_block1 = self.block1(encoded, state_block1)  # 1/2
            out_block2, state_block2 = self.block2(out_block1, state_block2)  # 1/4
            out_block3, state_block3 = self.block3(out_block2, state_block3)  # 1/8
            out_block4, state_block4 = self.block4(out_block3, state_block4)  # 1/16
            out_block5, state_block5 = self.block5(out_block4, state_block5)  # 1/32
            out_dense, state_dense = self.dense(out_block5, state_dense)


            ####### WITH FEATURE FUSION
            out_score_block3 = self.score_block3(out_block3)  # 1/8
            out_score_block4 = self.score_block4(out_block4)  # 1/16
            out_score_dense = self.score_dense(out_dense)  # 1/32

            out_upscore_2 = self.upscore_2(out_score_dense)  # 1/16
            out_upscore_block4 = self.upscore_block4(out_score_block4 + out_upscore_2)  # 1/8
            out_upscore_8 = self.upscore_8(out_score_block3 + out_upscore_block4)  # 1/1
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
            outputs.append(out_final)

        return torch.stack(outputs).sum(0) # Sum the time dimension

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        return torch.nn.functional.cross_entropy(y_pred, y)
