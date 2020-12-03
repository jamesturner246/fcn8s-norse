import argparse
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

import norse
from norse.torch.module.sequential import SequentialState
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.lif import LIFFeedForwardCell
from norse.torch.module.leaky_integrator import LIFeedForwardCell
from norse.torch.module.leaky_integrator import LICell


class DVSModel(pl.LightningModule):

    def __init__(self, n_class, height, width, iter_per_frame,
                 class_weights=None, method="super", alpha=100, dt=0.001):
        super(DVSModel, self).__init__()
        self.n_class = n_class
        self.height = height
        self.width = width
        self.iter_per_frame = iter_per_frame
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        # block 1
        self.block1 = SequentialState(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            # nn.Conv2d(64, 64, 3, padding=1, bias=False),
            # LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/2
        )

        # block 2
        self.block2 = SequentialState(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            # nn.Conv2d(128, 128, 3, padding=1, bias=False),
            # LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/4
        )

        # block 3
        self.block3 = SequentialState(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            # nn.Conv2d(256, 256, 3, padding=1, bias=False),
            # LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.BatchNorm2d(256),
        )

        # block 4
        self.block4 = SequentialState(
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.BatchNorm2d(512),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/16
        )

        self.block5 = SequentialState(
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
        self.upscore_4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, padding=1, bias=False
        )
        self.upscore_8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, padding=4, bias=False
        )

        self.final = LIFeedForwardCell(dt=dt)

    def forward(self, x):

        state_block1 = (
            state_block2
        ) = (
            state_block3
        ) = state_block4 = state_block5 = state_dense = state_final = None

        # output              batch,      class,        frame,      height,     width
        output = torch.empty((x.shape[0], self.n_class, x.shape[1], x.shape[3], x.shape[4]))

        # for each frame
        for i in range(x.shape[1]):
            frame = x[:, i, :, :, :].float()

            for j in range(self.iter_per_frame):

                self.log("input_mean", frame.mean())

                out_block1, state_block1 = self.block1(frame, state_block1)       # 1/2
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
                out_upscore_block4 = self.upscore_4(out_score_block4 + out_upscore_2)  # 1/8
                out_upscore_8 = self.upscore_8(out_score_block3 + out_upscore_block4)  # 1/1
                #######

                out_final, state_final = self.final(out_upscore_8, state_final)

            output[:, :, i, :, :] = out_final

        return output
        
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch

        # Before:  B, T, H, W, C
        # We need: B, T, C, H, W
        x = x.permute(0, 1, 4, 2, 3)

        # lightning moves all tensors to gpu, but we want to compute loss on cpu
        self.loss_fn.to('cpu')  # move loss class weights back to cpu
        y = y.to('cpu')         # move labels back to cpu

        z = self.forward(x)

        # print(x.device)
        # print(y.device)
        # print(z.device)

        loss = self.loss_fn(z, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-03)


class DVSNRPDataset(torch.utils.data.Dataset):
    
    def __init__(self, file='scenes_60.dat', skip_frames=None):
        super(DVSNRPDataset, self).__init__()
        self.data = torch.load(file)
        self.skip_frames = skip_frames

        # labels = torch.Tensor([d[1] for d in self.data])
        # print('unique labels: ', torch.unique(labels))
        
    def __getitem__(self, index):
        data, labels = self.data[index]
        return data[self.skip_frames:], labels[self.skip_frames:].astype(np.long)
    
    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', default=None, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    batch_size = 1
    n_class = 22
    height = 512
    width = 512
    iter_per_frame = 1

    dataset = DVSNRPDataset(skip_frames=args.skip)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    # inverse frequency class weighting
    y_count = torch.zeros(n_class, dtype=torch.long)
    for _, y in train_loader:
        l, c = y.unique(return_counts=True)
        for label, count in zip(l, c):
            y_count[label] += count
    y_weights = torch.true_divide(y_count.sum(), y_count)

    #print(y_count)
    #print(y_weights)

    loss_fn = torch.nn.CrossEntropyLoss(weight=y_weights)
    model = DVSModel(n_class, height, width, iter_per_frame, class_weights=y_weights)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader)
