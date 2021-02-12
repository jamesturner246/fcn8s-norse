import argparse
import pathlib
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import pytorch_lightning as pl
import torchvision
import tqdm

import norse
from norse.torch.module.sequential import SequentialState
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.lif import LIFFeedForwardCell
from norse.torch.module.leaky_integrator import LIFeedForwardCell
from norse.torch.module.leaky_integrator import LICell

from fcn8s_norse.dvsdata import DVSNRPDataset

class DVSModel(pl.LightningModule):

    def __init__(self, n_class, n_channels, height, width, iter_per_frame, class_weights=None, method="super", alpha=100, dt=0.001):
        super(DVSModel, self).__init__()
        self.n_class = n_class
        self.height = height
        self.width = width
        self.iter_per_frame = iter_per_frame
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        # block 1
        self.block1 = SequentialState(
            nn.Conv2d(n_channels, 64, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/2
            nn.BatchNorm2d(64),
        )

        # block 2
        self.block2 = SequentialState(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.BatchNorm2d(128),
        )

        # block 3
        self.block3 = SequentialState(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.BatchNorm2d(256),
        )

        # dense
        self.dense = SequentialState(
            nn.Conv2d(256, 256, 7, padding=3, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.Conv2d(256, 256, 1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.BatchNorm2d(256),
        )

        self.score_block2 = nn.Conv2d(128, n_class, 1, bias=False)
        self.deconv_block2 = nn.ConvTranspose2d(n_class, n_class, 8, stride=4, padding=2, bias=False)

        self.score_dense = nn.Conv2d(256, n_class, 1, bias=False)
        self.deconv_dense = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, padding=4, bias=False)

        self.final = LIFeedForwardCell(dt=dt)

    def forward(self, x):
        state_block1 = state_block2 = state_block3 = state_dense = state_final = None
        x = x.float()

        # output              batch,      class,        frame,      height,     width
        output = torch.empty((x.shape[0], self.n_class, x.shape[1], x.shape[3], x.shape[4]))

        # for each frame
        for i in range(x.shape[1]):
            frame = x[:, i, :, :, :]

            for j in range(self.iter_per_frame):

                out_block1, state_block1 = self.block1(frame, state_block1)       # 1/2
                out_block2, state_block2 = self.block2(out_block1, state_block2)  # 1/4
                out_block3, state_block3 = self.block3(out_block2, state_block3)  # 1/8
                out_dense, state_dense = self.dense(out_block3, state_dense)

                ####### WITH FEATURE FUSION
                out_score_block2 = self.score_block2(out_block2)
                out_deconv_block2 = self.deconv_block2(out_score_block2)

                #out_score_dense = checkpoint(self.score_dense, out_dense)
                out_score_dense = self.score_dense(out_dense)
                out_deconv_dense = self.deconv_dense(out_score_dense)

                out_deconv = out_deconv_block2 + out_deconv_dense
                #######

                out_final, state_final = self.final(out_deconv, state_final)

            output[:, :, i, :, :] = out_final

            self.log("input_mean", frame.mean())
            self.log("out_block1_mean", out_block1.mean())
            self.log("out_block2_mean", out_block2.mean())
            self.log("out_block3_mean", out_block3.mean())
            self.log("out_dense_mean", out_dense.mean())

        return output
        
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        x, y = batch
        z = self.forward(x)
        # or compute loss on gpu
        z = z.to('cuda')

        loss = self.loss_fn(z, y)
        # Log the loss
        self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-03, weight_decay=1e-5)

class DVSModelSimple(pl.LightningModule):

    def __init__(self, n_class, n_channels, height, width, iter_per_frame, class_weights=None, method="super", alpha=100, dt=0.001):
        super(DVSModelSimple, self).__init__()
        self.n_class = n_class
        self.height = height
        self.width = width
        self.iter_per_frame = iter_per_frame
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        # block 1
        self.block1 = SequentialState(
            nn.Conv2d(n_channels, 32, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/2
            nn.BatchNorm2d(32),
        )

        # block 2
        self.block2 = SequentialState(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.BatchNorm2d(64),
        )

        # block 3
        self.block3 = SequentialState(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.BatchNorm2d(128),
        )

        # dense
        self.dense = SequentialState(
            nn.Conv2d(128, 128, 7, padding=3, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.Conv2d(128, 128, 1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.BatchNorm2d(128),
        )

        self.score_block2 = nn.Conv2d(64, n_class, 1, bias=False)
        self.deconv_block2 = nn.ConvTranspose2d(n_class, n_class, 8, stride=4, padding=2, bias=False)

        self.score_dense = nn.Conv2d(128, n_class, 1, bias=False)
        self.deconv_dense = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, padding=4, bias=False)

        self.final = LIFeedForwardCell(dt=dt)

    def forward(self, x):
        state_block1 = state_block2 = state_block3 = state_dense = state_final = None
        x = x.float()

        # output              batch,      class,        frame,      height,     width
        output = torch.empty((x.shape[0], self.n_class, x.shape[1], x.shape[3], x.shape[4]))

        # for each frame
        for i in range(x.shape[1]):
            frame = x[:, i, :, :, :]

            for j in range(self.iter_per_frame):

                out_block1, state_block1 = self.block1(frame, state_block1)       # 1/2
                out_block2, state_block2 = self.block2(out_block1, state_block2)  # 1/4
                out_block3, state_block3 = self.block3(out_block2, state_block3)  # 1/8
                out_dense, state_dense = self.dense(out_block3, state_dense)

                ####### WITH FEATURE FUSION
                out_score_block2 = self.score_block2(out_block2)
                out_deconv_block2 = self.deconv_block2(out_score_block2)

                #out_score_dense = checkpoint(self.score_dense, out_dense)
                out_score_dense = self.score_dense(out_dense)
                out_deconv_dense = self.deconv_dense(out_score_dense)

                out_deconv = out_deconv_block2 + out_deconv_dense
                #######

                out_final, state_final = self.final(out_deconv, state_final)

            output[:, :, i, :, :] = out_final

            self.log("input_mean", frame.mean())
            self.log("out_block1_mean", out_block1.mean())
            self.log("out_block2_mean", out_block2.mean())
            self.log("out_block3_mean", out_block3.mean())
            self.log("out_dense_mean", out_dense.mean())

        return output
        
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        x, y = batch
        z = self.forward(x)
        # or compute loss on gpu
        z = z.to('cuda')

        loss = self.loss_fn(z, y)
        # Log the loss
        self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-03, weight_decay=1e-5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--window-length', default=50, type=int)
    parser.add_argument('--model', choices=['full', 'simple'], default='full', type=str)
    parser.add_argument('--scale', type=float, default=1, help="Scale the data (512x512) by this factor")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size")
    parser.add_argument('--frame-iter', type=int, default=1, help="Number of times each frame is presented (1 = real-time)")
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Constants 
    n_class = 9
    n_channels = 3
    height = int(512 * args.scale)
    width = int(512 * args.scale)

    # Setup data sets
    dataset = DVSNRPDataset(args.root, window_length=args.window_length, scale_factor = args.scale)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    # inverse frequency class weighting
    weights_filename = 'weights.pt'
    if not pathlib.Path(weights_filename).exists():
        y_count = torch.zeros(n_class, dtype=torch.long)
        for _, y in tqdm.tqdm(train_loader):
            l, c = y.unique(return_counts=True)
            for label, count in zip(l, c):
                y_count[label] += count
        y_weights = torch.true_divide(y_count.sum(), y_count)
        torch.save(y_weights, weights_filename)
    else:
        y_weights = torch.load(weights_filename)

    # Define model and start training
    if args.model == 'simple':
        model = DVSModelSimple(n_class, n_channels, height, width, iter_per_frame=args.frame_iter)
    else:
        model = DVSModel(n_class, n_channels, height, width, iter_per_frame=args.frame_iter)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader)
