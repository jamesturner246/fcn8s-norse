import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision

import norse
from norse.torch.module.sequential import SequentialState
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.lif import LIFFeedForwardCell
from norse.torch.module.leaky_integrator import LIFeedForwardCell
from norse.torch.module.leaky_integrator import LICell


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
            nn.Conv2d(256, 512, 7, padding=3, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.Conv2d(512, 512, 1, bias=False),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=alpha), dt=dt),
            nn.BatchNorm2d(512),
        )

        self.score_block2 = nn.Conv2d(128, n_class, 1, bias=False)
        self.deconv_block2 = nn.ConvTranspose2d(n_class, n_class, 8, stride=4, padding=2, bias=False)

        self.score_dense = nn.Conv2d(512, n_class, 1, bias=False)
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

                self.log("input_mean", frame.mean())

                out_block1, state_block1 = self.block1(frame, state_block1)       # 1/2
                out_block2, state_block2 = self.block2(out_block1, state_block2)  # 1/4
                out_block3, state_block3 = self.block3(out_block2, state_block3)  # 1/8
                out_dense, state_dense = self.dense(out_block3, state_dense)

                self.log("out_block1_mean", out_block1.mean())
                self.log("out_block2_mean", out_block2.mean())
                self.log("out_block3_mean", out_block3.mean())
                self.log("out_dense_mean", out_dense.mean())

                ####### WITH FEATURE FUSION
                out_score_block2 = self.score_block2(out_block2)
                out_deconv_block2 = self.deconv_block2(out_score_block2)

                out_score_dense = self.score_dense(out_dense)
                out_deconv_dense = self.deconv_dense(out_score_dense)

                out_deconv = out_deconv_block2 + out_deconv_dense
                #######

                out_final, state_final = self.final(out_deconv, state_final)

            output[:, :, i, :, :] = out_final

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
        return torch.optim.Adam(self.parameters(), lr=1e-03)


class DVSNRPDataset(torch.utils.data.Dataset):
    
    def __init__(self, file='scenes_60.dat', skip_frames=None, scale_factor=1):
        super(DVSNRPDataset, self).__init__()
        self.data = torch.load(file)
        self.skip_frames = skip_frames
        self.scale_factor = scale_factor
        
    def __getitem__(self, index):
        data, labels = self.data[index]
        data = torch.tensor(data[self.skip_frames:])
        labels = torch.tensor(labels[self.skip_frames:])
        return self.scale(data), self.scale(labels).long()

    def scale(self, data):
        is_label = len(data.shape) < 4
        # We need to re-size the data to prepare interpolation
        if is_label: # Labels
            image = data.view(1, *data.shape)
        else:                   # Inputs
            # Before:  H, W, C
            # We need: C, H, W
            image = data.view(*data.shape).permute(0, 3, 1, 2).float()
        scaled = torch.nn.functional.interpolate(image, scale_factor=self.scale_factor)
        return (scaled.view(*scaled.shape[1:]) if is_label else scaled)
    
    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', default=None, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    batch_size = 1
    n_class = 9
    n_channels = 2
    scale_factor = 0.25
    height = int(512 * scale_factor)
    width = int(512 * scale_factor)
    iter_per_frame = 1

    dataset = DVSNRPDataset(skip_frames=args.skip, scale_factor = scale_factor)
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
    model = DVSModel(n_class, n_channels, height, width, iter_per_frame, class_weights=y_weights)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader)
