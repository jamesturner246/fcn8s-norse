
import argparse
import numpy as np
import pathlib
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pytorch_lightning as pl


class FCN8s(pl.LightningModule):

    def __init__(self, n_class, height, width, class_weights):
        super(FCN8s, self).__init__()
        self.n_class = n_class
        self.height = height
        self.width = width
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        # block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=2),  # 1/2
        )

        # block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=2),  # 1/4
        )

        # block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=2),  # 1/8
        )

        # block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=2),  # 1/16
        )

        # block 5
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=2),  # 1/32
        )

        # dense
        self.dense = nn.Sequential(
            nn.Conv2d(512, 4096, 7, padding=3, bias=False),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1, bias=False),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(),
        )

        self.score_block3 = nn.Conv2d(256, n_class, 1, bias=False)
        self.score_block4 = nn.Conv2d(512, n_class, 1, bias=False)
        self.score_dense = nn.Conv2d(4096, n_class, 1, bias=False)

        self.upscore_2 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, padding=1, bias=False)
        self.upscore_block4 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, padding=1, bias=False)
        self.upscore_8 = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, padding=4, bias=False)


    def forward(self, x):

        out_block1 = self.block1(x)  # 1/2
        out_block2 = self.block2(out_block1)  # 1/4
        out_block3 = self.block3(out_block2)  # 1/8
        out_block4 = self.block4(out_block3)  # 1/16
        out_block5 = self.block5(out_block4)  # 1/32
        out_dense = self.dense(out_block5)



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


        out_final = out_upscore_8

        return out_final

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
    
    def __init__(self, root, scale_factor=1):
        super(DVSNRPDataset, self).__init__()
        self.root = pathlib.Path(root)
        assert self.root.is_dir(), "Root folder must be a folder"

        self.files = [path.absolute() for path in self.root.glob('*.dat')]
        self.data = []
        for file in self.files:
            _, labels, pixels = torch.load(file)
            for t in zip(pixels, labels):
                self.data.append(t)

    def __getitem__(self, index):
        data, labels = self.data[index]
        return data.permute(2, 0, 1).float(), labels.long() # self.scale(data), self.scale(labels).long()
    
    def __len__(self):
        return len(self.data)



def compare(y_pred, y, void_label):
    y_pred_np = y_pred.detach().numpy()
    y_pred_np = np.ma.masked_where((y_pred_np == void_label), y_pred_np)
    y_np = y.detach().numpy()
    y_np = np.ma.masked_where((y_np == void_label), y_np)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(y_pred_np, cmap='jet', vmin=0, vmax=20)
    ax[1].imshow(y_np, cmap='jet', vmin=0, vmax=20)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='scenes200/', type=str)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    batch_size = 8
    n_class = 9
    n_channels = 3
    scale_factor = 1
    height = int(512 * scale_factor)
    width = int(512 * scale_factor)

    dataset = DVSNRPDataset(args.root, scale_factor = scale_factor)
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
    print("Training")

    loss_fn = torch.nn.CrossEntropyLoss(weight=y_weights)
    model = FCN8s(n_class, height, width, class_weights=y_weights)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader)



if __name__ == '__main__':
    main()
