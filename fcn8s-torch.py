import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class FCN8s(nn.Module):

    def __init__(self, n_class, height, width, dtype=torch.float):
        super(FCN8s, self).__init__()
        self.n_class = n_class
        self.height = height
        self.width = width
        self.dtype = dtype

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


class VOCSegmentationNew(torchvision.datasets.VOCSegmentation):

    def __init__(self, *args, **kwargs):
        super(VOCSegmentationNew, self).__init__(*args, **kwargs)

    def __len__(self):
        length = super(VOCSegmentationNew, self).__len__()
        return length

    def __getitem__(self, i):
        data, labels = super(VOCSegmentationNew, self).__getitem__(i)
        labels = (labels * 256).long()
        return data, labels


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
    n_class = 21
    height = 256
    width = 256
    void_label = 256

    epochs = 500
    #learn_rate = 1e-4
    learn_rate = 3e-5
    #batch_size = 1
    batch_size = 32
    dev = 'cuda'

    transform = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])
    voc11seg_data = VOCSegmentationNew('./voc11seg', year='2011', image_set='train', download=True,
                                       transform=transform, target_transform=transform)
    loader = torch.utils.data.DataLoader(voc11seg_data, batch_size=batch_size, shuffle=True,
                                         pin_memory=True, num_workers=0)

    y_count = torch.zeros(n_class, dtype=torch.long)
    for _, y in loader:
        l, c = y.unique(return_counts=True)
        for label, count in zip(l, c):
            if label != void_label:
                y_count[label] += count

    # inverse frequency class weighting
    y_weights = torch.true_divide(y_count.sum(), y_count).to(dev)

    model = FCN8s(n_class, height, width).to(dev)
    optimiser = torch.optim.Adam(model.parameters(), lr=learn_rate)
    loss_fn = torch.nn.CrossEntropyLoss(weight=y_weights, ignore_index=void_label)

    for epoch in range(epochs):

        for i, (x, y) in enumerate(loader):

            x = x.to(dev)
            y = y[:, 0, :, :].to(dev)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if i % 10 == 0:
                print('iteration: ', i, ', loss: ', loss.item())
            #     compare(y_pred[0].cpu().argmax(dim=0), y[0].cpu(), void_label)

            if epoch % 10 == 0 and i == 0:
                print('epoch: ', epoch, ', loss: ', loss.item())
                compare(y_pred[0].cpu().argmax(dim=0), y[0].cpu(), void_label)


if __name__ == '__main__':
    main()
