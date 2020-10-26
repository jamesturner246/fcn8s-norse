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
from norse.torch.module.encode import PoissonEncoder


class FCN8s(nn.Module):

    def __init__(self, n_class, height, width, dt=0.001, method='super', alpha=100.0, dtype=torch.float):
        super(FCN8s, self).__init__()
        self.n_class = n_class
        self.height = height
        self.width = width
        self.dt = dt
        self.method = method
        self.alpha = alpha
        self.dtype = dtype

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


    def forward(self, x):

        state_block1 = state_block2 = state_block3 = state_block4 = state_block5 = state_dense = state_final = None

        for ts in range(len(x)):

            #inp = x[ts]
            #print(inp.shape)
            #print(inp[inp > 0].shape)
            #print(inp[0].unique())
            #print(inp[0])
            #print(inp[0][inp[0] > 0])
            #print(inp[0][inp[0].isnan() + inp[0].isinf()])


            out_block1, state_block1 = self.block1(x[ts], state_block1)  # 1/2
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



        # out = out_final
        # print(out[out != 0].shape)
        # print(out[out.isnan() + out.isinf()].shape)
        # if (out[out.isnan() + out.isinf()] != 0).any():
        #     print(out[out.isnan() + out.isinf()])
        #     print(np.unique(out[out.isnan() + out.isinf()].cpu().detach()))
        # print()


        return out_final
        #return state_final.v


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

    sim_steps = 20
    epochs = 500
    #learn_rate = 1e-4
    learn_rate = 3e-5
    batch_size = 1
    #batch_size = 8
    dev = 'cuda'

    #f_max = 100.0
    #dt = 0.001
    #f_max = 1.0
    #dt = 1.0
    f_max = 100.0
    dt = 0.1
    dt = 0.5
    dt = 1.0

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

    encoder = PoissonEncoder(sim_steps, f_max=f_max, dt=dt)
    model = FCN8s(n_class, height, width, dt=dt).to(dev)
    optimiser = torch.optim.Adam(model.parameters(), lr=learn_rate)
    loss_fn = torch.nn.CrossEntropyLoss(weight=y_weights, ignore_index=void_label)

    for epoch in range(epochs):

        for i, (x, y) in enumerate(loader):

            x = encoder.forward(x).to(dev)
            y = y[:, 0, :, :].to(dev)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if i % 10 == 0:
                print('iteration: ', i, ', loss: ', loss.item())
            #     compare(y_pred[0].cpu().argmax(dim=0), y[0].cpu(), void_label)

            if i == 0:
                print('epoch: ', epoch, ', loss: ', loss.item())
                compare(y_pred[0].cpu().argmax(dim=0), y[0].cpu(), void_label)


if __name__ == '__main__':
    main()
