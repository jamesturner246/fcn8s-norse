import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from norse.torch.module.sequential import SequentialState
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.lift import Lift
from norse.torch.module.lif import LIFFeedForwardCell
from norse.torch.module.leaky_integrator import LIFeedForwardCell
from norse.torch.module.encode import PoissonEncoder


class FCN8s(nn.Module):

    def __init__(self, dt=1.0, n_class=21, height=512, width=512, method='super', dtype=torch.float):
        super(FCN8s, self).__init__()
        self.dt = dt
        self.n_class = n_class
        self.height = height
        self.width = width
        self.method = method
        self.dtype = dtype

        # block 1
        self.block1 = SequentialState(
            nn.Conv2d(3, 64, 3, padding=1),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=100.0), dt=dt),
            nn.Conv2d(64, 64, 3, padding=1),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=100.0), dt=dt),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/2
        )

        # block 2
        self.block2 = SequentialState(
            nn.Conv2d(64, 128, 3, padding=1),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=100.0), dt=dt),
            nn.Conv2d(128, 128, 3, padding=1),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=100.0), dt=dt),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/4
        )

        # block 3
        self.block3 = SequentialState(
            nn.Conv2d(128, 256, 3, padding=1),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=100.0), dt=dt),
            nn.Conv2d(256, 256, 3, padding=1),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=100.0), dt=dt),
            nn.Conv2d(256, 256, 3, padding=1),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=100.0), dt=dt),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/8
        )

        # block 4
        self.block4 = SequentialState(
            nn.Conv2d(256, 512, 3, padding=1),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=100.0), dt=dt),
            nn.Conv2d(512, 512, 3, padding=1),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=100.0), dt=dt),
            nn.Conv2d(512, 512, 3, padding=1),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=100.0), dt=dt),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/16
        )

        # block 5
        self.block5 = SequentialState(
            nn.Conv2d(512, 512, 3, padding=1),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=100.0), dt=dt),
            nn.Conv2d(512, 512, 3, padding=1),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=100.0), dt=dt),
            nn.Conv2d(512, 512, 3, padding=1),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=100.0), dt=dt),
            nn.AvgPool2d(2, stride=2, ceil_mode=True),  # 1/32
        )

        # dense
        self.dense = SequentialState(
            nn.Conv2d(512, 4096, 7, padding=3),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=100.0), dt=dt),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            LIFFeedForwardCell(p=LIFParameters(method=method, alpha=100.0), dt=dt),
            nn.Dropout2d(),
        )

        self.score_block3 = nn.Conv2d(256, n_class, 1)
        self.score_block4 = nn.Conv2d(512, n_class, 1)
        self.score_dense = nn.Conv2d(4096, n_class, 1)

        self.upscore_2 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, padding=1, bias=False)
        self.upscore_block4 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, padding=1, bias=False)
        self.upscore_8 = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, padding=4, bias=False)

        #self.final = LIFeedForwardCell((n_class, height, width), dt=dt)
        self.final = LIFFeedForwardCell(p=LIFParameters(method=method, alpha=100.0), dt=dt)


    def forward(self, x):

        state_block1 = state_block2 = state_block3 = state_block4 = state_block5 = state_dense = state_final = None

        for ts in range(len(x)):
            out_block1, state_block1 = self.block1(x[ts], state_block1)  # 1/2
            out_block2, state_block2 = self.block2(out_block1, state_block2)  # 1/4
            out_block3, state_block3 = self.block3(out_block2, state_block3)  # 1/8
            out_block4, state_block4 = self.block4(out_block3, state_block4)  # 1/16
            out_block5, state_block5 = self.block5(out_block4, state_block5)  # 1/32
            out_dense, state_dense = self.dense(out_block5, state_dense)

            out_score_block3 = self.score_block3(out_block3)  # 1/8
            out_score_block4 = self.score_block4(out_block4)  # 1/16
            out_score_dense = self.score_dense(out_dense)  # 1/32

            out_upscore_2 = self.upscore_2(out_score_dense)  # 1/16
            out_upscore_block4 = self.upscore_block4(out_score_block4 + out_upscore_2)  # 1/8
            out_upscore_8 = self.upscore_8(out_score_block3 + out_upscore_block4)  # 1/1

            _, state_final = self.final(out_upscore_8, state_final)

        # print('block1 ', out_block1.shape)
        # print('block2 ', out_block2.shape)
        # print('block3 ', out_block3.shape)
        # print('block4 ', out_block4.shape)
        # print('block5 ', out_block5.shape)
        # print('dense ', out_dense.shape)

        #print('score_block3: ', out_score_block3.shape)
        #print('score_block4: ', out_score_block4.shape)
        #print('score_dense: ', out_score_dense.shape)

        #print('upscore_2: ', out_upscore_2.shape)
        #print('upscore_block4: ', out_upscore_block4.shape)
        #print('upscore_8: ', out_upscore_8.shape)

        # print('final ', _.shape)

        return state_final.v


class VOCSegmentationSpikes(torchvision.datasets.VOCSegmentation):

    def __init__(self, *args, **kwargs):
        super(VOCSegmentationSpikes, self).__init__(*args, **kwargs)

    def __len__(self):
        length = super(VOCSegmentationSpikes, self).__len__()
        return length

    def __getitem__(self, i):
        data, labels = super(VOCSegmentationSpikes, self).__getitem__(i)
        labels = (labels * 256).long()
        return data, labels


def imshow(img):
    import matplotlib.pyplot as plt
    npimg = img.numpy()
    plt.imshow(img)
    plt.show()


def compare(y_pred, y):
    y_pred_np = y_pred.detach().numpy()
    y_np = y.detach().numpy()
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(y_pred_np)
    ax[1].imshow(y_np)
    plt.show()


def main():
    sim_steps = 10
    epochs = 10
    dt = 1.0
    #height = 512
    #width = 512
    height = 256
    width = 256
    n_class = 21
    learn_rate = 1e-4
    #learn_rate = 1e-3
    #batch_size = 32
    #batch_size = 16
    #batch_size = 8
    batch_size = 2
    dev = 'cuda'

    transform = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])
    voc11seg_data = VOCSegmentationSpikes('./voc11seg', year='2011', image_set='train', download=True,
                                          transform=transform, target_transform=transform)
    loader = torch.utils.data.DataLoader(voc11seg_data, batch_size=batch_size, shuffle=False,
                                         pin_memory=True, num_workers=0)
    encoder = PoissonEncoder(sim_steps, f_max=1.0, dt=1.0)
    model = FCN8s(dt=dt, n_class=n_class, height=height, width=width).to(dev)
    optimiser = torch.optim.Adam(model.parameters(), lr=learn_rate)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=256)

    for epoch in range(epochs):

        for i, (x, y) in enumerate(loader):

            x = encoder.forward(x).to(dev)
            y = y[:, 0, :, :].to(dev)
            y_pred = model(x)

            # print('y_pred ', y_pred.shape)
            # print('y      ', y.shape)

            loss = loss_fn(y_pred, y)
            if i % 10 == 0:
                print(i, loss.item())

            if i % 250 == 0:
                compare(y_pred[0].cpu().argmax(dim=0), y[0].cpu())

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()


if __name__ == '__main__':
    main()
