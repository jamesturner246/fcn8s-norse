import argparse
import torch
import torchvision
import pytorch_lightning as pl
import torchvision.transforms as transforms

class VOCSegmentationNew(torchvision.datasets.VOCSegmentation):

    def __init__(self, *args, **kwargs):
        super(VOCSegmentationNew, self).__init__(*args, **kwargs)

    def __len__(self):
        length = super(VOCSegmentationNew, self).__len__()
        return length

    def __getitem__(self, i):
        data, labels = super(VOCSegmentationNew, self).__getitem__(i)
        # labels = (labels * 256).long()
        return data, labels.squeeze().long()


def compare(y_pred, y, void_label):
    y_pred_np = y_pred.detach().numpy()
    y_pred_np = np.ma.masked_where((y_pred_np == void_label), y_pred_np)
    y_np = y.detach().numpy()
    y_np = np.ma.masked_where((y_np == void_label), y_np)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(y_pred_np, cmap='jet', vmin=0, vmax=20)
    ax[1].imshow(y_np, cmap='jet', vmin=0, vmax=20)
    plt.show()


def main(args):
    n_class = 21
    height = 128 #256
    width = 128 #256
    void_label = 256

    sim_steps = 20
    epochs = 500
    #learn_rate = 1e-4
    learn_rate = 3e-5
    batch_size = 2
    #batch_size = 8
    dev = 'cuda'

    #f_max = 100.0
    dt = 0.001
    #f_max = 1.0
    #dt = 1.0
    f_max = 100.0

    transform = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])
    voc11seg_data = VOCSegmentationNew('./voc11seg', year='2011', image_set='train', download=True,
                                       transform=transform, target_transform=transform)
    loader = torch.utils.data.DataLoader(voc11seg_data, batch_size=batch_size, shuffle=True,
                                         pin_memory=True, num_workers=0)

    # y_count = torch.zeros(n_class, dtype=torch.long)
    # for _, y in loader:
    #     l, c = y.unique(return_counts=True)
    #     for label, count in zip(l, c):
    #         if label != void_label:
    #             y_count[label] += count

    # # inverse frequency class weighting
    # y_weights = torch.true_divide(y_count.sum(), y_count).to(dev)
    # optimiser = torch.optim.Adam(model.parameters(), lr=learn_rate)
    # loss_fn = torch.nn.CrossEntropyLoss(weight=y_weights, ignore_index=void_label)
    if args.model == 'norse':
        import fcn8s_norse as fcn8s
        from norse.torch.module.encode import PoissonEncoder
        model = fcn8s.FCN8s(n_class, height, width, timesteps=sim_steps, encoder = PoissonEncoder, dt=dt).to(dev)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--model', choices=['norse'], default='norse')
    main(parser.parse_args())
