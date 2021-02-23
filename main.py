import argparse
import pathlib
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import pytorch_lightning as pl
import torchvision
import tqdm

from dvsdata import DVSNRPDataset
from models import ann, snn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--window-length", default=50, type=int)
    parser.add_argument(
        "--model", choices=["ann", "full", "simple", "simple2"], default="full", type=str
    )
    parser.add_argument(
        "--scale", type=float, default=1, help="Scale the data (512x512) by this factor"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--frame-iter",
        type=int,
        default=1,
        help="Number of times each frame is presented (1 = real-time)",
    )
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Constants
    n_class = 9
    n_channels = 3
    height = int(512 * args.scale)
    width = int(512 * args.scale)

    # Setup data sets
    dataset = DVSNRPDataset(
        args.root, window_length=args.window_length, scale_factor=args.scale
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    # inverse frequency class weighting
    weights_filename = "weights.pt"
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

    # Define model
    if args.model == "ann":
        model_cls = ann.DVSModel
    elif args.model == "simple":
        model_cls = snn.DVSModelSimple
    elif args.model == "simple2":
        model_cls = snn.DVSModelSimple2
    else:
        model_cls = snn.DVSModel
    
    # Start training
    model = model_cls(n_class, n_channels, height, width, iter_per_frame=args.frame_iter)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader)
