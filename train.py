import sys
import argparse

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import V90kList
from model import MeanScaleHalfHyperprior

from compressai.datasets import ImageFolder


def parse_args(argv: list[str]):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use cuda",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )

    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=0,
        help="Dataloaders threads (default: %(default)s)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: %(default)s)",
    )

    parser.add_argument(
        "--train-data-cnt",
        type=int,
        default=32,
        help="Train data cnt (default: %(default)s)",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/",
        help="Training Data Directory (default: %(default)s)",
    )

    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(args.patch_size),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = V90kList(
        args.data_dir,
        transform=train_transform,
        cnt=args.train_data_cnt,
    )

    test_dataset = ImageFolder(
        args.data_dir,
        transform=test_transform,
        split="test",
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    model = MeanScaleHalfHyperprior(128, 192)

    for _, d in enumerate(train_dataloader):
        x_hat, y_0_likelihoods, z_likelihoods = model(d)
        break

    for _, d in enumerate(test_dataloader):
        x_hat, y_0_likelihoods, z_likelihoods = model(d)
        break


if __name__ == "__main__":
    main(sys.argv[1:])
