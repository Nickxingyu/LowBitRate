import sys
import argparse

from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import V90kList
from model import MeanScaleHalfHyperprior

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
        default=(7, 7),
        help="Size of the patches to be cropped (default: %(default)s)",
    )

    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )

    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=16, 
        help="Batch size (default: %(default)s)",
    )

    return parser.parse_args(argv)

def main(argv):
    args = parse_args(argv)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    
    train_transform = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = V90kList("../data/train/", transform=train_transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    model = MeanScaleHalfHyperprior(128, 192)
    
    print(train_dataset[0].shape)
    for i, d in enumerate(train_dataloader):
        print(d.shape)
        output = model(d)
        break


if __name__ == "__main__":
    main(sys.argv[1:])
