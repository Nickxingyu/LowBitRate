import sys
import argparse

from torchvision import transforms

from dataset import V90kList

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

    return parser.parse_args(argv)

def main(argv):
    args = parse_args(argv)
    train_transform = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = V90kList("../data/train/", transform=train_transform)
    for i in range(10):
        print(train_dataset[i])

if __name__ == "__main__":
    main(sys.argv[1:])