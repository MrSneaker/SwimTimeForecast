import argparse
from .train import train
from .test import test

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "test"], required=True)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--train_fraction", type=float, default=0.30)
parser.add_argument("--val_fraction", type=float, default=0.30)
args = parser.parse_args()

if args.mode == "train":
    train(args.epochs, args.batch_size, args.train_fraction, args.val_fraction)
elif args.mode == "test":
    test()
