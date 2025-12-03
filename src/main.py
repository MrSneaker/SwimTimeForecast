import argparse
from train import train
from test import test

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "test"], required=True)
args = parser.parse_args()

if args.mode == "train":
    train()

elif args.mode == "test":
    test()
