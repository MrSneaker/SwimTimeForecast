import argparse
from .train import train
from .train_v3 import train as train_v3
from .test import test

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "test"], required=True)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--train_fraction", type=float, default=0.30)
parser.add_argument("--val_fraction", type=float, default=0.30)
parser.add_argument("--use_optimized", action="store_true", default=False, 
                    help="Use hyperparameters loaded from best_params.json")
parser.add_argument("--save_test_plots", action="store_true", default=False,
                    help="Save plots of predictions vs true values during testing")
parser.add_argument("--model_version", choices=["v1", "v2", "v3"], default="v3",
                    help="Choose which model version to train/test (v1: LSTM, v2: FFN, v3: TFT)")
args = parser.parse_args()

if args.mode == "train":
    if args.model_version == "v3":
        train_v3(args.epochs, args.batch_size, args.use_optimized)
    else:
        train(args.epochs, args.batch_size, args.train_fraction, args.val_fraction, args.use_optimized, model_version=args.model_version)
elif args.mode == "test":
    test(save_figures=args.save_test_plots, model_version=args.model_version)
