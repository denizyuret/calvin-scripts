#!/usr/bin/env python

import sys
from argparse import ArgumentParser
from calvin_dataset import CalvinDataset
from sequence_classifier import train

parser = ArgumentParser(description="Supervised experiments on CALVIN")
parser.add_argument("-d", "--data", default="D", type=str, help="Training data: ABCD, ABC, D")
parser.add_argument("-m", "--model", default="MLP", type=str, help="Model type: MLP, LSTM, Transformer")
parser.add_argument("-b", "--batch_size", default=32, type=int, help="Batch size")
parser.add_argument("-l", "--learing_rate", default=0.0001, type=float, help="Learning rate")
parser.add_argument("-s", "--max_steps", default=100000, type=int, help="Max steps")
parser.add_argument("-H", "--hidden_size", default=512, type=int, help="Hidden layer dimension")
parser.add_argument("-L", "--num_layers", default=2, type=int, help="Number of layers")
parser.add_argument("-N", "--num_heads", default=2, type=int, help="Number of transformer heads")
parser.add_argument("-p", "--dropout", default=0.5, type=float, help="Dropout probability")
parser.add_argument("-w", "--weight_decay", default=0.1, type=float, help="Weight decay")
parser.add_argument("-i", "--instances_per_episode", default=1, type=int, help="Instances per episode")
parser.add_argument("-c", "--context_length", default=64, type=int, help="Context length")
parser.add_argument("-f", "--features", default="range(0,97)", type=str, help="Features")

args, _ = parser.parse_known_args()
print(args, file=sys.stderr)
args.features = eval(args.features)
trn = CalvinDataset(f"data/{args.data}-training.npz", **args.__dict__)
val = CalvinDataset("data/D-validation.npz", **args.__dict__)
val_acc = train(trn, val, **args.__dict__)
print(val_acc)
