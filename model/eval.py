import sys
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import loaddata as ld
import mlp
import rnn

parser = ArgumentParser(description="Evaluate a model on the D/validation dataset")
parser.add_argument("model", type=str, help="Model checkpoint")
parser.add_argument("-d", "--data", type=str, default="../data/D-validation")
parser.add_argument("-i", "--instances_per_episode",  type=int, default=1)
parser.add_argument("-c", "--context_length",  type=int, default=64)
parser.add_argument("-f", "--features",  type=str, default="range(1,74)")
args = parser.parse_args()

try:
    model = mlp.LitMLP.load_from_checkpoint(args.model)
except TypeError:
    model = rnn.LitRNN.load_from_checkpoint(args.model)
except:
    error('Cannot load model')

data = ld.CalvinDataset(args.data, eval(args.features), args.instances_per_episode, args.context_length)
tr = pl.Trainer(accelerator='gpu', devices=1, max_epochs=1)
tr.validate(model, DataLoader(data, batch_size=128))
