import sys
from argparse import ArgumentParser
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multiclass_confusion_matrix, multiclass_accuracy
import loaddata as ld
import mlp
import rnn

parser = ArgumentParser(description="Evaluate a model on a dataset")
parser.add_argument("model", type=str, help="Model checkpoint")
parser.add_argument("-d", "--data", type=str, default="data/D-validation")
parser.add_argument("-i", "--instances_per_episode",  type=int, default=1)
parser.add_argument("-c", "--context_length",  type=int, default=64)
parser.add_argument("-f", "--features",  type=str, default="range(1,98)")
args = parser.parse_args()

print(f"Loading model from {args.model}...", file=sys.stderr)
try:
    model = mlp.LitMLP.load_from_checkpoint(args.model)
except TypeError:
    model = rnn.LitRNN.load_from_checkpoint(args.model)
except:
    error('Cannot load model')

print(f"Loading data from {args.data}...", file=sys.stderr)
data = ld.CalvinDataset(args.data, eval(args.features), args.instances_per_episode, args.context_length)

print(f"Running validation...", file=sys.stderr)
tr = pl.Trainer(accelerator='gpu', devices=1, max_epochs=1)
v = tr.validate(model, DataLoader(data, batch_size=1024))
print(v)

print(f"Printing confusion matrix...", file=sys.stderr)
p = tr.predict(model, DataLoader(data, batch_size=1024))
preds = torch.cat([x[0] for x in p])
target = torch.cat([x[1] for x in p])
index = torch.cat([x[2] for x in p])
num_classes = 34
if isinstance(model, rnn.LitRNN):
    preds = preds[:,-1,:]
print(multiclass_accuracy(preds, target, num_classes, average='micro').item())
a = multiclass_accuracy(preds, target, num_classes, average='none')
c = multiclass_confusion_matrix(preds, target, num_classes)
for j in range(num_classes):
    print(f"{j:>3}", end="")
print("")
for i in range(num_classes):
    for j in range(num_classes):
        if c[i,j] == 0:
            print("  .", end="")
        else:
            print(f"{c[i,j]:>3}", end="")
    print(f"{i:>3} {ld.int2task[i]} ({a[i]:.4f})")

print(f"Saving data and predictions in eval.out...", file=sys.stderr)
(x, *rest) = next(iter(DataLoader(data, batch_size=len(data))))
x = x[:,-1,:]
cpreds = preds.argmax(axis=1)
dump = np.hstack((index.reshape(-1,1), target.reshape(-1,1), cpreds.reshape(-1,1), x))
fmt = ['%d']*3 + ['%.4f']*(dump.shape[1]-3)
np.savetxt('eval.out', dump, fmt=fmt, delimiter='\t')
