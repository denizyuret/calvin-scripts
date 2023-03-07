import sys
import torch
import numpy as np
import mlp
import loaddata as ld
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset
from argparse import ArgumentParser


class CalvinIterableDataset(IterableDataset):
    def __init__(self, prefix="../data/ABCD-training", features=range(1,74), instances_per_episode=32, context_length=32):
        (self.data, self.pos2id, self.id2pos) = ld.loaddata(prefix)
        self.features = features
        self.instances_per_episode=instances_per_episode
        self.context_length=context_length

    def __iter__(self):
        for p in range(0, len(self.pos2id) - self.context_length):
            x = torch.tensor(self.data[np.ix_(range(p, p+self.context_length), self.features)].reshape(-1))
            y = torch.tensor(0)
            idx = torch.tensor(self.pos2id[p+self.context_length-1])
            yield(x, y, idx)


if __name__ == "__main__":
    parser = ArgumentParser(description="Output predictions of a model on a CALVIN dataset")
    parser.add_argument("-m", "--model", type=str, default="./mlp_project/i7t7yfv6/checkpoints/epoch=16-step=390422.ckpt", help="Model checkpoint")
    parser.add_argument("-d", "--data",  type=str, default="../data/debug-validation", help="Prefix of data path")
    args = parser.parse_args()
    tr = pl.Trainer(accelerator='gpu', devices=1, max_epochs=1)
    model = mlp.LitMLP.load_from_checkpoint(args.model)
    data = CalvinIterableDataset(args.data)
    preds = tr.predict(model, DataLoader(data, batch_size=1024))
    print("Writing to predict.out...", file=sys.stderr)
    with open('predict.out', 'w') as f:
        for (yhat, y, idnum) in preds:
            for i in range(len(idnum)):
                print(idnum[i].item(), end='\t', file=f)
                print('\t'.join('{:g}'.format(x) for x in yhat[i,:].tolist()), file=f)
