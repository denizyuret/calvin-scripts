import torch
from loaddata import calvindataset1
from torch.utils.data import DataLoader
import torch.nn as nn
import pytorch_lightning as pl

class LitMLP(pl.LightningModule):
    def __init__(self, sizes):
        super().__init__()
        self.mlp = nn.Sequential()
        for i in range(1, len(sizes)-1):
            self.mlp.append(nn.Linear(sizes[i-1], sizes[i]))
            self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(sizes[-2], sizes[-1]))

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.mlp(x)
        loss = nn.functional.cross_entropy(yhat, y)
        self.log("trn_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.mlp(x)
        loss = nn.functional.cross_entropy(yhat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def train(pre = '../data/D'):
    global mlp
    trn = DataLoader(calvindataset1(prefix = pre + '-training'), batch_size=128, shuffle=True, num_workers=12)
    val = DataLoader(calvindataset1(prefix = pre + '-validation'), batch_size=128, shuffle=False, num_workers=12)
    xsize = trn.dataset.tensors[0].shape[1]
    ysize = 1 + max(trn.dataset.tensors[1]).item()
    mlp = LitMLP((xsize, 128, ysize))
    trainer = pl.Trainer(accelerator='gpu', devices=1)
    trainer.fit(model=mlp, train_dataloaders=trn)
