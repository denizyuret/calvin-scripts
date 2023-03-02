import torch
import torch.nn as nn
import pytorch_lightning as pl
from loaddata import calvindataset1
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import accuracy
from torch.nn.functional import cross_entropy


class LitMLP(pl.LightningModule):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes
        self.mlp = nn.Sequential()
        for i in range(1, len(sizes)-1):
            self.mlp.append(nn.Linear(sizes[i-1], sizes[i]))
            self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(sizes[-2], sizes[-1]))

    def step(self, batch, batch_idx):
        x, y = batch
        yhat = self.mlp(x)
        loss = cross_entropy(yhat, y)
        acc = accuracy(yhat, y, task='multiclass', num_classes=self.sizes[-1])
        return {'loss': loss, 'acc': acc}

    def epoch_end(self, outputs, log_prefix):
        loss = torch.stack([output["loss"] for output in outputs]).mean()
        acc = torch.stack([output["acc"] for output in outputs]).mean()
        self.log(log_prefix + "_loss", loss, on_step=False, on_epoch=True)
        self.log(log_prefix + "_acc",   acc, on_step=False, on_epoch=True)
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, 'trn')

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, 'val')
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def train(pre = '../data/D', batch_size=128, hidden=[128], features=range(1,74), window=32, name='001', max_epochs=32):
    global mlp
    torch.set_float32_matmul_precision('medium')
    trn = calvindataset1(prefix = pre + '-training',   features=features, window=window)
    val = calvindataset1(prefix = pre + '-validation', features=features, window=window)
    trn = DataLoader(trn, batch_size=batch_size, shuffle=True, num_workers=6)
    val = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=6)
    xsize = trn.dataset.tensors[0].shape[1]
    ysize = 1 + max(max(trn.dataset.tensors[1]).item(), max(val.dataset.tensors[1]).item())
    mlp = LitMLP((xsize, *hidden, ysize))
    wandb_logger = WandbLogger(project='mlp_project', name=name)
    wandb_logger.experiment.config["batch_size"] = batch_size
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=max_epochs, logger=wandb_logger)
    trainer.fit(mlp, trn, val)
