import torch
import torch.nn as nn
import pytorch_lightning as pl
from loaddata import calvindataset1
from loaddata import calvindataset2
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import accuracy
from torch.nn.functional import cross_entropy


class LitMLP(pl.LightningModule):
    def __init__(self, sizes, dropout=0.5, lr=0.001, weight_decay=0.01):
        super().__init__()
        self.sizes = sizes
        self.mlp = nn.Sequential()
        for i in range(1, len(sizes)-1):
            self.mlp.append(nn.Linear(sizes[i-1], sizes[i]))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(p=dropout))
        self.mlp.append(nn.Linear(sizes[-2], sizes[-1]))
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def step(self, batch, batch_idx):
        x, y = batch
        yhat = self.mlp(x)
        loss = cross_entropy(yhat, y)
        acc = accuracy(yhat, y, task='multiclass', num_classes=self.sizes[-1])
        wnorm = torch.linalg.vector_norm(self.mlp[3].weight)
        return {'loss': loss, 'acc': acc, 'wnorm': wnorm}

    def epoch_end(self, outputs, log_prefix):
        loss = torch.stack([output["loss"] for output in outputs]).mean()
        acc = torch.stack([output["acc"] for output in outputs]).mean()
        wnorm = torch.stack([output["wnorm"] for output in outputs]).mean()
        self.log(log_prefix + "_loss",  loss,  on_step=False, on_epoch=True)
        self.log(log_prefix + "_acc",   acc,   on_step=False, on_epoch=True)
        self.log(log_prefix + "_wnorm", wnorm, on_step=False, on_epoch=True)
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, 'trn')

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, 'val')
        


def train(pre = '../data/D', dataset=calvindataset2, batch_size=128, hidden=[1024,1024], features=range(1,74), window=32, name='001', max_epochs=10, dropout=0, weight_decay=0, lr=0.001):
    global trndict, valdict, mlp
    torch.set_float32_matmul_precision('medium')
    if pre not in trndict:
        trn1 = dataset(prefix = pre + '-training',   features=features, window=window)
        val1 = dataset(prefix = pre + '-validation', features=features, window=window)
        trndict[pre] = DataLoader(trn1, batch_size=batch_size, shuffle=True, num_workers=6)
        valdict[pre] = DataLoader(val1, batch_size=batch_size, shuffle=False, num_workers=6)
    trn = trndict[pre]
    val = valdict[pre]
    xsize = trn.dataset.tensors[0].shape[1]
    ysize = 1 + max(max(trn.dataset.tensors[1]).item(), max(val.dataset.tensors[1]).item())
    mlp = LitMLP((xsize, *hidden, ysize), dropout=dropout, lr=lr, weight_decay=weight_decay)
    wandb_logger = WandbLogger(project='mlp_project', name=name)
    wandb_logger.experiment.config["batch_size"] = batch_size
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=max_epochs, logger=wandb_logger)
    trainer.fit(mlp, trn, val)

trndict = {}
valdict = {}
