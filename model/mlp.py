# Usage:
# import mlp
# import loaddata as ld
# trn = ld.calvindataset1('../data/D-training')
# val = ld.calvindataset1('../data/D-validation')
# mlp.train(trn, val)

import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import loaddata as ld
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import accuracy
from torch.nn.functional import cross_entropy


class LitMLP(pl.LightningModule):
    def __init__(self, sizes, lr=0.001, weight_decay=0.0, dropout=0.0):
        super().__init__()
        self.save_hyperparameters() # need this to load from checkpoints
        self.sizes = sizes
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.mlp = nn.Sequential()
        for i in range(1, len(sizes)-1):
            self.mlp.append(nn.Linear(sizes[i-1], sizes[i]))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(p=dropout))
        self.mlp.append(nn.Linear(sizes[-2], sizes[-1]))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def forward(self, x):
        return self.mlp(x)

    def predict_step(self, batch, batch_idx):
        return self(batch[0])

    def training_step(self, batch, batch_idx):
        x, y, *rest = batch
        yhat = self(x)
        return {
            'loss': cross_entropy(yhat, y),
            # 'wnorm': torch.linalg.vector_norm(self.mlp[3].weight),
            # 'yhat': yhat.cpu(),
            # 'y': y.cpu()
        }

    def training_epoch_end(self, outputs):
        # yhats = torch.cat([output['yhat'] for output in outputs])
        # ys = torch.cat([output['y'] for output in outputs])
        # self.log("trn_acc", accuracy(yhats, ys, task='multiclass', num_classes=self.sizes[-1]))
        # self.log("wnorm",torch.stack([output["wnorm"] for output in outputs]).max())
        self.log("trn_loss", torch.stack([output["loss"] for output in outputs]).mean())

    def validation_step(self, batch, batch_idx):
        x, y, *rest = batch
        return {'yhat': self(x).cpu(), 'y': y.cpu()}

    def validation_epoch_end(self, outputs):
        yhats = torch.cat([output['yhat'] for output in outputs])
        ys = torch.cat([output['y'] for output in outputs])
        self.log("val_acc", accuracy(yhats, ys, task='multiclass', num_classes=self.sizes[-1]))
        self.log("val_loss", cross_entropy(yhats, ys))
    

# Given trn and val datasets and optional hyperparameters, train and return an mlp
def train(trn_set, val_set, hidden=[512], batch_size=128, max_epochs=20, dropout=0.5, weight_decay=0, lr=0.001, name=None):
    global trainer, mlp, trn_loader, val_loader #DBG
    xsize = trn_set.tensors[0].shape[1]
    ysize = 1 + max(max(trn_set.tensors[1]).item(), max(val_set.tensors[1]).item())
    mlp = LitMLP((xsize, *hidden, ysize), dropout=dropout, lr=lr, weight_decay=weight_decay)
    trn_loader = DataLoader(trn_set, batch_size=batch_size, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=6)
    wandb_logger = WandbLogger(project='mlp_project', name=name)
    wandb_logger.experiment.config["batch_size"] = batch_size
    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=max_epochs, logger=wandb_logger)
    trainer.fit(mlp, trn_loader, val_loader)
    wandb.finish()


