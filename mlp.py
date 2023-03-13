# Usage:
# import mlp
# from calvindataset import CalvinDataset
# trn = CalvinDataset('data/D-training.npz')
# val = CalvinDataset('data/D-validation.npz')
# mlp.train(trn, val)

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.linalg import norm
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import MeanMetric, MaxMetric, Accuracy
from torch.nn.functional import cross_entropy, softmax


class LitMLP(pl.LightningModule):
    def __init__(self, sizes, dropout=0.0, weight_decay=0.0, lr=0.0001):
        super().__init__()
        self.save_hyperparameters() # need this to load from checkpoints
        self.sizes = sizes
        self.num_classes = sizes[-1]
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.mlp = nn.Sequential()
        self.mlp.append(nn.Flatten())
        for i in range(1, len(sizes)-1):
            self.mlp.append(nn.Linear(sizes[i-1], sizes[i]))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(p=dropout))
        self.mlp.append(nn.Linear(sizes[-2], sizes[-1]))

        self.trn_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.trn_acc = Accuracy(task = 'multiclass', num_classes = self.num_classes)
        self.val_acc = Accuracy(task = 'multiclass', num_classes = self.num_classes)
        self.wnorm = MaxMetric()
        self.hp_metric = MaxMetric()


    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def forward(self, x):       # (B,T,X) => (B,T*X) => (B,C)
        return self.mlp(x)

    def predict_step(self, batch, batch_idx): # (B,T,X) => (B,T*X) => (B,C)
        x, *rest = batch
        return (softmax(self(x), dim=1), *rest)

    def training_step(self, batch, batch_idx):
        x, y, *rest = batch
        yhat = self(x)
        loss = cross_entropy(yhat, y)
        self.trn_loss.update(loss, len(y))
        self.trn_acc.update(yhat, y)
        self.wnorm.update(norm(self.mlp[4].weight))
        return loss

    def training_epoch_end(self, outputs):
        self.log("trn_loss", self.trn_loss.compute())
        self.log("trn_acc",  self.trn_acc.compute())
        self.log("wnorm",    self.wnorm.compute())
        self.trn_loss.reset(); self.trn_acc.reset(); self.wnorm.reset()

    def validation_step(self, batch, batch_idx):
        x, y, *rest = batch
        yhat = self(x)
        loss = cross_entropy(yhat, y)
        self.val_loss.update(loss, len(y))
        self.val_acc.update(yhat, y)

    def validation_epoch_end(self, outputs):
        self.log("val_loss", self.val_loss.compute())
        val_acc = self.val_acc.compute()
        self.log("val_acc", val_acc)
        self.hp_metric.update(val_acc)
        self.log("hp_metric", self.hp_metric.compute())
        self.val_loss.reset(); self.val_acc.reset()
    


# Given trn and val datasets and optional hyperparameters, train and return an mlp
def train(trn_set, val_set, num_classes=34, hidden=[512], batch_size=32, max_epochs=-1, max_steps=-1, dropout=0.5, weight_decay=0, lr=0.001, name=None):
    global trainer, mlp, trn_loader, val_loader #DBG
    xsize = trn_set[0][0].numel()
    mlp = LitMLP((xsize, *hidden, num_classes), dropout=dropout, lr=lr, weight_decay=weight_decay)
    trn_loader = DataLoader(trn_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    checkpoint_callback = ModelCheckpoint(monitor = "val_acc", mode = 'max')
    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=max_epochs, max_steps=max_steps, callbacks=[checkpoint_callback]) #, logger=wandb_logger)
    trainer.fit(mlp, trn_loader, val_loader)
