import torch
from loaddata import calvindataset1
from torch.utils.data import DataLoader
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torchmetrics


class LitMLP(pl.LightningModule):
    def __init__(self, sizes):
        super().__init__()
        self.mlp = nn.Sequential()
        for i in range(1, len(sizes)-1):
            self.mlp.append(nn.Linear(sizes[i-1], sizes[i]))
            self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(sizes[-2], sizes[-1]))
        self.trn_acc = torchmetrics.Accuracy(task="multiclass", num_classes=sizes[-1])
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=sizes[-1])
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.mlp(x)
        self.trn_acc.update(yhat, y)
        #self.log("trn_acc", self.trn_acc) # , on_step=False, on_epoch=True)
        loss = nn.functional.cross_entropy(yhat, y)
        self.log("trn_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.mlp(x)
        self.val_acc.update(yhat, y)
        #self.log("val_acc", self.val_acc) # , on_step=False, on_epoch=True)
        loss = nn.functional.cross_entropy(yhat, y)
        self.log("val_loss", loss)

    def training_epoch_end(self, outputs):
        self.log("trn_acc", self.trn_acc.compute())
        self.trn_acc.reset()

    def validation_epoch_end(self, outputs):
        self.log("val_acc", self.val_acc.compute())
        self.val_acc.reset()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def train(pre = '../data/D', batch_size=128, hidden=[128]):
    global mlp
    torch.set_float32_matmul_precision('medium')
    trn = DataLoader(calvindataset1(prefix = pre + '-training'), batch_size=batch_size, shuffle=True, num_workers=6)
    val = DataLoader(calvindataset1(prefix = pre + '-validation'), batch_size=batch_size, shuffle=False, num_workers=6)
    xsize = trn.dataset.tensors[0].shape[1]
    ysize = 1 + max(max(trn.dataset.tensors[1]).item(), max(val.dataset.tensors[1]).item())
    mlp = LitMLP((xsize, *hidden, ysize))
    wandb_logger = WandbLogger(project='mlp_project')
    wandb_logger.experiment.config["batch_size"] = batch_size
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=32, logger=wandb_logger)
    trainer.fit(mlp, trn, val)
