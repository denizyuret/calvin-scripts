# Usage:
# import rnn
# from calvin_dataset import CalvinDataset
# trn = CalvinDataset('data/D-training.npz')
# val = CalvinDataset('data/D-validation.npz')
# rnn.train(trn, val)

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.linalg import norm
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import MeanMetric, MaxMetric, Accuracy
from torch.nn.functional import cross_entropy, softmax


class LitRNN(pl.LightningModule):
    """
    RNN model. Input size (B,T,X). Output size (B,T,C) if output_interval <= 0 (default).
    If output_interval > 0, then only the output for the last output_interval steps (and their losses etc) are computed.
    In particular, if output_interval==1, only the output/loss for the last step are computed.
    It is assumed that all time steps have the same target class.
    """
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, bias=True, batch_first=True, dropout=0.0, weight_decay=0.0, lr=0.0001, output_interval=-1):
        super().__init__()
        self.save_hyperparameters()    # need this to load from checkpoints
        self.__dict__.update(locals()) # convert each local variable (incl args) to self.var

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout)
        self.proj = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(hidden_size, num_classes))

        self.trn_loss = [ MeanMetric() for _ in range(3) ]
        self.val_loss = [ MeanMetric() for _ in range(3) ]
        self.trn_acc = [ Accuracy(task = 'multiclass', num_classes = self.num_classes) for _ in range(3) ]
        self.val_acc = [ Accuracy(task = 'multiclass', num_classes = self.num_classes) for _ in range(3) ]
        self.wnorm = MaxMetric()
        self.hp_metric = MaxMetric()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def forward(self, x):       # (B,T,X) => (B,T,C)
        lstm_out, (h_n, c_n) = self.lstm(x)
        return self.proj(lstm_out)

    def predict_step(self, batch, batch_idx): # (B,T,X) => (B,T,C)
        x, *rest = batch
        return (softmax(self(x), dim=2), *rest)

    def step(self, batch, loss, acc):
        x, y, *rest = batch     # x=(B,T,X), y=(B)
        yhat = self(x)          # yhat=(B,T,C)
        # All frames
        yhat0 = yhat.view(-1, yhat.shape[-1])
        y0 = y.repeat_interleave(yhat.shape[1])
        loss0 = cross_entropy(yhat0, y0)
        loss[0].update(loss0.cpu(), len(y0))
        acc[0].update(yhat0.cpu(), y0.cpu())
        # Last frame
        yhat1 = yhat[:,-1,:]
        y1 = y
        loss1 = cross_entropy(yhat1, y1)
        loss[1].update(loss1.cpu(), len(y1))
        acc[1].update(yhat1.cpu(), y1.cpu())
        # Last output_interval frames
        if self.output_interval <= 0 or self.output_interval >= yhat.shape[1]:
            yhat2, y2, loss2 = yhat0, y0, loss0
        elif self.output_interval == 1:
            yhat2, y2, loss2 = yhat1, y1, loss1
        else:
            yhat2 = yhat[:,-self.output_interval:,:].reshape(-1, yhat.shape[-1])
            y2 = y.repeat_interleave(self.output_interval)
            loss2 = cross_entropy(yhat2, y2)
        loss[2].update(loss2.cpu(), len(y2))
        acc[2].update(yhat2.cpu(), y2.cpu())
        return loss2

    def epoch_end(self, pre, loss, acc):
        self.log(pre+"_loss_all", loss[0].compute())
        self.log(pre+"_acc_all",  acc[0].compute())
        self.log(pre+"_loss_1", loss[1].compute())
        self.log(pre+"_acc_1",  acc[1].compute())
        self.log(pre+"_loss_n", loss[2].compute())
        self.log(pre+"_acc_n",  acc[2].compute())
        for x in (*loss, *acc):
            x.reset()

    def training_step(self, batch, batch_idx):
        self.wnorm.update(norm(self.proj[-1].weight))
        return self.step(batch, self.trn_loss, self.trn_acc)

    def validation_step(self, batch, batch_idx):
        self.step(batch, self.val_loss, self.val_acc)

    def training_epoch_end(self, outputs):
        self.epoch_end("trn", self.trn_loss, self.trn_acc)
        self.log("wnorm", self.wnorm.compute())
        self.wnorm.reset()

    def validation_epoch_end(self, outputs):
        self.hp_metric.update(self.val_acc[1].compute())
        self.log("hp_metric", self.hp_metric.compute())
        self.epoch_end("val", self.val_loss, self.val_acc)
    



# Given trn and val datasets and optional hyperparameters, train and return an rnn
def train(trn_set, val_set, batch_size=32, max_epochs=-1, max_steps=-1, name=None,
          hidden_size=512, num_classes=34, num_layers=2, bias=True, batch_first=True,
          dropout=0.5, weight_decay=0.1, lr=0.0001, output_interval=16):
    global trainer, rnn, trn_loader, val_loader #DBG
    input_size = trn_set[0][0].shape[1]
    rnn = LitRNN(input_size, hidden_size, num_classes, num_layers, bias, batch_first, dropout, weight_decay, lr, output_interval)
    trn_loader = DataLoader(trn_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)
    checkpoint_callback = ModelCheckpoint(monitor = "val_acc_1", mode = 'max')
    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=max_epochs, max_steps=max_steps, callbacks=[checkpoint_callback])
    trainer.fit(rnn, trn_loader, val_loader)
