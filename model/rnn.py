import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import loaddata as ld
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
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
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, bias=True, batch_first=True, dropout=0.0, weight_decay=0.0, lr=0.001, output_interval=-1):
        super().__init__()
        self.save_hyperparameters()    # need this to load from checkpoints
        self.__dict__.update(locals()) # convert each local variable (incl args) to self.var

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout)
        self.proj = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(hidden_size, num_classes))

        self.trn_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.trn_acc = Accuracy(task = 'multiclass', num_classes = self.num_classes)
        self.val_acc = Accuracy(task = 'multiclass', num_classes = self.num_classes)
        self.wnorm = MaxMetric()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        if self.output_interval > 0:
            lstm_out = lstm_out[:,-self.output_interval:,:] # (B,T,C) => (B,O,C)
        return self.proj(lstm_out)

    def step(self, batch, loss, acc, wnorm):
        x, y, *rest = batch     # x=(B,T,X), y=(B)
        yhat = self(x)          # yhat=(B,O,C)
        y = y.repeat_interleave(yhat.shape[1])      # y->(B*O)
        yhat = yhat.view(-1, self.num_classes)      # yhat->(B*O,C)
        xent = cross_entropy(yhat, y)
        loss.update(xent, len(y))
        acc.update(yhat, y)
        wnorm.update(torch.linalg.vector_norm(self.proj[-1].weight))
        return xent

    def training_step(self, batch, batch_idx):
        return self.step(batch, self.trn_loss, self.trn_acc, self.wnorm)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, self.val_loss, self.val_acc, self.wnorm)

    def training_epoch_end(self, outputs):
        self.log("trn_loss", self.trn_loss.compute())
        self.log("trn_acc",  self.trn_acc.compute())
        self.log("wnorm",    self.wnorm.compute())
        self.trn_loss.reset(); self.trn_acc.reset(); self.wnorm.reset()

    def validation_epoch_end(self, outputs):
        self.log("val_loss", self.val_loss.compute())
        self.log("val_acc",  self.val_acc.compute())
        self.val_loss.reset(); self.val_acc.reset(); 
    
    def predict_step(self, batch, batch_idx):
        x, *rest = batch
        return (softmax(self(x), dim=2), *rest)


# Given trn and val datasets and optional hyperparameters, train and return an rnn
def train(trn_set, val_set, batch_size=128, max_epochs=-1, max_steps=-1, name=None,
          hidden_size=512, num_classes=34, num_layers=2, bias=True, batch_first=True,
          dropout=0.5, weight_decay=0.1, lr=0.001, output_interval=32):
    global trainer, rnn, trn_loader, val_loader #DBG
    input_size = trn_set[0][0].shape[1]
    rnn = LitRNN(input_size, hidden_size, num_classes, num_layers, bias, batch_first, dropout, weight_decay, lr, output_interval)
    trn_loader = DataLoader(trn_set, batch_size, shuffle=True,  num_workers=6)
    val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=6)
    wandb_logger = WandbLogger(project='mlp_project', name=name)
    wandb_logger.experiment.config["batch_size"] = batch_size
    torch.set_float32_matmul_precision('medium')
    checkpoint_callback = ModelCheckpoint(monitor = "val_acc", mode = 'max')
    trainer = pl.Trainer(accelerator='gpu', devices=1, callbacks=[checkpoint_callback], max_epochs=max_epochs, max_steps=max_steps, logger=wandb_logger)
    trainer.fit(rnn, trn_loader, val_loader)
    wandb.finish()


