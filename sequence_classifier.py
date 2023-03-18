import math
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics import MeanMetric, MaxMetric, Accuracy
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.functional import cross_entropy, softmax


class SequenceClassifier(pl.LightningModule):
    """
    I/O Dimensions: (B,T,X) => (B,C), i.e. we use batch_first=True.
    all models use: input_size, hidden_size, num_classes, num_layers
    """
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1,
                 num_heads=1, dim_feedforward=-1, # transformer specific
                 dropout=0.0, weight_decay=0.0, lr=0.0001, model="MLP"):
        super().__init__()
        self.save_hyperparameters()    # need this to load from checkpoints
        self.__dict__.update(locals()) # convert each local variable (incl args) to self.var
        self.classifier = getattr(self, f"_init_{model}")()
        self._init_metrics()

    def _init_metrics(self):
        self.trn_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.trn_acc = Accuracy(task = 'multiclass', num_classes = self.num_classes)
        self.val_acc = Accuracy(task = 'multiclass', num_classes = self.num_classes)
        self.hp_metric = MaxMetric() # tensorboard uses this

    def _init_MLP(self):
        m = nn.Sequential(nn.Flatten())
        m.extend((nn.Linear(self.input_size, self.hidden_size), nn.ReLU(), nn.Dropout(p=self.dropout)))
        for _ in range(self.num_layers - 1):
            m.extend((nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Dropout(p=self.dropout)))
        m.append(nn.Linear(self.hidden_size, self.num_classes))
        return m

    def _init_LSTM(self):
        return nn.Sequential(
            nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout), # bias=True by default
            RNNLastOutput(),    # [B,T,H]=>[B,H]
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.num_classes))
    
    def _init_Transformer(self):
        if self.hidden_size % self.num_heads != 0:
            self.hidden_size = self.num_heads * (self.hidden_size // self.num_heads + 1)
        if self.dim_feedforward <= 0:
            self.dim_feedforward = 4 * self.hidden_size
        encoder_layer = nn.TransformerEncoderLayer(self.hidden_size, self.num_heads, self.dim_feedforward, self.dropout, batch_first=True)
        return nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),          # [B,T,X] => [B,T,H]; in word_language_model there is also a scaling *math.sqrt(hidden)?
            PositionalEncoding(self.hidden_size),                  # [B,T,H] => [B,T,H]; do not apply dropout to input
            nn.TransformerEncoder(encoder_layer, self.num_layers), # [B,T,H] => [B,T,H]
            TransformerAvgPool(),                                  # [B,T,H] => [B,H]
            nn.Linear(self.hidden_size, self.num_classes))
        
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def forward(self, x):       # (B,T,X) => (B,C)
        return self.classifier(x)

    def predict_step(self, batch, batch_idx): # (B,T,X) => (B,C)
        x, *rest = batch
        return (softmax(self(x), dim=1), *rest)

    def training_step(self, batch, batch_idx):
        return self.step(batch, self.trn_loss, self.trn_acc)

    def validation_step(self, batch, batch_idx):
        self.step(batch, self.val_loss, self.val_acc)

    def step(self, batch, loss, acc):
        inputs, target, *_ = batch        # inputs=(B,T,X), target=(B)
        preds = self(inputs)              # preds=(B,C)
        loss1 = cross_entropy(preds, target)
        loss.update(loss1, len(target))
        acc.update(preds, target)
        return loss1

    def training_epoch_end(self, outputs):
        self.log("wnorm", torch.norm(torch.cat([p.view(-1) for p in self.parameters()])))
        self.log("trn_loss", self.trn_loss.compute()); self.trn_loss.reset()
        self.log("trn_acc", self.trn_acc.compute());   self.trn_acc.reset()

    def validation_epoch_end(self, outputs):
        self.log("val_loss", self.val_loss.compute()); self.val_loss.reset()
        val_acc = self.val_acc.compute(); self.val_acc.reset()
        self.log("val_acc", val_acc)
        self.hp_metric.update(val_acc)
        self.log("hp_metric", self.hp_metric.compute())


class TransformerAvgPool(nn.Module):
    def forward(self,x):        # [B,T,H] => [B,H]
        return torch.mean(x, dim=1)


# From: https://stackoverflow.com/questions/44130851/simple-lstm-in-pytorch-with-sequential-module
class RNNLastOutput(nn.Module):
    def forward(self,x):
        tensor, _ = x           # LSTM output x = output[B,T,H], (h_n[L,B,H], c_n[L,B,H]) where B:batch,T:seqlen,H:hidden,L:layers,batch_first=True
        return tensor[:,-1,:]   # Should be [B,H]


# From: pytorch/examples/word_language_model/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=64, batch_first=True):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        pe = torch.zeros(max_len, d_model) # [T,H]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if batch_first:
            pe = pe.unsqueeze(0) # [T,H] => [1,T,H]
        else:
            pe = pe.unsqueeze(0).transpose(0, 1) # [T,H] => [1,T,H] => [T,1,H]
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :] # x[B,T,H] + pe[1,0:T,H]
        else:
            x = x + self.pe[:x.size(0), :, :]  # x[T,B,H] + pe[0:T,1,H]
        return self.dropout(x)


# Given trn and val datasets and optional hyperparameters, train and return a model
def train(trn_set, val_set, batch_size=32, max_epochs=-1, max_steps=-1,
          hidden_size=512, num_classes=34, num_layers=2,
          num_heads=1, dim_feedforward=0,
          dropout=0.5, weight_decay=0.1, lr=0.0001, model="MLP",
          **kwargs):
    global trainer, classifier, trn_loader, val_loader #DBG
    if dim_feedforward == 0:
        dim_feedforward = 4*hidden_size
    if model == "MLP":
        input_size = trn_set[0][0].numel() # [T,H] => T*H
    else: 
        input_size = trn_set[0][0].shape[-1] # [T,H] => H
    classifier = SequenceClassifier(input_size, hidden_size, num_classes, num_layers, num_heads, dim_feedforward, dropout, weight_decay, lr, model)
    trn_loader = DataLoader(trn_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)
    checkpoint_callback = ModelCheckpoint(monitor = "val_acc", mode = 'max')
    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=max_epochs, max_steps=max_steps, callbacks=[checkpoint_callback])
    trainer.fit(classifier, trn_loader, val_loader)
    return classifier.hp_metric.compute().item()
