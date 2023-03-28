import math
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics import MeanSquaredError, MinMetric
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from warnings import warn
from positional_encoding import PositionalEncoding


class GPT(pl.LightningModule):
    """
    I/O Dimensions: (B,T,X) => (B,T,Y), i.e. we use batch_first=True.by default and produce continuous output
    """
    def __init__(self, input_size, hidden_size, output_size, 
                 num_layers=1, num_heads=1, dim_feedforward=-1, 
                 batch_first=True, norm_first=False, norm=None, # transformer
                 dropout=0, weight_decay=0, learning_rate=0.0001, # optimizer
                 data="debug", in_features=[], out_features=[], window=0): # these are not used by the model, just here for logging
        super(GPT, self).__init__()
        self.save_hyperparameters()    # need this to load from checkpoints
        self.__dict__.update(locals()) # convert each local variable (incl args) to self.var
        self._init_metrics()
        self._init_model()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def _init_metrics(self):
        self.trn_loss = MeanSquaredError(squared=False)
        self.val_loss = MeanSquaredError(squared=False)
        self.hp_metric = MinMetric() # tensorboard uses this

    def _init_model(self):
        if self.hidden_size % self.num_heads != 0:
            self.hidden_size = self.num_heads * (self.hidden_size // self.num_heads + 1)
        if self.dim_feedforward <= 0:
            self.dim_feedforward = 4 * self.hidden_size
        # is_causal doesn't work so we need a mask: https://github.com/pytorch/pytorch/issues/96941
        self.src_mask = None
        layer = nn.TransformerEncoderLayer(self.hidden_size, self.num_heads, dim_feedforward=self.dim_feedforward, 
                                           dropout=self.dropout, batch_first=self.batch_first, norm_first=self.norm_first)
        self.encoder = nn.Linear(self.input_size, self.hidden_size), # [B,T,X] => [B,T,H]; in word_language_model there is also a scaling *math.sqrt(hidden)?
        self.pos_encoder = PositionalEncoding(self.hidden_size, dropout=self.dropout, batch_first=self.batch_first), # [B,T,H] => [B,T,H]
        self.transformer = nn.TransformerEncoder(layer, self.num_layers, norm=self.norm), # [B,T,H] => [B,T,H]
        self.decoder = nn.Linear(self.hidden_size, self.output_size)         # [B,T,H] => [B,T,Y]
        
    def forward(self, x, has_mask=True):       # (B,T,X) => (B,T,Y)
        x = self.encoder(x)
        x = self.pos_encoder(x)
        if has_mask:
            mask_size = x.shape[1] if self.batch_first else x.shape[0]
            if self.src_mask == None or self.src_mask.shape[0] != mask_size:
                self.src_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
            x = self.transformer(x, self.src_mask)
        else:
            x = self.transformer(x)
        return self.decoder(x)

    def predict_step(self, batch, batch_idx): # ((B,T,X), *rest) => ((B,T,Y), *rest)
        x, *rest = batch
        return (self(x), *rest)

    def training_step(self, batch, batch_idx):
        inputs, target, *_ = batch        # inputs=(B,T,X), target=(B,T,Y)
        return self.trn_loss(self(inputs), target)

    def validation_step(self, batch, batch_idx):
        inputs, target, *_ = batch        # inputs=(B,T,X), target=(B,T,Y)
        self.val_loss.update(self(inputs), target)

    def training_epoch_end(self, outputs):
        self.log("wnorm", torch.norm(torch.cat([p.view(-1) for p in self.parameters()])))
        self.log("trn_loss", self.trn_loss.compute())
        self.trn_loss.reset()

    def validation_epoch_end(self, outputs):
        val_loss = self.val_loss.compute()
        self.val_loss.reset()
        self.log("val_loss", val_loss)
        self.hp_metric.update(val_loss)
        self.log("hp_metric", self.hp_metric.compute())


# Given trn and val datasets and optional hyperparameters, train and return final val_loss
def train(trn_set, val_set, batch_size=32, max_epochs=-1, max_steps=-1,
          hidden_size=512, num_layers=1, num_heads=1, dim_feedforward=-1, batch_first=True, norm_first=False, norm=None, # transformer
          dropout=0, weight_decay=0, learning_rate=0.0001, # optimizer
          data="debug", in_features=[], out_features=[], window=0, # these are not used, just here for logging
          **kwargs):
    global trainer, transformer, trn_loader, val_loader #DBG
    if kwargs:
        warn(f"Warning: train: Unrecognized kwargs: {kwargs}")
    if dim_feedforward <= 0:
        dim_feedforward = 4*hidden_size
    input_size = trn_set[0][0].shape[-1]
    output_size = trn_set[0][1].shape[-1]
    transformer = GPT(input_size, hidden_size, output_size, # transformer
                      num_layers=num_layers, num_heads=num_heads, dim_feedforward=dim_feedforward, # transformer
                      batch_first=batch_first, norm_first=norm_first, norm=norm, # transformer
                      dropout=dropout, weight_decay=weight_decay, learning_rate=learning_rate, # optimizer
                      data=data, in_features=in_features, out_features=out_features, window=window) # these are not used by the model, just here for logging
    trn_loader = DataLoader(trn_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)
    checkpoint_callback = ModelCheckpoint(monitor = "val_loss", mode = 'min')
    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=max_epochs, max_steps=max_steps, callbacks=[checkpoint_callback])
    trainer.fit(transformer, trn_loader, val_loader)
    return transformer.hp_metric.compute().item()
