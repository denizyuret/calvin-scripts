from importlib import reload
import loaddata as ld
import mlp
import numpy as np
import torch
import torch.nn as nn
import wandb
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
tr = pl.Trainer(accelerator='gpu', devices=1, max_epochs=1, enable_progress_bar=False)
# dtrn = ld.calvindataset("../data/D-training")
# dval = ld.calvindataset("../data/D-validation")
# abcval = ld.calvindataset("../data/ABC-validation")
# abcdval = ld.calvindataset2("../data/ABCD-validation") # same as abcval
