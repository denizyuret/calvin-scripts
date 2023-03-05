import loaddata as ld
import mlp
import numpy as np
import torch
import torch.nn as nn
import wandb
import pytorch_lightning as pl
from importlib import reload
trn = ld.calvindataset1("../data/D-training")
val = ld.calvindataset1("../data/D-validation")
