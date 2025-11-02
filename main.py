# Final code for reproducibility paper
# 1 is fraudulent
#%% Settings for runs
seeded_run = False
num_epochs = 200


#%% Setup
#Importing all packages
import platform
import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger



pc = platform.system()
if pc == "Darwin":
    os.chdir("/Users/lambertusvanzyl/Desktop/Reproducibility_paper")
else:
    os.chdir("/Users/Lambertus/Desktop/Reproducibility_paper")
    
if seeded_run:
    torch.manual_seed(42)
    np.random.seed(42)
else:
    seed = np.random.SeedSequence().entropy

# Importing custom libraries
from pre_processing import EllipticDataset, IBMAMLDataset

#%% Reading in data and pre-processing
#Processing elliptic dataset
elliptic_data = EllipticDataset(root='/Users/lambertusvanzyl/Documents/Datasets/Elliptic_dataset')[0]
#Processing IBM AML dataset
IBM_data = IBMAMLDataset(root='/Users/lambertusvanzyl/Documents/Datasets/IBM_AML_dataset')[0]


# %%
