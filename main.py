# Final code for reproducibility paper
# 1 is fraudulent
#%% Settings for runs
seeded_run = False
prototyping = False
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
from models import GCN, ModelWrapper


#%% Reading in data and pre-processing
if pc == "Darwin":
    #Processing elliptic dataset
    elliptic_data = EllipticDataset(root='/Users/lambertusvanzyl/Documents/Datasets/Elliptic_dataset')[0]
    #Processing IBM AML dataset
    IBM_data = IBMAMLDataset(root='/Users/lambertusvanzyl/Documents/Datasets/IBM_AML_dataset')[0]
else:
    #Processing elliptic dataset
    elliptic_data = EllipticDataset(root='/Users/Lambertus/Desktop/Datasets/Elliptic_dataset')[0]
    #Processing IBM AML dataset
    IBM_data = IBMAMLDataset(root='/Users/Lambertus/Desktop/Datasets/IBM_AML_dataset')[0]


# %%
from torch.optim import Adam
if prototyping:
    data = elliptic_data
    #data = IBM_data
    # testing whether pre processing worked
    hidden_units = 64
    learning_rate=0.05
    loss = nn.CrossEntropyLoss()
    model = GCN(in_channels= data.num_node_features, hidden_channels=hidden_units, out_channels=2, dropout=0.5)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    model_wrapper = ModelWrapper(model, optimizer, loss)
    for i in range(1):
        train_loss = model_wrapper.train_step(data, data.train_perf_eval_mask)
        val_loss, val_metrics = model_wrapper.evaluate(data, data.val_perf_eval_mask)
        print(f"Epoch {i+1:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1 illicit: {val_metrics['f1_illicit']:.4f}")

# %%
#from training_and_testing import train_and_test
#test_metrics, best_f1 = train_and_test(model_wrapper, data, data.test_perf_eval_mask, num_epochs=num_epochs)
# %% Optuna runs
from hyperparameter_tuning import run_optimization
#datasets = ["Elliptic", "IBM_AML_HiSmall", "IBM_AML_LiSmall", "IBM_AML_HiMedium", "IBM_AML_LiMedium"]
datasets = ["Elliptic"]
for x in datasets:
    data_for_optimization = x
    model_parameters, testing_results = run_optimization(
        models=['MLP', 'GCN', 'GAT', 'GIN'],
        data=elliptic_data,
        train_perf_eval=elliptic_data.train_perf_eval_mask,
        val_perf_eval=elliptic_data.val_perf_eval_mask,
        test_perf_eval=elliptic_data.test_perf_eval_mask,
        train_mask=elliptic_data.train_mask,
        val_mask=elliptic_data.val_mask,
        data_for_optimization=data_for_optimization
    )
# %%
def save_testing_results_csv(results, path=f"{data_for_optimization}_testing_results.csv"):
    df = pd.DataFrame(results)
    df.to_csv(f"csv_results/{data_for_optimization}_testing_results.csv", index=False)

save_testing_results_csv(testing_results)
# %%

