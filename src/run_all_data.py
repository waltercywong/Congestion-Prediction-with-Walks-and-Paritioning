import os
import shutil
import numpy as np
import pickle
import torch
import torch.nn
from torch_geometric.data import Dataset, Data

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from collections import defaultdict
from pyg_dataset import NetlistDataset

current_dir = os.path.dirname(os.path.abspath(__file__))

workspace_dir = os.path.dirname(current_dir)

data_dir = os.path.join(workspace_dir, "data", "superblue")

# Initialize dataset with the dynamically constructed path
dataset = NetlistDataset(
    data_dir=data_dir, 
    load_pe=True, 
    load_pd=True, 
    pl=True, 
    processed=False, 
    load_indices=None, 
    density=True
)

print(dataset[0])
