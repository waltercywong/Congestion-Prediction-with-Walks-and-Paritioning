import os
import shutil
import numpy as np
import json
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

with open("config.json", "r") as f:
    config = json.load(f)

dataset = NetlistDataset(
    data_dir=data_dir,
    load_pe=config["load_pe"],
    load_pd=config["load_pd"],
    num_eigen=config["num_eigen"],
    pl=config["pl"],
    processed=config["processed"],
    density=config["density"],
    method_type=config["method_type"]
)

print(dataset[0])