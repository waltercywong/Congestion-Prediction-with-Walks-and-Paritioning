import os
from sndhdr import whathdr
import numpy as np
import pickle
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.data import Dataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from torch_geometric.utils import scatter

import time
import wandb
from tqdm import tqdm
from collections import Counter

import sys
sys.path.insert(1, 'data/')
from pyg_dataset import NetlistDataset

sys.path.append("models/layers/")
from models.model_att import GNN_node
from sklearn.metrics import accuracy_score, precision_score, recall_score

import matplotlib.pyplot as plt

def tensor_describe(tensor):
    # If tensor has more than one dimension, flatten it
    tensor = tensor.flatten()
    count = tensor.numel()
    mean = torch.mean(tensor).item()
    std = torch.std(tensor).item()
    min_val = torch.min(tensor).item()
    max_val = torch.max(tensor).item()
    q25 = torch.quantile(tensor, 0.25).item()  # requires PyTorch 1.7+
    q50 = torch.quantile(tensor, 0.50).item()
    q75 = torch.quantile(tensor, 0.75).item()

    return {
        'count': count,
        'mean': mean,
        'std': std,
        'min': min_val,
        '25%': q25,
        '50%': q50,
        '75%': q75,
        'max': max_val
    }

# Function to compute accuracy, precision, and recall
def compute_metrics(true_labels, predicted_labels):
    # Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Precision
    precision = precision_score(true_labels, predicted_labels, average='binary')
    
    # Recall
    recall = recall_score(true_labels, predicted_labels, average='binary')
    
    return accuracy, precision, recall

### hyperparameter ###
test = False # if only test but not train
restart = False # if restart training
reload_dataset = False # if reload already processed h_dataset

if test:
    restart = True

model_type = "dehnn" #this can be one of ["dehnn", "dehnn_att", "digcn", "digat"] "dehnn_att" might need large memory usage
num_layer = 3 #large number will cause OOM
num_dim = 64 #large number will cause OOM
vn = True #use virtual node or not
trans = False #use transformer or not
aggr = "add" #use aggregation as one of ["add", "max"]
device = "cpu" #use cuda or cpu
learning_rate = 0.001 # could reduce by factor of 10 TODO

num_epochs = 100

# Add these lists to store losses
train_losses_node = []
train_losses_net = []
val_losses_node = []
val_losses_net = []
test_losses_node = []  # New list for test losses
test_losses_net = []   # New list for test losses

if not reload_dataset:
    dataset = NetlistDataset(data_dir="data/superblue", load_pe = True, pl = True, processed = False, load_indices=None)
    h_dataset = []
    
    # Create list to store VN features
    vn_features_list = []
    
    for data in tqdm(dataset):
        num_instances = data.node_features.shape[0]
        data.num_instances = num_instances
        data.edge_index_sink_to_net[1] = data.edge_index_sink_to_net[1] - num_instances
        data.edge_index_source_to_net[1] = data.edge_index_source_to_net[1] - num_instances
        
        out_degrees = data.net_features[:, 1]
        mask = (out_degrees < 3000)
        mask_edges = mask[data.edge_index_source_to_net[1]] 
        filtered_edge_index_source_to_net = data.edge_index_source_to_net[:, mask_edges]
        data.edge_index_source_to_net = filtered_edge_index_source_to_net

        mask_edges = mask[data.edge_index_sink_to_net[1]] 
        filtered_edge_index_sink_to_net = data.edge_index_sink_to_net[:, mask_edges]
        data.edge_index_sink_to_net = filtered_edge_index_sink_to_net

        h_data = HeteroData()
        h_data['node'].x = data.node_features
        h_data['net'].x = data.net_features
        
        edge_index = torch.concat([data.edge_index_sink_to_net, data.edge_index_source_to_net], dim=1)
        h_data['node', 'to', 'net'].edge_index, h_data['node', 'to', 'net'].edge_weight = gcn_norm(edge_index, add_self_loops=False)
        h_data['node', 'to', 'net'].edge_type = torch.concat([torch.zeros(data.edge_index_sink_to_net.shape[1]), torch.ones(data.edge_index_source_to_net.shape[1])]).bool()
        h_data['net', 'to', 'node'].edge_index, h_data['net', 'to', 'node'].edge_weight = gcn_norm(edge_index.flip(0), add_self_loops=False)
        
        h_data['design_name'] = data['design_name']
        h_data.num_instances = data.node_features.shape[0]
        variant_data_lst = []
        
        node_demand = data.node_demand
        net_demand = data.net_demand
        net_hpwl = data.net_hpwl
        
        batch = data.batch
        num_vn = len(np.unique(batch))
        vn_node = torch.concat([global_mean_pool(h_data['node'].x, batch), 
                global_max_pool(h_data['node'].x, batch)], dim=1)
        
        # Save VN features
        design_num = int(data['design_name'].split('_')[1])  # Extract design number
        for vn_idx in range(num_vn):
            vn_features_list.append({
                'design': design_num,
                'vn_id': vn_idx,
                **{f'mean_feature_{i}': vn_node[vn_idx, i].item() for i in range(h_data['node'].x.shape[1])},
                **{f'max_feature_{i}': vn_node[vn_idx, i + h_data['node'].x.shape[1]].item() for i in range(h_data['node'].x.shape[1])}
            })

        # standardiziing TODO
        # node_demand = (node_demand - torch.mean(node_demand)) / torch.std(node_demand)
        # net_hpwl = (net_hpwl - torch.mean(net_hpwl)) / torch.std(net_hpwl)
        # net_demand = (net_demand - torch.mean(net_demand))/ torch.std(net_demand)

        variant_data_lst.append((node_demand, net_hpwl, net_demand, batch, num_vn, vn_node)) 
        h_data['variant_data_lst'] = variant_data_lst
        h_dataset.append(h_data)
        
    # Save VN features to CSV
    # vn_features_df = pd.DataFrame(vn_features_list)
    # vn_features_df.to_csv('vn_features_init.csv', index=False)
    # print(f"Saved initial VN features for {len(vn_features_df)} virtual nodes to vn_features_init.csv")
    
    torch.save(h_dataset, "h_dataset.pt")
    
else:
    dataset = torch.load("h_dataset.pt")
    h_dataset = []
    for data in dataset:
        h_dataset.append(data)
    
sys.path.append("models/layers/")

h_data = h_dataset[0]
if restart:
    model = torch.load(f"{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_model.pt")
else:
    model = GNN_node(num_layer, num_dim, 1, 1, node_dim = h_data['node'].x.shape[1], net_dim = h_data['net'].x.shape[1], gnn_type=model_type, vn=vn, trans=trans, aggr=aggr, JK="Normal").to(device)

criterion_node = nn.MSELoss()
criterion_net = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate,  weight_decay=0.01)
load_data_indices = [idx for idx in range(len(h_dataset))]
all_train_indices = load_data_indices[:5] + load_data_indices[6:11]
all_valid_indices = load_data_indices[11:]
all_test_indices = load_data_indices[5:6]
best_total_val = None

if not test:
    # Create DataFrame to store losses
    losses_df = pd.DataFrame(columns=['epoch', 'node_train_loss', 'net_train_loss', 
                                    'node_val_loss', 'net_val_loss',
                                    'node_test_loss', 'net_test_loss'])  # Added test losses
    
    for epoch in range(num_epochs):
        np.random.shuffle(all_train_indices)
        loss_node_all = 0
        loss_net_all = 0
        val_loss_node_all = 0
        val_loss_net_all = 0
        test_loss_node_all = 0  # New variable for test loss
        test_loss_net_all = 0   # New variable for test loss
        
        all_train_idx = 0
        for data_idx in tqdm(all_train_indices):
            data = h_dataset[data_idx]
            for inner_data_idx in range(len(data.variant_data_lst)):
                target_node, target_net_hpwl, target_net_demand, batch, num_vn, vn_node = data.variant_data_lst[inner_data_idx]
                optimizer.zero_grad()
                data.batch = batch
                data.num_vn = num_vn
                data.vn = vn_node
                node_representation, net_representation = model(data, device)
                node_representation = torch.squeeze(node_representation)
                net_representation = torch.squeeze(net_representation)
                # print(tensor_describe(node_representation))
                loss_node = criterion_node(node_representation, target_node.to(device))
                loss_net = criterion_net(net_representation, target_net_demand.to(device))
                loss = loss_node + loss_net
                loss.backward()
                optimizer.step()   
    
                loss_node_all += loss_node.item()
                loss_net_all += loss_net.item()
                all_train_idx += 1
            
        # Calculate average losses for this epoch
        avg_train_loss_node = torch.sqrt(torch.tensor(loss_node_all/all_train_idx)).item()
        avg_train_loss_net = torch.sqrt(torch.tensor(loss_net_all/all_train_idx)).item()
        train_losses_node.append(avg_train_loss_node)
        train_losses_net.append(avg_train_loss_net)
        print(f"Train RMSE losses - Node: {avg_train_loss_node:.4f}, Net: {avg_train_loss_net:.4f}")
    
        all_valid_idx = 0
        for data_idx in tqdm(all_valid_indices):
            data = h_dataset[data_idx]
            for inner_data_idx in range(len(data.variant_data_lst)):
                target_node, target_net_hpwl, target_net_demand, batch, num_vn, vn_node = data.variant_data_lst[inner_data_idx]
                data.batch = batch
                data.num_vn = num_vn
                data.vn = vn_node
                node_representation, net_representation = model(data, device)
                node_representation = torch.squeeze(node_representation)
                net_representation = torch.squeeze(net_representation)
                
                val_loss_node = criterion_node(node_representation, target_node.to(device))
                val_loss_net = criterion_net(net_representation, target_net_demand.to(device))
                val_loss_node_all += val_loss_node.item()
                val_loss_net_all += val_loss_net.item()
                all_valid_idx += 1
            
        # Calculate average validation losses
        avg_val_loss_node = torch.sqrt(torch.tensor(val_loss_node_all/all_valid_idx)).item()
        avg_val_loss_net = torch.sqrt(torch.tensor(val_loss_net_all/all_valid_idx)).item()
        val_losses_node.append(avg_val_loss_node)
        val_losses_net.append(avg_val_loss_net)
        print(f"Validation RMSE losses - Node: {avg_val_loss_node:.4f}, Net: {avg_val_loss_net:.4f}")
        
        # Calculate test loss for this epoch
        all_test_idx = 0
        for data_idx in tqdm(all_test_indices, desc="Testing"):
            data = h_dataset[data_idx]
            for inner_data_idx in range(len(data.variant_data_lst)):
                target_node, target_net_hpwl, target_net_demand, batch, num_vn, vn_node = data.variant_data_lst[inner_data_idx]
                data.batch = batch
                data.num_vn = num_vn
                data.vn = vn_node
                node_representation, net_representation = model(data, device)
                node_representation = torch.squeeze(node_representation)
                net_representation = torch.squeeze(net_representation)
                
                test_loss_node = criterion_node(node_representation, target_node.to(device))
                test_loss_net = criterion_net(net_representation, target_net_demand.to(device))
                test_loss_node_all += test_loss_node.item()
                test_loss_net_all += test_loss_net.item()
                all_test_idx += 1
        
        # Calculate average test losses
        avg_test_loss_node = torch.sqrt(torch.tensor(test_loss_node_all/all_test_idx)).item()
        avg_test_loss_net = torch.sqrt(torch.tensor(test_loss_net_all/all_test_idx)).item()
        test_losses_node.append(avg_test_loss_node)
        test_losses_net.append(avg_test_loss_net)
        print(f"Test RMSE losses - Node: {avg_test_loss_node:.4f}, Net: {avg_test_loss_net:.4f}")
        
        # Save losses to DataFrame
        new_row = pd.DataFrame({
            'epoch': [epoch],
            'node_train_loss': [avg_train_loss_node],
            'net_train_loss': [avg_train_loss_net],
            'node_val_loss': [avg_val_loss_node],
            'net_val_loss': [avg_val_loss_net],
            'node_test_loss': [avg_test_loss_node],
            'net_test_loss': [avg_test_loss_net]
        })
        losses_df = pd.concat([losses_df, new_row], ignore_index=True)
        
        # Save DataFrame to CSV after each epoch
        losses_df.to_csv(f'training_losses_{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_{method}.csv', index=False)
        
        # Plot losses (now directly using RMSE values)
        plt.figure(figsize=(12, 5))
        
        # Node loss subplot
        plt.subplot(1, 2, 1)
        plt.plot(train_losses_node, label='Train Node Loss')
        plt.plot(val_losses_node, label='Val Node Loss')
        plt.plot(test_losses_node, label='Test Node Loss')
        plt.title('Node Demand Loss')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE Loss')
        plt.legend()
        plt.grid(True)
        
        # Net loss subplot
        plt.subplot(1, 2, 2)
        plt.plot(train_losses_net, label='Train Net Loss')
        plt.plot(val_losses_net, label='Val Net Loss')
        plt.plot(test_losses_net, label='Test Net Loss')
        plt.title('Net Demand Loss')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'training_losses_{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_{method}.png')
        plt.close()

        if (best_total_val is None) or (avg_train_loss_node < best_total_val):
            best_total_val = avg_train_loss_node
            torch.save(model, f"{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_{method}_model.pt")
            
            # Also save the loss values
            loss_data = {
                'train_losses_node': train_losses_node,
                'train_losses_net': train_losses_net,
                'val_losses_node': val_losses_node,
                'val_losses_net': val_losses_net,
                'test_losses_node': test_losses_node,
                'test_losses_net': test_losses_net,
                'best_epoch': epoch
            }
            torch.save(loss_data, f"{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_{method}_losses.pt")

else:
    all_test_idx = 0
    test_loss_node_all = 0
    test_loss_net_all = 0
    for data_idx in tqdm(all_test_indices):
        data = h_dataset[data_idx]
        for inner_data_idx in range(len(data.variant_data_lst)):
            target_node, target_net_hpwl, target_net_demand, batch, num_vn, vn_node = data.variant_data_lst[inner_data_idx]
            data.batch = batch
            data.num_vn = num_vn
            data.vn = vn_node
            node_representation, net_representation = model(data, device)
            node_representation = torch.squeeze(node_representation)
            net_representation = torch.squeeze(net_representation)
            
            test_loss_node = criterion_node(node_representation, target_node.to(device))
            test_loss_net = criterion_net(net_representation, target_net_demand.to(device))
            test_loss_node_all +=  test_loss_node.item()
            test_loss_net_all += test_loss_net.item()
            all_test_idx += 1
    print("avg test node demand mse: ", test_loss_node_all/all_test_idx)
    print("avg test net demand mse: ", test_loss_net_all/all_test_idx)
