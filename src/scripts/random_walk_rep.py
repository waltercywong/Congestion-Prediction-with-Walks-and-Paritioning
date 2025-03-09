import random
import numpy as np
import torch
import time
from tqdm import tqdm
import pickle
from copy import deepcopy
import os

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def random_walk(start_node, source_to_net, sink_to_net, data, max_steps=100):
    """Modified to allow revisiting nodes"""
    path = [start_node]
    current_node = start_node
    
    for _ in range(max_steps):
        connected_nets = source_to_net[1][source_to_net[0] == current_node]
        
        if len(connected_nets) == 0:
            break
            
        # Randomly select one of the connected nets
        selected_net = connected_nets[torch.randint(0, len(connected_nets), (1,))]
        
        # Get all sink nodes connected to selected net
        possible_next_nodes = sink_to_net[0][sink_to_net[1] == selected_net]
        
        if len(possible_next_nodes) == 0:
            break
            
        # Randomly select next node (can be previously visited)
        next_node = possible_next_nodes[torch.randint(0, len(possible_next_nodes), (1,))].item()
        
        path.append(next_node)
        current_node = next_node

    # Calculate distances from start position
    start_pos = data['pos_lst'][start_node]
    distances = []
    for node in path:
        pos = data['pos_lst'][node]
        dist = torch.sqrt(((pos[0] - start_pos[0])**2 + (pos[1] - start_pos[1])**2).float())
        distances.append((node, round(dist.item(), 7)))
    return distances

# Get the current directory (src/scripts)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (src)
parent_dir = os.path.dirname(current_dir)

# Define the data directory (data/superblue)
# Navigate from src/scripts to data/superblue using ".."
data_dir = os.path.join(parent_dir, "..", "data", "superblue")

# Get all available designs
design_list = []
for d in os.listdir(data_dir):
    if d.startswith("superblue"):
        try:
            # Extract design number and check if pyg_data.pkl exists
            design_num = int(d.replace("superblue_", ""))
            if os.path.exists(os.path.join(data_dir, d, "pyg_data.pkl")):
                design_list.append(design_num)
        except ValueError:
            continue

design_list.sort()
print(f"Found {len(design_list)} designs: {design_list}")

# Outer progress bar for designs
for design in tqdm(design_list, desc="Processing designs", position=0):
    print(f"\nProcessing design {design}", flush=True)
    file_path = os.path.join(data_dir, f"superblue_{design}", "pyg_data.pkl")
    
    # Load original data
    with open(file_path, 'rb') as file:
        data = torch.load(file)
    
    # Create deep copy for modification
    modified_data = deepcopy(data)
    
    source_to_net = data['edge_index_source_to_net']
    sink_to_net = data['edge_index_sink_to_net']
    
    # Get the maximum net index from original data
    max_net_idx = max(
        torch.max(source_to_net[1]).item(),
        torch.max(sink_to_net[1]).item()
    )
    next_net_idx = max_net_idx + 1
    
    new_source_edges = []
    new_sink_edges = []
    new_net_features = []
    
    # Inner progress bar for random walks
    pbar = tqdm(total=1000,
                desc=f"Random walks for design {design}", 
                position=1,
                leave=True,
                ncols=80,
                mininterval=0.001,
                unit='walks')
    
    for i in range(1000):
        # Randomly select start node from source nodes
        start_cell = source_to_net[0][torch.randint(0, len(source_to_net[0]), (1,))].item()
        path = random_walk(start_cell, source_to_net, sink_to_net, data)
        
        # Check nodes at distance 5 or more hops
        for j in range(5, len(path)):
            if path[j][1] < 0.01:  # If physical distance is small
                source_node = path[0][0]
                dest_node = path[j][0]
                
                # Create new dummy net
                new_source_edges.extend([[source_node, next_net_idx]])
                new_sink_edges.extend([[dest_node, next_net_idx]])
                
                # Create new net features (copying mean of existing net features)
                new_net_features.append(torch.mean(data['net_features'], dim=0))
                
                next_net_idx += 1
        
        pbar.update(1)
        pbar.refresh()
    
    pbar.close()
    
    if new_source_edges:  # Only modify if we found valid pairs
        # Convert new edges to tensor and combine with original
        new_source_edges = torch.tensor(new_source_edges).t()
        new_sink_edges = torch.tensor(new_sink_edges).t()
        
        modified_data['edge_index_source_to_net'] = torch.cat([
            modified_data['edge_index_source_to_net'],
            new_source_edges
        ], dim=1)
        
        modified_data['edge_index_sink_to_net'] = torch.cat([
            modified_data['edge_index_sink_to_net'],
            new_sink_edges
        ], dim=1)
        
        # Add new net features
        new_net_features = torch.stack(new_net_features)
        modified_data['net_features'] = torch.cat([
            modified_data['net_features'],
            new_net_features
        ], dim=0)
        
        # Update net demands and HPWLs for new nets
        zero_demands = torch.zeros(len(new_net_features))
        modified_data['net_demand'] = torch.cat([
            modified_data['net_demand'],
            zero_demands
        ])
        modified_data['net_hpwl'] = torch.cat([
            modified_data['net_hpwl'],
            zero_demands
        ])
    
    # Save modified data
    save_path = os.path.join(data_dir, f"superblue_{design}", "pyg_data_modified.pkl")
    torch.save(modified_data, save_path)
    print(f"Saved modified data to {save_path}")
    print(f"Added {len(new_source_edges)} new connections")

print("\nCompleted processing all designs")