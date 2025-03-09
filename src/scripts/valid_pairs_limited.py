import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='X does not have valid feature names')

from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import torch
import random
import pickle

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Get the current directory (src/scripts)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (src)
parent_dir = os.path.dirname(current_dir)

# Define the data directory (data/superblue)
# Navigate from src/scripts to data/superblue using ".."
data_dir = os.path.join(parent_dir, "..", "data", "superblue")

def extract_features(source_features, dest_features):
    """
    Extract combined features for XGBoost prediction
    Only uses features from index 6 to 15 inclusive
    Simply concatenates source and destination features
    """
    # Convert to numpy arrays if they're tensors
    if isinstance(source_features, torch.Tensor):
        source_features = source_features.numpy()
    if isinstance(dest_features, torch.Tensor):
        dest_features = dest_features.numpy()
        
    # Reshape if needed
    if len(source_features.shape) == 1:
        source_features = source_features.reshape(1, -1)
    if len(dest_features.shape) == 1:
        dest_features = dest_features.reshape(1, -1)
    
    # Select only features from index 6 to 15
    source_features = source_features[:, 6:16]
    dest_features = dest_features[:, 6:16]
    
    # Simply concatenate source and destination features
    return np.concatenate([source_features, dest_features], axis=1)

def random_walk_no_revisit(start_node, source_to_net, sink_to_net, node_features, model, max_steps=100):
    """
    Performs random walk starting from given node, without revisiting nodes
    Uses XGBoost classifier to validate pairs
    Limits to 2 connections per source node
    """
    visited = set([start_node])
    path = [start_node]
    current_node = start_node
    valid_pairs = []
    connections_added = 0  # Counter for connections from this source
    
    # Get source features once since they'll be reused
    source_features = node_features[start_node]
    
    for _ in range(max_steps):
        # Stop if we've already added 2 connections for this source
        if connections_added >= 2:
            break
            
        # Get nets connected to current node
        connected_nets = source_to_net[1][source_to_net[0] == current_node]
        
        if len(connected_nets) == 0:
            break
            
        # Randomly select one of the connected nets
        selected_net = connected_nets[torch.randint(0, len(connected_nets), (1,))]
        
        # Get all sink nodes connected to selected net
        possible_next_nodes = sink_to_net[0][sink_to_net[1] == selected_net]
        
        # Filter out already visited nodes
        unvisited_nodes = [n.item() for n in possible_next_nodes if n.item() not in visited]
        
        if not unvisited_nodes:
            break
            
        # Randomly select next unvisited node
        next_node = unvisited_nodes[torch.randint(0, len(unvisited_nodes), (1,)).item()]
        
        # Check if this pair is valid using XGBoost
        dest_features = node_features[next_node]
        combined_features = extract_features(source_features, dest_features)
        
        # Get prediction directly from the model (it will handle scaling internally)
        prediction = model.predict_proba(combined_features)[0]
        is_valid = prediction[1] > 0.8  # Using 0.8 as confidence threshold
        
        if is_valid:
            valid_pairs.append(next_node)
            connections_added += 1  # Increment counter when a valid connection is found
            
        visited.add(next_node)
        path.append(next_node)
        current_node = next_node
        
    return valid_pairs, path

design_list = [1, 2, 3, 5, 6, 7, 9, 11, 14, 16, 18, 19]
total_pairs_all = 0
valid_pairs_all = []
num_sources = 10000

# Load XGBoost model and scaler
print("Loading XGBoost model...")
with open('trained_xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Outer progress bar for designs
for design in tqdm(design_list, desc="Processing designs", position=0):
    print(f"\nProcessing design {design}", flush=True)
    design_total_pairs = 0
    design_valid_pairs = 0
    
    # Construct file path using data_dir
    file_path = os.path.join(data_dir, f"superblue_{design}", "pyg_data.pkl")
    with open(file_path, 'rb') as file:
        data = torch.load(file)
    
    source_to_net = data['edge_index_source_to_net']
    sink_to_net = data['edge_index_sink_to_net']
    node_features = data['node_features']
    
    pbar = tqdm(total=num_sources,
                desc=f"Random walks for design {design}", 
                position=1,
                leave=True,
                ncols=80,
                mininterval=0.0001,
                unit='walks')
    
    start_cell_list = source_to_net[0][torch.randperm(len(source_to_net[0]))[:num_sources]]
    for i in range(num_sources):
        start_cell = start_cell_list[i].item()
        valid_pairs, node_path = random_walk_no_revisit(start_cell, source_to_net, sink_to_net, node_features, model)
        
        # Count pairs checked in this walk
        pairs_in_walk = len(node_path) - 1
        design_total_pairs += pairs_in_walk
        design_valid_pairs += len(valid_pairs)
        
        # Add valid pairs to the list (maximum of 2 per source)
        for dest_node in valid_pairs:
            # Get source and destination features
            source_features = node_features[start_cell].tolist()
            dest_features = node_features[dest_node].tolist()
            
            # Create dictionary with all features
            pair_dict = {
                'source': start_cell,
                'destination': dest_node,
                'design': design,
                'path': node_path[:node_path.index(dest_node) + 1]  # Include path up to the destination node
            }
            
            # Add only features from index 6 to 15
            for idx in range(6, 16):
                pair_dict[f'source_feature_{idx}'] = source_features[idx]
                pair_dict[f'destination_feature_{idx}'] = dest_features[idx]
            
            valid_pairs_all.append(pair_dict)
        
        pbar.update(1)
        pbar.refresh()
        time.sleep(0.01)
    
    pbar.close()
    total_pairs_all += design_total_pairs
    
    # Print statistics for this design
    valid_ratio = (design_valid_pairs / design_total_pairs * 100) if design_total_pairs > 0 else 0
    print(f"Design {design} statistics:")
    print(f"  Total pairs checked: {design_total_pairs}")
    print(f"  Valid pairs found: {design_valid_pairs}")
    print(f"  Valid pair ratio: {valid_ratio:.2f}%")
    print(flush=True)

# Print overall statistics
overall_ratio = (len(valid_pairs_all) / total_pairs_all * 100) if total_pairs_all > 0 else 0
print("\nOverall statistics:")
print(f"Total pairs checked across all designs: {total_pairs_all}")
print(f"Total valid pairs found: {len(valid_pairs_all)}")
print(f"Overall valid pair ratio: {overall_ratio:.2f}%")

# Convert to DataFrame and save to CSV
df = pd.DataFrame(valid_pairs_all)
df.to_csv('valid_pairs_limited.csv', index=False)
print("\nResults saved to valid_pairs_limited.csv")

# Now add dummy connections for the valid pairs
print("\nAdding dummy connections for valid pairs...")

# Group pairs by design
design_pairs = df.groupby('design')

# Process each design
for design in tqdm(design_list, desc="Processing designs"):
    print(f"\nProcessing design {design}")
    
    # Load original pyg data
    file_path = os.path.join(data_dir, f"superblue_{design}", "pyg_data.pkl")
    output_path = os.path.join(data_dir, f"superblue_{design}", "pyg_data_with_valid_limited.pkl")
    
    with open(file_path, 'rb') as file:
        data = torch.load(file)
    
    # Get pairs for this design
    design_data = design_pairs.get_group(design) if design in design_pairs.groups else pd.DataFrame()
    
    if len(design_data) == 0:
        print(f"No valid pairs found for design {design}, copying original file")
        # Add real_net_mask for consistency even when no dummy nets
        data['real_net_mask'] = torch.ones(len(data['net_features']), dtype=torch.bool)
        torch.save(data, output_path)
        continue
    
    # Get existing edge indices and features
    source_to_net = data['edge_index_source_to_net']
    sink_to_net = data['edge_index_sink_to_net']
    net_features = data['net_features']
    net_demand = data['net_demand']
    net_hpwl = data['net_hpwl']
    
    # Calculate the next available net index
    dummy_net_idx = max(
        torch.max(source_to_net[1]).item(),
        torch.max(sink_to_net[1]).item()
    ) - max(torch.max(source_to_net[0]).item(), torch.max(sink_to_net[0]).item())
    
    # Create new edges for valid pairs
    new_source_edges_0 = []  # Sources
    new_source_edges_1 = []  # Nets
    new_sink_edges_0 = []    # Sinks
    new_sink_edges_1 = []    # Nets
    
    # Add new edges for each valid pair
    for _, row in design_data.iterrows():
        source = row['source']
        destination = row['destination']
        
        # Connect both source and destination to the dummy net
        new_source_edges_0.append(source)
        new_source_edges_1.append(dummy_net_idx)
        
        new_sink_edges_0.append(destination)
        new_sink_edges_1.append(dummy_net_idx)
    
    # Convert to tensors
    new_source_edges = torch.tensor([new_source_edges_0, new_source_edges_1], dtype=source_to_net.dtype)
    new_sink_edges = torch.tensor([new_sink_edges_0, new_sink_edges_1], dtype=sink_to_net.dtype)
    
    # Create dummy features and demands (all zeros)
    dummy_net_features = torch.zeros((1, net_features.size(1)), dtype=net_features.dtype)
    dummy_net_demand = torch.zeros(1, dtype=net_demand.dtype)
    dummy_net_hpwl = torch.zeros(1, dtype=net_hpwl.dtype)
    
    # Create mask for real (non-dummy) nets
    real_net_mask = torch.ones(len(net_features) + 1, dtype=torch.bool)  # +1 for dummy net
    real_net_mask[dummy_net_idx] = False  # Mark dummy net as False
    
    # Add dummy features and demands to existing tensors
    data['net_features'] = torch.cat([net_features, dummy_net_features], dim=0)
    data['net_demand'] = torch.cat([net_demand, dummy_net_demand], dim=0)
    data['net_hpwl'] = torch.cat([net_hpwl, dummy_net_hpwl], dim=0)
    data['real_net_mask'] = real_net_mask
    
    # Concatenate new edges with existing edges
    data['edge_index_source_to_net'] = torch.cat([source_to_net, new_source_edges], dim=1)
    data['edge_index_sink_to_net'] = torch.cat([sink_to_net, new_sink_edges], dim=1)
    
    # Save modified data
    print(f"Saving modified data for design {design}")
    print(f"Added {len(design_data)} new connections using dummy net {dummy_net_idx}")
    print(f"Original edges: source_to_net={source_to_net.shape[1]}, sink_to_net={sink_to_net.shape[1]}")
    print(f"New edges: source_to_net={data['edge_index_source_to_net'].shape[1]}, sink_to_net={data['edge_index_sink_to_net'].shape[1]}")
    print(f"Net features shape: {data['net_features'].shape}")
    print(f"Net demand shape: {data['net_demand'].shape}")
    print(f"Net HPWL shape: {data['net_hpwl'].shape}")
    print(f"Real net mask shape: {data['real_net_mask'].shape}")
    
    torch.save(data, output_path)

print("\nProcessing complete! Modified data saved to pyg_data_with_valid_limited.pkl files")