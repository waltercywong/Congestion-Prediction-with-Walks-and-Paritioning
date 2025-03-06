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

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

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

def calculate_edge_weight(node1_features, node2_features, feature_stats):
    """Calculate edge weight using learned feature patterns"""
    weight = 0.0
    total_importance = 0.0
    
    for feat_idx, stats in feature_stats.items():
        if not stats or 'importance_score' not in stats:
            continue
            
        # Get feature difference
        feat_diff = abs(node1_features[feat_idx] - node2_features[feat_idx])
        
        # Get feature importance and stability
        importance = stats['importance_score']
        stability = stats['stability']
        
        # Weight features that change less during paths more heavily
        feature_weight = importance * stability
        
        # Calculate similarity score (inverse of difference)
        similarity = np.exp(-feat_diff)
        
        # Weight the similarity by feature importance
        weight += feature_weight * similarity
        total_importance += feature_weight
    
    # Normalize final weight
    if total_importance > 0:
        weight = weight / total_importance
    
    return weight

def weighted_random_walk_no_revisit(start_node, source_to_net, sink_to_net, node_features, model, feature_stats, max_steps=100):
    """
    Performs weighted random walk starting from given node, without revisiting nodes
    Uses XGBoost classifier to validate pairs with a limit of 3 valid nodes per walk
    """
    visited = set([start_node])
    path = [start_node]
    current_node = start_node
    valid_pairs = []
    
    # Get source features once since they'll be reused
    source_features = node_features[start_node]
    
    # Counter for valid nodes found
    valid_nodes_found = 0
    
    for _ in range(max_steps):
        if valid_nodes_found >= 3:  # Stop if we've found 3 valid nodes
            break
            
        # Get nets connected to current node
        connected_nets = source_to_net[1][source_to_net[0] == current_node]
        
        if len(connected_nets) == 0:
            break
            
        # Get possible next nodes and calculate weights
        possible_next_nodes = []
        weights = []
        
        current_features = node_features[current_node].numpy()
        
        # For each connected net, get possible sink nodes
        for net in connected_nets:
            sink_nodes = sink_to_net[0][sink_to_net[1] == net]
            
            for node in sink_nodes:
                node = node.item()
                if node not in visited:
                    node_features_np = node_features[node].numpy()
                    weight = calculate_edge_weight(
                        current_features, 
                        node_features_np,
                        feature_stats
                    )
                    possible_next_nodes.append(node)
                    weights.append(weight)
        
        if not possible_next_nodes:
            break
            
        # Normalize weights and make weighted choice
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        next_node = np.random.choice(possible_next_nodes, p=weights)
        
        # Check if this pair is valid using XGBoost
        dest_features = node_features[next_node]
        combined_features = extract_features(source_features, dest_features)
        
        # Get prediction from the model
        prediction = model.predict_proba(combined_features)[0]
        is_valid = prediction[1] > 0.8  # Using 0.8 as confidence threshold
        
        if is_valid:
            valid_pairs.append(next_node)
            valid_nodes_found += 1
            
        visited.add(next_node)
        path.append(next_node)
        current_node = next_node
        
    return valid_pairs, path

design_list = [1, 2, 3, 5, 6, 7, 9, 11, 14, 16, 18, 19]
total_pairs_all = 0
valid_pairs_all = []
num_sources = 100

# Load XGBoost model and feature weights
print("Loading models and weights...")
with open('trained_xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('feature_weights.pkl', 'rb') as f:
    feature_stats = pickle.load(f)

# Outer progress bar for designs
for design in tqdm(design_list, desc="Processing designs", position=0):
    print(f"\nProcessing design {design}", flush=True)
    design_total_pairs = 0
    design_valid_pairs = 0
    
    file_path = f"de_hnn/data/superblue/superblue_{design}/pyg_data.pkl"
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
        valid_pairs, node_path = weighted_random_walk_no_revisit(
            start_cell, source_to_net, sink_to_net, node_features, model, feature_stats)
        
        # Count pairs checked in this walk
        pairs_in_walk = len(node_path) - 1
        design_total_pairs += pairs_in_walk
        design_valid_pairs += len(valid_pairs)
        
        # Add valid pairs to the list
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
df.to_csv('valid_pairs_weighted.csv', index=False)
print("\nResults saved to valid_pairs_weighted.csv")

# Now add dummy connections for the valid pairs
print("\nAdding dummy connections for valid pairs...")

# Group pairs by design
design_pairs = df.groupby('design')

# Process each design
for design in tqdm(design_list, desc="Processing designs"):
    print(f"\nProcessing design {design}")
    
    # Load original pyg data
    file_path = f"de_hnn/data/superblue/superblue_{design}/pyg_data.pkl"
    output_path = f"de_hnn/data/superblue/superblue_{design}/pyg_data_with_valid_weighted.pkl"
    
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

print("\nProcessing complete! Modified data saved to pyg_data_with_valid_weighted.pkl files") 