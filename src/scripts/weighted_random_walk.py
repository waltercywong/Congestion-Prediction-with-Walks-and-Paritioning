import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def calculate_edge_weight(node1_features, node2_features):
    """
    Calculate edge weight based on feature differences between nodes
    
    Args:
        node1_features: Features of first node
        node2_features: Features of second node
    Returns:
        float: Edge weight between 0 and 1
    """
    # Primary features (37-39) weight
    primary_weight = 1.0 / (1.0 + np.mean(np.abs(
        node1_features[37:40] - node2_features[37:40])))
    
    # Secondary features (22-24) weight
    secondary_weight = 1.0 / (1.0 + np.mean(np.abs(
        node1_features[22:25] - node2_features[22:25])))
    
    # Tertiary features (33-35) weight
    tertiary_weight = 1.0 / (1.0 + np.mean(np.abs(
        node1_features[33:36] - node2_features[33:36])))
    
    # Base features (5-7) weight
    base_weight = 1.0 / (1.0 + np.mean(np.abs(
        node1_features[5:8] - node2_features[5:8])))
    
    # Combine weights
    weight = (0.4 * primary_weight + 
             0.3 * secondary_weight + 
             0.2 * tertiary_weight + 
             0.1 * base_weight)
    
    return weight

def build_weighted_edges(data):
    """
    Build weighted edge representation from source_to_net and sink_to_net
    
    Args:
        data: PyG data object containing edge indices and node features
    Returns:
        dict: Dictionary mapping (source_node, dest_node) to weight
    """
    print("Building weighted edge representation...")
    edge_weights = {}
    
    # Get unique nets
    nets = torch.unique(data['edge_index_source_to_net'][1])
    
    # For each net
    for net in tqdm(nets):
        # Get source nodes connected to this net
        source_nodes = data['edge_index_source_to_net'][0][
            data['edge_index_source_to_net'][1] == net]
        
        # Get sink nodes connected to this net  
        sink_nodes = data['edge_index_sink_to_net'][0][
            data['edge_index_sink_to_net'][1] == net]
        
        # Calculate weights between all source-sink pairs
        for source in source_nodes:
            source_features = data['node_features'][source]
            
            for sink in sink_nodes:
                sink_features = data['node_features'][sink]
                
                # Calculate weight
                weight = calculate_edge_weight(
                    source_features.numpy(), 
                    sink_features.numpy()
                )
                
                # Store in dictionary
                edge_weights[(source.item(), sink.item())] = weight
                
    return edge_weights

def weighted_random_walk_no_revisit(start_node, edge_weights, data, max_steps=100):
    """
    Performs weighted random walk starting from given node, without revisiting nodes
    
    Args:
        start_node: Starting node ID
        edge_weights: Dictionary mapping (source, dest) pairs to weights
        data: PyG data object containing edge indices
        max_steps: Maximum number of steps to take
        
    Returns:
        tuple: (distances, node_path) where distances is list of (node, distance) tuples
               and node_path is list of node IDs in traversal order
    """
    visited = set([start_node])
    path = [start_node]
    current_node = start_node
    
    for _ in range(max_steps):
        # Get nets connected to current node
        connected_nets = data['edge_index_source_to_net'][1][
            data['edge_index_source_to_net'][0] == current_node]
        
        if len(connected_nets) == 0:
            break
            
        # Get all possible next nodes and their weights
        possible_next_nodes = []
        weights = []
        
        for net in connected_nets:
            # Get sink nodes connected to this net
            sink_nodes = data['edge_index_sink_to_net'][0][
                data['edge_index_sink_to_net'][1] == net]
            
            # Filter unvisited nodes and get their weights
            for node in sink_nodes:
                node = node.item()
                if node not in visited:
                    possible_next_nodes.append(node)
                    weight = edge_weights.get((current_node, node), 0.0)
                    weights.append(weight)
        
        if not possible_next_nodes:
            break
            
        # Convert to numpy arrays for weighted choice
        possible_next_nodes = np.array(possible_next_nodes)
        weights = np.array(weights)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Weighted random choice
        next_node = np.random.choice(possible_next_nodes, p=weights)
        
        visited.add(next_node)
        path.append(next_node)
        current_node = next_node
        
    # Calculate distances from start position
    start_pos = data['pos_lst'][start_node]
    distances = []
    for node in path:
        pos = data['pos_lst'][node]
        dist = torch.sqrt(((pos[0] - start_pos[0])**2 + 
                          (pos[1] - start_pos[1])**2).float())
        distances.append((node, round(dist.item(), 7)))
        
    return distances, path

def main():
    """Main function to run weighted random walks and generate valid pairs"""
    # design_list = [1, 2, 3, 5, 6, 7, 9, 11, 14, 16]
    design_list = [1]
    valid_pairs_all = []
    num_sources = 1000  # Number of source nodes per design
    
    # Outer progress bar for designs
    for design in tqdm(design_list, desc="Processing designs", position=0):
        print(f"\nProcessing design {design}", flush=True)
        
        # Load data
        file_path = f"de_hnn/data/superblue/superblue_{design}/pyg_data.pkl"
        with open(file_path, 'rb') as file:
            data = torch.load(file)
            
        # Build weighted edge representation
        edge_weights = build_weighted_edges(data)
        
        # Progress bar for random walks
        pbar = tqdm(total=num_sources,
                   desc=f"Random walks for design {design}",
                   position=1,
                   leave=True,
                   ncols=80,
                   mininterval=0.0001,
                   unit='walks')
        
        # Randomly select source nodes
        start_cell_list = data['edge_index_source_to_net'][0][
            torch.randperm(len(data['edge_index_source_to_net'][0]))[:num_sources]]
        
        # Perform random walks
        for i in range(num_sources):
            start_cell = start_cell_list[i].item()
            path, node_path = weighted_random_walk_no_revisit(
                start_cell, edge_weights, data)
            
            # Process valid pairs (distance < 0.01)
            for j in range(5, len(path)):
                if path[j][1] < 0.01:
                    # Get source and destination features
                    source_features = data['node_features'][path[0][0]].tolist()
                    dest_features = data['node_features'][path[j][0]].tolist()
                    
                    # Create dictionary with all features
                    pair_dict = {
                        'source': path[0][0],
                        'destination': path[j][0],
                        'design': design,
                        'path': node_path[:j+1],  # Include path up to destination
                        'path_weight': sum(edge_weights.get((node_path[k], node_path[k+1]), 0.0) 
                                        for k in range(len(node_path[:j+1])-1))
                    }
                    
                    # Add source features
                    for idx, feat in enumerate(source_features):
                        pair_dict[f'source_feature_{idx+1}'] = feat
                        
                    # Add destination features
                    for idx, feat in enumerate(dest_features):
                        pair_dict[f'destination_feature_{idx+1}'] = feat
                    
                    valid_pairs_all.append(pair_dict)
            
            pbar.update(1)
            pbar.refresh()
        
        pbar.close()
        print(f"{len(valid_pairs_all)} valid pairs found.")
        print(flush=True)
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(valid_pairs_all)
    df.to_csv('weighted_valid_pairs.csv', index=False)
    print("\nResults saved to weighted_valid_pairs.csv")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main() 