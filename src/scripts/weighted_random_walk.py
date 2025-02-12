import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def calculate_edge_weight(node1_features, node2_features, feature_stats):
    """
    Calculate edge weight using learned feature patterns and entropy-based weights
    
    Args:
        node1_features: Features of first node
        node2_features: Features of second node
        feature_stats: Dictionary of feature statistics and weights
    Returns:
        float: Edge weight between 0 and 1
    """
    weight = 0.0
    total_importance = 0.0
    
    for feat_idx, stats in feature_stats.items():
        # Skip if feature stats not available
        if not stats or 'importance_score' not in stats:
            continue
            
        # Get feature difference
        feat_diff = abs(node1_features[feat_idx] - node2_features[feat_idx])
        
        # Calculate feature-specific weight components
        importance = stats['importance_score']
        gradient_contribution = abs(stats['mean_slope']) if 'mean_slope' in stats else 0.0
        consistency = stats['mean_r2'] if 'mean_r2' in stats else 0.0
        
        # New weighting formula that matches the analysis
        feature_weight = (
            0.4 * importance +  # Overall importance score
            0.3 * gradient_contribution +  # Contribution from gradient strength
            0.3 * consistency  # Contribution from consistency
        )
        
        # Normalize difference using feature statistics
        if 'slope_std' in stats and stats['slope_std'] > 0:
            norm_diff = feat_diff / stats['slope_std']
        else:
            norm_diff = feat_diff
        
        # Calculate similarity score (inverse of normalized difference)
        similarity = np.exp(-norm_diff)
        
        # Weight the similarity by feature importance
        weight += feature_weight * similarity
        total_importance += feature_weight
    
    # Normalize final weight
    if total_importance > 0:
        weight = weight / total_importance
    
    return weight

def weighted_random_walk_no_revisit(start_node, data, feature_stats, max_steps=100):
    """
    Performs weighted random walk using learned feature patterns
    
    Args:
        start_node: Starting node ID
        data: PyG data object containing edge indices and node features
        feature_stats: Dictionary of feature statistics and weights
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
            
        # Get all possible next nodes and calculate weights
        possible_next_nodes = []
        weights = []
        
        current_features = data['node_features'][current_node].numpy()
        
        for net in connected_nets:
            # Get sink nodes connected to this net
            sink_nodes = data['edge_index_sink_to_net'][0][
                data['edge_index_sink_to_net'][1] == net]
            
            # Filter unvisited nodes and calculate weights
            for node in sink_nodes:
                node = node.item()
                if node not in visited:
                    node_features = data['node_features'][node].numpy()
                    
                    # Calculate weight using learned patterns
                    weight = calculate_edge_weight(
                        current_features, 
                        node_features,
                        feature_stats
                    )
                    
                    possible_next_nodes.append(node)
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
    """Main function to run weighted random walks using learned patterns"""
    print("Loading feature weights...")
    with open('feature_weights.pkl', 'rb') as f:
        feature_stats = pickle.load(f)
    
    design_list = [1, 2, 3, 5, 6, 7, 9, 11, 14, 16]  # All designs
    valid_pairs_all = []
    num_sources = 1000
    
    # Outer progress bar for designs
    for design in tqdm(design_list, desc="Processing designs", position=0):
        print(f"\nProcessing design {design}", flush=True)
        
        # Load data
        file_path = f"de_hnn/data/superblue/superblue_{design}/pyg_data.pkl"
        with open(file_path, 'rb') as file:
            data = torch.load(file)
        
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
        
        # Track statistics for this design
        design_valid_pairs = 0
        design_total_pairs = 0
        
        # Perform random walks
        for i in range(num_sources):
            start_cell = start_cell_list[i].item()
            distances, path = weighted_random_walk_no_revisit(
                start_cell, data, feature_stats)
            
            # Count pairs checked in this walk (excluding first 5 nodes)
            pairs_in_walk = max(0, len(path) - 5)
            design_total_pairs += pairs_in_walk
            
            # Process valid pairs (distance < 0.01)
            for j in range(5, len(distances)):
                if distances[j][1] < 0.01:
                    design_valid_pairs += 1
                    # Get source and destination features
                    source_features = data['node_features'][distances[0][0]].tolist()
                    dest_features = data['node_features'][distances[j][0]].tolist()
                    
                    # Create dictionary with all features
                    pair_dict = {
                        'source': distances[0][0],
                        'destination': distances[j][0],
                        'design': design,
                        'path': path[:j+1],  # Include path up to destination
                        'path_length': j + 1,
                        'physical_distance': distances[j][1]
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
        
        # Print statistics for this design
        valid_ratio = (design_valid_pairs / design_total_pairs * 100) if design_total_pairs > 0 else 0
        print(f"\nDesign {design} statistics:")
        print(f"  Total pairs checked: {design_total_pairs}")
        print(f"  Valid pairs found: {design_valid_pairs}")
        print(f"  Valid pair ratio: {valid_ratio:.2f}%")
        print(flush=True)
    
    # Print overall statistics
    total_pairs = sum(1 for pair in valid_pairs_all if pair['path_length'] > 5)
    print("\nOverall statistics:")
    print(f"Total valid pairs found: {len(valid_pairs_all)}")
    print(f"Average path length: {np.mean([pair['path_length'] for pair in valid_pairs_all]):.2f}")
    
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