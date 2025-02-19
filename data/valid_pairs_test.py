import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(42)
#     torch.cuda.manual_seed_all(42)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

def random_walk_no_revisit(start_node, source_to_net, sink_to_net, max_steps=100):
    """
    Performs random walk starting from given node, without revisiting nodes
    
    Args:
        start_node: Starting node ID
        source_to_net: Edge index tensor connecting source nodes to nets
        sink_to_net: Edge index tensor connecting sink nodes to nets 
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
        
        visited.add(next_node)
        path.append(next_node)
        current_node = next_node
        
    # Calculate distances from start position
    start_pos = data['pos_lst'][start_node]
    distances = []
    for node in path:
        pos = data['pos_lst'][node]
        dist = torch.sqrt(((pos[0] - start_pos[0])**2 + (pos[1] - start_pos[1])**2).float())
        distances.append((node, round(dist.item(), 7)))
    return distances, path

design_list = [18, 19]
total_pairs_all = 0
valid_pairs_all = []
num_sources = 10000

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
    node_features = data['node_features']  # Get node features
    
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
        path, node_path = random_walk_no_revisit(start_cell, source_to_net, sink_to_net)
        
        # Count pairs checked in this walk (excluding first 5 nodes)
        pairs_in_walk = max(0, len(path) - 5)
        design_total_pairs += pairs_in_walk
        
        for j in range(5, len(path)):
            if path[j][1] < 0.01:
                design_valid_pairs += 1
                # Get source and destination features
                source_features = node_features[path[0][0]].tolist()
                dest_features = node_features[path[j][0]].tolist()
                
                # Create dictionary with all features
                pair_dict = {
                    'source': path[0][0],
                    'destination': path[j][0],
                    'design': design,
                    'path': node_path[:j+1]  # Include path up to the destination node
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
df.to_csv('valid_pairs_test.csv', index=False)
print("\nResults saved to valid_pairs_test.csv")