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
        path: List of visited node IDs in order
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
    return distances


from tqdm import tqdm
import time
import pandas as pd
import torch

design_list = [1, 2, 3, 5, 6, 7, 9, 11, 14, 16]
valid_pairs_all = []

# Outer progress bar for designs
for design in tqdm(design_list, desc="Processing designs", position=0):
    print(f"\nProcessing design {design}", flush=True)
    file_path = f"de_hnn/data/superblue/superblue_{design}/pyg_data.pkl"
    with open(file_path, 'rb') as file:
        data = torch.load(file)
    source_to_net = data['edge_index_source_to_net']
    sink_to_net = data['edge_index_sink_to_net']
    node_features = data['node_features']  # Get node features
    
    pbar = tqdm(total=1000,
                desc=f"Random walks for design {design}", 
                position=1,
                leave=True,
                ncols=80,
                mininterval=0.001,
                unit='walks')
    
    for i in range(1000):
        start_cell = source_to_net[0][torch.randint(0, len(source_to_net[0]), (1,))].item()
        start_net = source_to_net[1][source_to_net[0] == start_cell]
        path = random_walk_no_revisit(start_cell, source_to_net, sink_to_net)
        for j in range(5, len(path)):
            if path[j][1] < 0.01:
                # Get source and destination features
                source_features = node_features[path[0][0]].tolist()
                dest_features = node_features[path[j][0]].tolist()
                
                # Create dictionary with all features
                pair_dict = {
                    'source': path[0][0],
                    'destination': path[j][0],
                    'design': design
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
    print(flush=True)

# Convert to DataFrame and save to CSV
df = pd.DataFrame(valid_pairs_all)
df.to_csv('valid_pairs.csv', index=False)
print("\nResults saved to valid_pairs.csv")