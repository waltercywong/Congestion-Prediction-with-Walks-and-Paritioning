import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import torch
from tqdm import tqdm
import os

# Read both CSV files
print("Reading CSV files...")
valid_pairs_3 = pd.read_csv('valid_pairs_3.csv')
valid_pairs_test = pd.read_csv('valid_pairs_test.csv')

# Combine the pairs from both files
all_pairs = pd.concat([valid_pairs_3, valid_pairs_test], ignore_index=True)

# Group pairs by design
design_pairs = all_pairs.groupby('design')

# Process each design
design_list = [1, 2, 3, 5, 6, 7, 9, 11, 14, 16, 18, 19]

for design in tqdm(design_list, desc="Processing designs"):
    print(f"\nProcessing design {design}")
    
    # Load original pyg data
    file_path = f"de_hnn/data/superblue/superblue_{design}/pyg_data.pkl"
    output_path = f"de_hnn/data/superblue/superblue_{design}/pyg_data_with_valid.pkl"
    
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
    
    # Calculate the next available net index (we'll use just one new net for all connections)
    dummy_net_idx = max(
        torch.max(source_to_net[1]).item(),
        torch.max(sink_to_net[1]).item()
    ) - max(torch.max(source_to_net[0]).item(), torch.max(sink_to_net[0]).item())
    
    # Create new edges for valid pairs
    new_source_edges_0 = []  # Sources
    new_source_edges_1 = []  # Nets (all pointing to dummy net)
    new_sink_edges_0 = []    # Sinks
    new_sink_edges_1 = []    # Nets (all pointing to dummy net)
    
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
    data['real_net_mask'] = real_net_mask  # Add mask to identify real vs dummy nets
    
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

print("\nProcessing complete! Modified data saved to pyg_data_with_valid.pkl files") 