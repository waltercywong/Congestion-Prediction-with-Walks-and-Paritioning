import os
import kahypar as kahypar
import pickle
import torch
from tqdm import tqdm
from collections import defaultdict

def load_hypergraph_data(file_path):
    """
    Load hypergraph data from a file.
    This function should be customized to match the format of your data.
    """
    with open(file_path, 'rb') as file:
        data = torch.load(file)
    
    # Given input lists
    hyperedge_id_list = data['edge_index_sink_to_net'][1].tolist() + data['edge_index_source_to_net'][1].tolist()
    node_id_list = data['edge_index_sink_to_net'][0].tolist() + data['edge_index_source_to_net'][0].tolist()

    # Step 1: Organize nodes by hyperedge
    hyperedge_dict = defaultdict(set)
    for hyperedge, node in zip(hyperedge_id_list, node_id_list):
        hyperedge_dict[hyperedge].add(node)

    # Step 2: Convert to hyperedges & hyperedge_indices format
    hyperedges = []
    hyperedge_indices = []
    current_index = 0

    for hyperedge in sorted(hyperedge_dict.keys()):  # Ensure order
        nodes = sorted(hyperedge_dict[hyperedge])
        if not nodes:  # Skip empty hyperedges
            print(f"Skipping empty hyperedge: {hyperedge}")
            continue
        hyperedge_indices.append(current_index)
        hyperedges.extend(nodes)
        current_index += len(nodes)

    # Example structure, adjust according to your data
    num_nodes = data['node_features'].shape[0]
    num_nets = data['net_features'].shape[0]
    #hyperedges = data['edge_index_sink_to_net'][1].tolist() + data['edge_index_source_to_net'][1].tolist()
    #hyperedge_indices = data['edge_index_sink_to_net'][0].tolist() + data['edge_index_source_to_net'][0].tolist()
    node_weights = [1] * num_nodes
    edge_weights = [1] * num_nets
    #node_weights = data['node_weights']
    #edge_weights = data['edge_weights']
    
    return num_nodes, num_nets, hyperedge_indices, hyperedges, node_weights, edge_weights

def partition_hypergraph(file_path, config_path, k=2, epsilon=0.03):
    # Load hypergraph data
    num_nodes, num_nets, hyperedge_indices, hyperedges, node_weights, edge_weights = load_hypergraph_data(file_path)

    # Create a hypergraph
    hypergraph = kahypar.Hypergraph(num_nodes, num_nets, hyperedge_indices, hyperedges, k, edge_weights, node_weights)

    # Set up the context
    context = kahypar.Context()
    context.loadINIconfiguration(config_path)
    context.setK(k)
    context.setEpsilon(epsilon)

    # Perform partitioning
    kahypar.partition(hypergraph, context)

    # Extract partitioning result
    partition = hypergraph.getPartition()
    print("Partitioning result:", partition)

    return partition

def main():
    design_list = [1, 2, 3, 5, 6, 7, 9, 11, 14, 16]
    config_path = "./src/config/cut_kKaHyPar_sea20.ini"

    for design in tqdm(design_list, desc="Processing designs", position=0):
        print(f"\nProcessing design {design}", flush=True)
        file_path = f"./data/superblue/superblue_{design}/pyg_data.pkl"
        partition_result_path = f"./data/superblue/superblue_{design}/kahypar_part_dict.pkl"

        # Perform partitioning
        partition = partition_hypergraph(file_path, config_path)

        # Save the partition result
        with open(partition_result_path, 'wb') as file:
            pickle.dump(partition, file)

if __name__ == "__main__":
    main() 