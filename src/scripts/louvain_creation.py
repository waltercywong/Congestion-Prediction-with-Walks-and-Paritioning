import os
from tqdm import tqdm
import numpy as np
import networkx as nx
import community.community_louvain as community
import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Get the current directory (src/scripts)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (src)
parent_dir = os.path.dirname(current_dir)

# Define the data directory (data/superblue)
# Navigate from src/scripts to data/superblue using ".."
data_dir = os.path.join(parent_dir, "..", "data", "superblue")

def detect_communities_louvain(G):
    """
    Detects communities in the graph using the Louvain method.
    
    Parameters:
    G (networkx.Graph): An undirected graph.

    Returns:
    dict: Partition mapping nodes to community IDs.
    """
    return community.best_partition(G)

def compute_modularity(G, partition):
    """
    Compute the modularity of a given graph G and its partition efficiently.
    
    Parameters:
    G (networkx.Graph): An undirected graph.
    partition (dict): A dictionary where keys are node IDs and values are community IDs.
    
    Returns:
    float: Modularity score.
    """
    m = G.number_of_edges()  # Total number of edges
    degrees = dict(G.degree())  # Degree of each node
    community_map = {node: partition[node] for node in G.nodes()}  # Community lookup
    Q = 0.0  # Initialize modularity sum

    # Create a progress bar for edge iteration
    pbar = tqdm(total=G.number_of_edges(), desc="Processing edges", unit="edge")

    # Iterate only over existing edges
    for u, v in G.edges():
        A_uv = 1  # Edge exists
        P_uv = (degrees[u] * degrees[v]) / (2 * m)  # Expected edge probability

        # Only count if u and v belong to the same community
        if community_map[u] == community_map[v]:
            Q += (A_uv - P_uv)

        pbar.update(1)  # Update progress bar
        pbar.refresh()

    pbar.close()  # Close progress bar

    # Normalize by 2m
    Q /= (2 * m)
    
    return Q

def main():
    design_list = [1, 2, 3, 5, 6, 7, 9, 11, 14, 16, 18, 19]

    for design in tqdm(design_list, desc="Processing designs", position=0):
        print(f"\nProcessing design {design}", flush=True)
        # Construct file paths using data_dir
        file_path = os.path.join(data_dir, f"superblue_{design}", "bipartite.pkl")
        com_path = os.path.join(data_dir, f"superblue_{design}", "community.pkl")
        mod_path = os.path.join(data_dir, f"superblue_{design}", "modularity.pkl")

        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        # Create an empty undirected graph
        G = nx.Graph()

        # Extract edge indices
        edges = zip(data['edge_index'][0], data['edge_index'][1])

        # Add edges to the graph
        G.add_edges_from(edges)

        # Step 1: Automatically detect communities
        partition = detect_communities_louvain(G)
        # print("Detected communities:", partition)
        with open(com_path, "wb") as file:
            pickle.dump(partition, file)

        # Step 2: Compute modularity
        modularity = compute_modularity(G, partition)
        print(f"Detected Modularity: {modularity:.4f}")
        with open(mod_path, "wb") as file:
            pickle.dump(modularity, file)

if __name__ == "__main__":
    main()