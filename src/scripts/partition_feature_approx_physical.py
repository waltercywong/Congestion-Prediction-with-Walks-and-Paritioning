from tqdm import tqdm
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import pickle
import torch

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def partition_by_feature_similarity(node_features, num_clusters=81):
    """
    Partitions nodes based on feature similarity using K-Means clustering.
    Specifically, we use the node features 7 through 15, which may approximate the physical closeness.

    Parameters:
    node_features (np.ndarray): Node features array.
    num_clusters (int): Number of clusters to form.
    
    Returns:
    dict: Partition mapping nodes to cluster IDs.
    """
    # Extract features 7 through 15
    # These features possibly? correspond to the physical closeness
    #features = node_features[:, 6:15]
    # Testing node type + top10 eigen only
    features = node_features[:, [0, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]]
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    
    # Create a partition dictionary
    partition = {i: label for i, label in enumerate(labels)}
    return partition


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
    design_list = [1, 2, 3, 5, 6, 7, 9, 11, 14, 16]

    for design in tqdm(design_list, desc="Processing designs", position=0):
        print(f"\nProcessing design {design}", flush=True)
        file_path = f"./data/superblue/superblue_{design}/pyg_data.pkl"
        part_dict_path = f"./data/superblue/superblue_{design}/feature_approx_physical_part_dict.pkl"
        mod_path = f"./data/superblue/superblue_{design}/feature_approx_physical_modularity.pkl"
        
        # Load the PyTorch Geometric data
        data = torch.load(file_path)

        # Create an empty undirected graph
        #G = nx.Graph()

        # Extract edge indices
        #edges = data.edge_index.numpy().T  # Convert to numpy and transpose for (u, v) pairs

        # Add edges to the graph
        #G.add_edges_from(edges)

        # Load node features
        node_features = data.node_features.numpy()  # Assuming node_features is a tensor

        # Step 1: Partition nodes based on feature similarity
        partition = partition_by_feature_similarity(node_features)
        with open(part_dict_path, "wb") as file:
            pickle.dump(partition, file)

        # Step 2: Compute modularity
        #modularity = compute_modularity(G, partition)
        #print(f"Detected Modularity: {modularity:.4f}")
        #with open(mod_path, "wb") as file:
        #    pickle.dump(modularity, file)


if __name__ == "__main__":
    main() 

