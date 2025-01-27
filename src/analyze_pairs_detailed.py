import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import networkx as nx

# Read the data
df = pd.read_csv('valid_pairs.csv')

# 1. Path Length Analysis
print("=== Path Length Analysis ===")
path_lengths = defaultdict(int)
for _, row in df.iterrows():
    source = row['source']
    dest = row['destination']
    design = row['design']
    path_lengths[(source, dest, design)] = 5  # Since these are 5-hop paths

print(f"Average path length: {np.mean(list(path_lengths.values())):.2f}")
print(f"Total unique paths: {len(path_lengths)}")

# 2. Network Analysis per Design
print("\n=== Network Analysis per Design ===")
for design in df['design'].unique():
    design_df = df[df['design'] == design]
    
    # Create a directed graph
    G = nx.DiGraph()
    for _, row in design_df.iterrows():
        G.add_edge(row['source'], row['destination'])
    
    print(f"\nDesign {design}:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    # Calculate degree statistics
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    
    print(f"Average in-degree: {np.mean(in_degrees):.2f}")
    print(f"Average out-degree: {np.mean(out_degrees):.2f}")
    print(f"Max in-degree: {max(in_degrees)}")
    print(f"Max out-degree: {max(out_degrees)}")

# 3. Feature Pattern Analysis
print("\n=== Feature Pattern Analysis ===")
source_features = df[[col for col in df.columns if col.startswith('source_feature_')]]
dest_features = df[[col for col in df.columns if col.startswith('destination_feature_')]]

# Calculate feature differences
feature_diffs = source_features.values - dest_features.values
feature_diff_means = np.mean(np.abs(feature_diffs), axis=0)

# Plot feature differences
plt.figure(figsize=(15, 5))
plt.bar(range(len(feature_diff_means)), feature_diff_means)
plt.title('Average Absolute Difference Between Source and Destination Features')
plt.xlabel('Feature Index')
plt.ylabel('Mean Absolute Difference')
plt.savefig('feature_differences.png')
plt.close()

# 4. Source-Destination Distance Distribution
print("\n=== Source-Destination Analysis ===")
for design in df['design'].unique():
    design_df = df[df['design'] == design]
    
    # Create scatter plot of source vs destination nodes
    plt.figure(figsize=(10, 10))
    plt.scatter(design_df['source'], design_df['destination'], alpha=0.1)
    plt.title(f'Source vs Destination Nodes - Design {design}')
    plt.xlabel('Source Node ID')
    plt.ylabel('Destination Node ID')
    plt.savefig(f'source_dest_scatter_design_{design}.png')
    plt.close()

# 5. Feature Clustering
print("\n=== Feature Clustering Analysis ===")
from sklearn.cluster import KMeans

# Combine source and destination features
combined_features = np.concatenate([source_features.values, dest_features.values])

# Perform K-means clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(combined_features)

# Analyze cluster distributions
source_clusters = clusters[:len(source_features)]
dest_clusters = clusters[len(source_features):]

print("\nCluster Distribution:")
for i in range(n_clusters):
    print(f"\nCluster {i}:")
    print(f"Sources: {np.sum(source_clusters == i)}")
    print(f"Destinations: {np.sum(dest_clusters == i)}")

# 6. Save Detailed Statistics
detailed_stats = {
    'per_design': {},
    'feature_patterns': {
        'mean_differences': feature_diff_means.tolist(),
        'correlation_matrix': source_features.corr().to_dict()
    },
    'clustering': {
        f'cluster_{i}': {
            'source_count': int(np.sum(source_clusters == i)),
            'dest_count': int(np.sum(dest_clusters == i))
        } for i in range(n_clusters)
    }
}

for design in df['design'].unique():
    design_df = df[df['design'] == design]
    G = nx.DiGraph()
    for _, row in design_df.iterrows():
        G.add_edge(row['source'], row['destination'])
    
    detailed_stats['per_design'][int(design)] = {
        'num_pairs': len(design_df),
        'unique_sources': int(design_df['source'].nunique()),
        'unique_destinations': int(design_df['destination'].nunique()),
        'avg_in_degree': float(np.mean([d for n, d in G.in_degree()])),
        'avg_out_degree': float(np.mean([d for n, d in G.out_degree()]))
    }

import json
with open('detailed_analysis.json', 'w') as f:
    json.dump(detailed_stats, f, indent=4)

print("\nDetailed analysis complete. Results saved to detailed_analysis.json") 