import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Read the data
df = pd.read_csv('valid_pairs.csv')

# 1. Basic Statistics
print("=== Basic Statistics ===")
print(f"Total number of pairs: {len(df)}")
print(f"Number of unique sources: {df['source'].nunique()}")
print(f"Number of unique destinations: {df['destination'].nunique()}")
print("\nPairs per design:")
print(df['design'].value_counts().sort_index())

# 2. Feature Analysis
# Create separate dataframes for source and destination features
source_features = df[[col for col in df.columns if col.startswith('source_feature_')]]
dest_features = df[[col for col in df.columns if col.startswith('destination_feature_')]]

print("\n=== Feature Statistics ===")
print("\nSource Features Summary:")
print(source_features.describe())
print("\nDestination Features Summary:")
print(dest_features.describe())

# 3. Visualizations
plt.figure(figsize=(15, 10))

# Distribution of pairs across designs
plt.subplot(2, 2, 1)
df['design'].value_counts().sort_index().plot(kind='bar')
plt.title('Number of Pairs per Design')
plt.xlabel('Design Number')
plt.ylabel('Count')

# Source reuse analysis
plt.subplot(2, 2, 2)
source_counts = df['source'].value_counts()
plt.hist(source_counts.values[~np.isnan(source_counts.values)], bins=50)
plt.title('Distribution of Source Node Reuse')
plt.xlabel('Times a Source Node is Used')
plt.ylabel('Count')

# Destination reuse analysis
plt.subplot(2, 2, 3)

dest_counts = df['destination'].value_counts()
plt.hist(dest_counts.values[~np.isnan(dest_counts.values)], bins=50)
plt.title('Distribution of Destination Node Reuse')
plt.xlabel('Times a Destination Node is Used')
plt.ylabel('Count')

# Feature correlation heatmap
plt.subplot(2, 2, 4)
corr = source_features.corrwith(dest_features).sort_values(ascending=False)
corr = corr[~np.isnan(corr)]
plt.hist(corr, bins=20)
plt.title('Distribution of Source-Destination Feature Correlations')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('pairs_analysis.png')
plt.close()

# 4. Node Reuse Analysis
print("\n=== Node Reuse Analysis ===")
print("\nTop 10 Most Used Source Nodes:")
print(source_counts.head(10))
print("\nTop 10 Most Used Destination Nodes:")
print(dest_counts.head(10))

# 5. Feature Similarity Analysis
print("\n=== Feature Similarity Analysis ===")
# Calculate feature-wise correlation between source and destination
feature_correlations = []
for i in range(1, len(source_features.columns) + 1):
    source_col = f'source_feature_{i}'
    dest_col = f'destination_feature_{i}'
    corr = df[source_col].corr(df[dest_col])
    feature_correlations.append((i, corr))

# Sort by absolute correlation
feature_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
print("\nTop 10 Most Correlated Features between Source and Destination:")
for feat_num, corr in feature_correlations[:10]:
    print(f"Feature {feat_num}: {corr:.3f}")

# 6. Design-specific Analysis
print("\n=== Design-specific Analysis ===")
for design in df['design'].unique():
    design_df = df[df['design'] == design]
    print(f"\nDesign {design}:")
    print(f"Number of pairs: {len(design_df)}")
    print(f"Unique sources: {design_df['source'].nunique()}")
    print(f"Unique destinations: {design_df['destination'].nunique()}")
    
# 7. Save Summary Statistics
summary = {
    'total_pairs': int(len(df)),
    'unique_sources': int(df['source'].nunique()),
    'unique_destinations': int(df['destination'].nunique()),
    'pairs_per_design': {int(k): int(v) for k, v in df['design'].value_counts().to_dict().items()},
    'max_source_reuse': int(source_counts.max()),
    'max_dest_reuse': int(dest_counts.max()),
    'feature_correlations': {int(k): float(v) for k, v in dict(feature_correlations).items()}
}

# Save summary to file
import json
with open('pairs_analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)

print("\nAnalysis complete. Results saved to pairs_analysis_summary.json and pairs_analysis.png") 