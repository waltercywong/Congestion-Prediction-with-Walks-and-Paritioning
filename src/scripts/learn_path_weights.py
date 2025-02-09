import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pickle

def analyze_valid_paths(valid_pairs_df):
    """
    Analyze valid paths to learn what feature patterns lead to successful connections
    
    Args:
        valid_pairs_df: DataFrame containing valid pairs and their paths
    Returns:
        dict: Feature importance weights and statistics
    """
    print("Analyzing valid paths to learn feature patterns...")
    
    # Initialize statistics
    feature_transitions = []
    path_lengths = []
    successful_patterns = []
    
    # Analyze each valid path
    for _, row in tqdm(valid_pairs_df.iterrows(), total=len(valid_pairs_df)):
        path = row['path']
        path_lengths.append(len(path))
        
        # Analyze feature transitions along path
        for i in range(len(path)-1):
            source_features = np.array([row[f'source_feature_{j}'] for j in range(1, 46)])
            dest_features = np.array([row[f'destination_feature_{j}'] for j in range(1, 46)])
            
            # Record feature differences that led to successful paths
            feature_diff = np.abs(source_features - dest_features)
            feature_transitions.append(feature_diff)
            
            # Record successful feature patterns
            pattern = {
                'source_features': source_features,
                'dest_features': dest_features,
                'path_length': len(path)
            }
            successful_patterns.append(pattern)
    
    # Convert to numpy arrays
    feature_transitions = np.array(feature_transitions)
    
    # Calculate feature importance based on consistency in transitions
    feature_importance = 1.0 / (np.mean(feature_transitions, axis=0) + 1e-6)
    feature_importance = feature_importance / np.sum(feature_importance)
    
    # Group features by importance
    sorted_indices = np.argsort(feature_importance)[::-1]
    feature_groups = {
        'primary': sorted_indices[:5],    # Top 5 most important
        'secondary': sorted_indices[5:10], # Next 5
        'tertiary': sorted_indices[10:15]  # Next 5
    }
    
    # Calculate transition statistics
    transition_stats = {
        'mean_path_length': np.mean(path_lengths),
        'std_path_length': np.std(path_lengths),
        'feature_importance': feature_importance,
        'feature_groups': feature_groups,
        'transition_means': np.mean(feature_transitions, axis=0),
        'transition_stds': np.std(feature_transitions, axis=0)
    }
    
    return transition_stats

def calculate_edge_weight(node1_features, node2_features, transition_stats):
    """
    Calculate edge weight based on learned feature patterns
    
    Args:
        node1_features: Features of first node
        node2_features: Features of second node
        transition_stats: Statistics from valid path analysis
    Returns:
        float: Edge weight between 0 and 1
    """
    feature_diff = np.abs(node1_features - node2_features)
    
    # Weight based on feature groups
    weights = np.zeros_like(feature_diff)
    
    # Primary features (highest weight)
    weights[transition_stats['feature_groups']['primary']] = 0.5
    
    # Secondary features
    weights[transition_stats['feature_groups']['secondary']] = 0.3
    
    # Tertiary features
    weights[transition_stats['feature_groups']['tertiary']] = 0.2
    
    # Calculate similarity scores
    mean_diffs = transition_stats['transition_means']
    std_diffs = transition_stats['transition_stds']
    
    # Normalize feature differences
    normalized_diff = (feature_diff - mean_diffs) / (std_diffs + 1e-6)
    
    # Calculate weight as similarity to learned patterns
    weight = np.exp(-np.sum(weights * np.abs(normalized_diff)))
    
    return weight

def main():
    """Learn path weights from valid pairs data"""
    print("Loading valid pairs data...")
    valid_pairs_df = pd.read_csv('valid_pairs_2.csv')
    
    # Convert string representation of paths to lists
    valid_pairs_df['path'] = valid_pairs_df['path'].apply(eval)
    
    # Learn transition patterns
    transition_stats = analyze_valid_paths(valid_pairs_df)
    
    # Save learned patterns
    print("\nSaving learned transition patterns...")
    with open('transition_stats.pkl', 'wb') as f:
        pickle.dump(transition_stats, f)
    
    print("\nMost important features for path transitions:")
    for group, indices in transition_stats['feature_groups'].items():
        print(f"\n{group.capitalize()} features:")
        for idx in indices:
            importance = transition_stats['feature_importance'][idx]
            print(f"Feature {idx+1}: {importance:.4f}")
    
    print("\nResults saved to transition_stats.pkl")

if __name__ == "__main__":
    main() 