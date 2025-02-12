import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress, entropy
import pickle

def calculate_feature_entropy(feature_values, num_bins=10):
    """Calculate entropy of feature values along paths with safety checks"""
    if len(feature_values) < 2:
        return 0.0
    
    # Check for non-zero variance
    if np.std(feature_values) == 0:
        return 0.0
        
    hist, _ = np.histogram(feature_values, bins=num_bins, density=True)
    hist = hist[hist > 0]  # Remove zero probabilities
    if len(hist) == 0:
        return 0.0
    return entropy(hist)

def analyze_feature_patterns(feature_values, all_path_positions, design_values, path_efficiencies):
    """Analyze feature patterns including entropy"""
    pattern_stats = {}
    
    # Calculate basic statistics
    valid_mask = ~np.isnan(feature_values)
    if np.sum(valid_mask) > 1:
        # Gradient and consistency analysis
        slopes = []
        r_values = []
        for design in np.unique(design_values):
            mask = (design_values == design) & valid_mask
            if np.sum(mask) > 1:
                x = all_path_positions[mask]
                y = feature_values[mask]
                if len(np.unique(y)) > 1 and np.std(y) > 0:
                    slope, _, r_value, _, _ = linregress(x, y)
                    slopes.append(slope)
                    r_values.append(r_value)
        
        # Entropy analysis
        feature_entropy = calculate_feature_entropy(feature_values[valid_mask])
        
        # Position-based entropy
        position_entropy = []
        for pos in np.linspace(0, 1, 10):
            pos_mask = (np.abs(all_path_positions - pos) < 0.1) & valid_mask
            if np.sum(pos_mask) > 1:
                pos_entropy = calculate_feature_entropy(feature_values[pos_mask])
                position_entropy.append(pos_entropy)
            else:
                position_entropy.append(0.0)
        
        # Efficiency correlation
        valid_both = valid_mask & ~np.isnan(path_efficiencies)
        if np.sum(valid_both) > 1 and np.std(feature_values[valid_both]) > 0 and np.std(path_efficiencies[valid_both]) > 0:
            eff_corr = np.corrcoef(feature_values[valid_both], path_efficiencies[valid_both])[0,1]
        else:
            eff_corr = 0.0
        
        # Feature stability
        feat_std = np.std(feature_values[valid_mask]) if np.sum(valid_mask) > 1 else 1.0
        feat_stability = 1.0 / (feat_std + 1.0)
        
        pattern_stats = {
            'mean_slope': float(np.mean(slopes)) if slopes else 0.0,
            'slope_std': float(np.std(slopes)) if slopes else 0.0,
            'mean_r2': float(np.mean(np.abs(r_values))) if r_values else 0.0,
            'r2_std': float(np.std(np.abs(r_values))) if r_values else 0.0,
            'entropy': float(feature_entropy),
            'position_entropy': [float(e) for e in position_entropy],
            'efficiency_correlation': float(eff_corr),
            'stability': float(feat_stability)
        }
        
        # Calculate importance score
        pattern_stats['importance_score'] = (
            0.3 * pattern_stats['mean_r2'] +  # Consistency weight
            0.3 * abs(pattern_stats['efficiency_correlation']) +  # Efficiency correlation weight
            0.2 * (1.0 - pattern_stats['entropy'] / (np.log(10) + 1e-10)) +  # Entropy weight
            0.2 * pattern_stats['stability']  # Stability weight
        )
        
    return pattern_stats

def analyze_valid_paths(valid_pairs_df):
    """Analyze valid paths to learn feature patterns and calculate weights"""
    print("Analyzing valid paths to learn feature patterns...")
    
    # Get number of features
    num_features = sum(1 for col in valid_pairs_df.columns if col.startswith('source_feature_'))
    
    # Initialize arrays for feature analysis
    feature_values = [[] for _ in range(num_features)]
    all_path_positions = []
    design_values = []
    path_efficiencies = []
    
    # Process each path
    print("Processing paths...")
    for _, row in tqdm(valid_pairs_df.iterrows(), total=len(valid_pairs_df), desc="Processing paths"):
        path = row['path']
        path_length = len(path)
        efficiency = row['path_efficiency']
        
        for pos, node in enumerate(path):
            norm_pos = pos / (path_length - 1)
            all_path_positions.append(norm_pos)
            design_values.append(row['design'])
            path_efficiencies.append(efficiency)
            
            if pos == 0:
                for feat_idx in range(num_features):
                    feature_values[feat_idx].append(row[f'source_feature_{feat_idx+1}'])
            elif pos == len(path) - 1:
                for feat_idx in range(num_features):
                    feature_values[feat_idx].append(row[f'destination_feature_{feat_idx+1}'])
    
    # Convert to numpy arrays
    all_path_positions = np.array(all_path_positions)
    design_values = np.array(design_values)
    path_efficiencies = np.array(path_efficiencies)
    feature_values = [np.array(feat_vals) for feat_vals in feature_values]
    
    # Calculate feature patterns
    print("\nAnalyzing feature patterns...")
    feature_stats = {}
    for feat_idx in tqdm(range(num_features), desc="Analyzing features"):
        if not np.all(np.isnan(feature_values[feat_idx])):
            feature_stats[feat_idx] = analyze_feature_patterns(
                feature_values[feat_idx],
                all_path_positions,
                design_values,
                path_efficiencies
            )
    
    return feature_stats

def main():
    """Learn path weights from valid pairs data"""
    print("Loading valid pairs data...")
    valid_pairs_df = pd.read_csv('valid_pairs_2.csv')
    
    # Convert string representation of paths to lists
    valid_pairs_df['path'] = valid_pairs_df['path'].apply(eval)
    
    # Calculate path efficiency if not present
    if 'path_efficiency' not in valid_pairs_df.columns:
        valid_pairs_df['path_efficiency'] = valid_pairs_df['physical_distance'] / valid_pairs_df['path_length']
    
    # Learn feature patterns and weights
    feature_stats = analyze_valid_paths(valid_pairs_df)
    
    # Save learned patterns
    print("\nSaving learned weights and patterns...")
    with open('feature_weights.pkl', 'wb') as f:
        pickle.dump(feature_stats, f)
    
    # Print feature importance summary
    print("\nFeature importance summary:")
    sorted_features = sorted(
        [(i, stats['importance_score']) for i, stats in feature_stats.items() if stats],
        key=lambda x: x[1],
        reverse=True
    )
    
    print("\nTop 10 most important features:")
    for feat_idx, importance in sorted_features[:10]:
        stats = feature_stats[feat_idx]
        print(f"\nFeature {feat_idx+1}:")
        print(f"  - Importance score: {importance:.4f}")
        print(f"  - Gradient strength: {stats['mean_slope']:.4f}")
        print(f"  - Consistency (R²): {stats['mean_r2']:.4f}")
        print(f"  - Efficiency correlation: {abs(stats['efficiency_correlation']):.4f}")
        print(f"  - Stability: {stats['stability']:.4f}")
    
    print("\nResults saved to feature_weights.pkl")

if __name__ == "__main__":
    main() 