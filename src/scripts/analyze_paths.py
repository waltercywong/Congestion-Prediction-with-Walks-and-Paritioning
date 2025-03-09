import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from collections import defaultdict
import json
import warnings
import pickle
import torch
from scipy.stats import linregress, entropy
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Create directory for plots if it doesn't exist
os.makedirs('path_analysis_plots', exist_ok=True)

def load_physical_positions(design):
    """Load physical positions from pyg_data.pkl file for a given design"""
    try:
        with open(f'de_hnn/data/superblue/superblue_{design}/pyg_data.pkl', 'rb') as f:
            pyg_data = torch.load(f)
        return pyg_data.pos_lst
    except (FileNotFoundError, AttributeError) as e:
        print(f"Warning: Could not load positions for design {design}: {e}")
        return None

def euclidean_distance(source_pos, dest_pos):
    """Calculate Euclidean distance between two position vectors"""
    if source_pos is None or dest_pos is None:
        return None
    
    # Convert to numpy if they are torch tensors
    if torch.is_tensor(source_pos):
        source_pos = source_pos.cpu().numpy()
    if torch.is_tensor(dest_pos):
        dest_pos = dest_pos.cpu().numpy()
        
    return float(np.sqrt(np.sum((source_pos - dest_pos) ** 2)))

def calculate_path_metrics(row, pos_dict):
    """Calculate physical distance and path metrics using loaded positions"""
    design = row['design']
    path = row['path']
    
    if design not in pos_dict or pos_dict[design] is None:
        return None, None
    
    positions = pos_dict[design]
    
    # Calculate physical distance between source and destination
    try:
        # Convert indices to integers to avoid indexing issues
        source_idx = int(path[0])
        dest_idx = int(path[-1])
        
        source_pos = positions[source_idx]
        dest_pos = positions[dest_idx]
        physical_dist = euclidean_distance(source_pos, dest_pos)
        
        if physical_dist is not None and len(path) > 0:
            path_efficiency = physical_dist / len(path)
            return physical_dist, path_efficiency
    except (IndexError, KeyError, TypeError) as e:
        print(f"Warning: Error calculating metrics for path in design {design}: {e}")
    
    return None, None

def calculate_path_directness(row, pos_dict):
    """Calculate path directness using physical positions"""
    path = row['path']
    design = row['design']
    
    if design not in pos_dict or pos_dict[design] is None or len(path) < 2:
        return None
        
    positions = pos_dict[design]
    
    try:
        # Convert indices to integers
        source_idx = int(path[0])
        dest_idx = int(path[-1])
        
        source_pos = positions[source_idx]
        dest_pos = positions[dest_idx]
        direct_distance = euclidean_distance(source_pos, dest_pos)
        
        # Calculate total path distance
        total_path_distance = 0
        for i in range(len(path)-1):
            pos1 = positions[int(path[i])]
            pos2 = positions[int(path[i+1])]
            segment_dist = euclidean_distance(pos1, pos2)
            if segment_dist is not None:
                total_path_distance += segment_dist
        
        if total_path_distance > 0 and direct_distance is not None:
            return direct_distance / total_path_distance
            
    except (IndexError, KeyError, TypeError) as e:
        print(f"Warning: Error calculating path directness for design {design}: {e}")
    
    return None

def calculate_feature_entropy(feature_values, num_bins=10):
    """Calculate entropy of feature values along paths"""
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

def analyze_feature_patterns(feature_values, path_positions, design_values, path_efficiencies):
    """Analyze feature patterns including entropy"""
    pattern_stats = {}
    
    valid_mask = ~np.isnan(feature_values)
    if np.sum(valid_mask) > 1:
        # Calculate entropy
        feature_entropy = calculate_feature_entropy(feature_values[valid_mask])
        
        # Calculate position-based entropy
        position_entropy = []
        for pos in np.linspace(0, 1, 10):
            pos_mask = (np.abs(path_positions - pos) < 0.1) & valid_mask
            if np.sum(pos_mask) > 1:
                pos_entropy = calculate_feature_entropy(feature_values[pos_mask])
                position_entropy.append(pos_entropy)
            else:
                position_entropy.append(0.0)
        
        # Calculate feature stability
        feat_std = np.std(feature_values[valid_mask])
        feat_stability = 1.0 / (feat_std + 1.0)
        
        # Calculate importance score
        importance_score = (
            0.4 * feat_stability +  # Stability weight
            0.3 * (1.0 - feature_entropy / np.log(10)) +  # Entropy weight
            0.3 * (1.0 - np.mean(position_entropy) / np.log(10))  # Position entropy weight
        )
        
        pattern_stats = {
            'entropy': float(feature_entropy),
            'position_entropy': position_entropy,
            'stability': float(feat_stability),
            'importance_score': float(importance_score)
        }
        
    return pattern_stats

# Helper function for JSON serialization
def serialize_numpy(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

# Read the valid pairs data
print("Loading data...")
df = pd.read_csv('valid_pairs_3.csv')

# Convert string representation of paths to lists
df['path'] = df['path'].apply(literal_eval)

# Calculate path lengths
df['path_length'] = df['path'].apply(len)

# Create design to index mapping
unique_designs = sorted(df['design'].unique())
design_to_idx = {design: idx for idx, design in enumerate(unique_designs)}

# Load physical positions for all designs
print("Loading physical positions...")
pos_dict = {}
for design in tqdm(unique_designs, desc="Loading physical positions"):
    pos_dict[design] = load_physical_positions(design)

# Calculate physical distances and path metrics
print("Calculating path metrics...")
metrics = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating path metrics"):
    metrics.append(calculate_path_metrics(row, pos_dict))
df['physical_distance'], df['path_efficiency'] = zip(*metrics)

# Calculate path directness
print("Calculating path directness...")
df['path_directness'] = df.apply(lambda row: calculate_path_directness(row, pos_dict), axis=1)

# Remove rows with invalid metrics
df = df.dropna(subset=['physical_distance', 'path_efficiency', 'path_directness'])

# Get the number of features per node
num_features = sum(1 for col in df.columns if col.startswith('source_feature_'))

# Initialize lists to store feature values along paths
print("Processing path features...")
all_path_positions = []
feature_values = [[] for _ in range(num_features)]
design_values = []
path_indices = []
path_efficiencies = []

# Analyze each path
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing paths"):
    path = row['path']
    path_length = len(path)
    efficiency = row['path_efficiency']
    
    for pos, node in enumerate(path):
        norm_pos = pos / (path_length - 1)
        all_path_positions.append(norm_pos)
        path_indices.append(int(norm_pos * 100))
        design_values.append(row['design'])
        path_efficiencies.append(efficiency)
        
        if pos == 0:
            for feat_idx in range(num_features):
                feature_values[feat_idx].append(row[f'source_feature_{feat_idx+1}'])
        elif pos == len(path) - 1:
            for feat_idx in range(num_features):
                feature_values[feat_idx].append(row[f'destination_feature_{feat_idx+1}'])
        else:
            prev_occurrence = df[df['destination'] == node].iloc[0] if len(df[df['destination'] == node]) > 0 else None
            if prev_occurrence is not None:
                for feat_idx in range(num_features):
                    feature_values[feat_idx].append(prev_occurrence[f'destination_feature_{feat_idx+1}'])
            else:
                next_occurrence = df[df['source'] == node].iloc[0] if len(df[df['source'] == node]) > 0 else None
                if next_occurrence is not None:
                    for feat_idx in range(num_features):
                        feature_values[feat_idx].append(next_occurrence[f'source_feature_{feat_idx+1}'])
                else:
                    for feat_idx in range(num_features):
                        feature_values[feat_idx].append(np.nan)

# Convert to numpy arrays
all_path_positions = np.array(all_path_positions)
feature_values = [np.array(feat_vals) for feat_vals in feature_values]
design_values = np.array(design_values)
path_indices = np.array(path_indices)
path_efficiencies = np.array(path_efficiencies)

# Calculate feature gradients and identify key features
print("\nCalculating feature gradients and identifying key features...")
key_features = []
feature_stats = {}

for feat_idx in range(num_features):
    if not np.all(np.isnan(feature_values[feat_idx])):
        slopes = []
        r_values = []
        
        # Remove NaN values before calculations
        valid_features = ~np.isnan(feature_values[feat_idx])
        
        for design in unique_designs:
            mask = (design_values == design) & valid_features
            if np.sum(mask) > 1:  # Need at least 2 points for regression
                x = all_path_positions[mask]
                y = feature_values[feat_idx][mask]
                
                if len(np.unique(y)) > 1:  # Check if we have varying values
                    slope, _, r_value, _, _ = linregress(x, y)
                    slopes.append(slope)
                    r_values.append(r_value)
        
        if slopes and r_values:  # Only if we have valid calculations
            feature_stats[feat_idx] = {
                'slopes': slopes,
                'r_values': r_values,
                'mean_slope': float(np.mean(slopes)),
                'mean_r2': float(np.mean(np.abs(r_values))),
                'value_range': [
                    float(np.nanmin(feature_values[feat_idx])),
                    float(np.nanmax(feature_values[feat_idx]))
                ]
            }
            print(f'Features {feat_idx+1} has a mean r2 of {np.mean(np.abs(r_values))}')
            # Lower the threshold for key features
            if np.mean(np.abs(r_values)) > 0.1:  # Changed from 0.2 to 0.1
                key_features.append(feat_idx + 1)

print('--------------------------------')
print(key_features)
print('--------------------------------')
# Calculate feature importance
print("\nCalculating feature importance...")
feature_importance = np.zeros(num_features)

for feat_idx in range(num_features):
    if feat_idx in feature_stats:  # Only process features with valid statistics
        # 1. Gradient consistency (R² value)
        r2_score = max(0.0, feature_stats[feat_idx]['mean_r2'])
        
        # 2. Correlation with path efficiency
        feat_diff = np.abs(df[f'source_feature_{feat_idx+1}'] - 
                          df[f'destination_feature_{feat_idx+1}'])
        valid_mask = ~np.isnan(feat_diff) & ~np.isnan(df['path_efficiency'])
        if np.sum(valid_mask) > 1:
            eff_corr = abs(np.corrcoef(feat_diff[valid_mask], 
                                     df['path_efficiency'][valid_mask])[0,1])
        else:
            eff_corr = 0.0
        
        # 3. Feature stability (with better handling of edge cases)
        feat_std = np.nanstd(feature_values[feat_idx])
        feat_stability = 1.0 / (feat_std + 1.0) if feat_std > 0 else 0.0
        
        # Combine metrics with safety checks
        if not (np.isnan(r2_score) or np.isnan(eff_corr) or np.isnan(feat_stability)):
            feature_importance[feat_idx] = (0.4 * r2_score + 
                                          0.4 * eff_corr + 
                                          0.2 * feat_stability)

# Normalize importance scores (avoiding division by zero)
total_importance = np.sum(feature_importance)
if total_importance > 0:
    feature_importance = feature_importance / total_importance

# Print feature analysis results
print("\n3. Feature Pattern Analysis:")
print("\na) Key features with consistent gradients:")
for feat_idx in sorted(key_features):
    stats = feature_stats[feat_idx-1]
    print(f"\nFeature {feat_idx}:")
    print(f"   - Average gradient: {stats['mean_slope']:.3f}")
    print(f"   - Gradient consistency (R²): {stats['mean_r2']:.3f}")
    print(f"   - Value range: [{stats['value_range'][0]:.3f}, {stats['value_range'][1]:.3f}]")
    print(f"   - Importance for path finding: {feature_importance[feat_idx-1]:.3f}")

# Sort features by importance
feature_importance_pairs = [(i+1, imp) for i, imp in enumerate(feature_importance) if i in feature_stats]
sorted_features = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)

print("\n4. Recommendations for Network Enhancement:")
print("\na) Primary features for dummy net connections:")
print("   Based on the analysis, the following features show the strongest patterns:")

# Print top features
for feat_idx, importance in sorted_features[:10]:
    stats = feature_stats[feat_idx-1]
    print(f"\n   Feature {feat_idx}:")
    print(f"   - Importance score: {importance:.3f}")
    print(f"   - Gradient strength: {stats['mean_slope']:.3f}")
    print(f"   - Consistency (R²): {stats['mean_r2']:.3f}")

print("\nb) Suggested weighting strategy for random walks:")
print("   1. Primary weights:")
for feat_idx, importance in sorted_features:
    stats = feature_stats[feat_idx-1]
    # New weighting formula that combines metrics additively instead of multiplicatively
    weight = (
        0.4 * importance +  # Overall importance score
        0.3 * abs(stats['mean_slope']) +  # Contribution from gradient strength
        0.3 * stats['mean_r2']  # Contribution from consistency
    )
    print(f"      - Feature {feat_idx}: weight = {weight:.3f}")

print("\n   2. Secondary weights (for fine-tuning):")
for feat_idx, importance in sorted_features[3:7]:
    stats = feature_stats[feat_idx-1]
    # Same formula but scaled down for secondary features
    weight = 0.5 * (
        0.4 * importance +
        0.3 * abs(stats['mean_slope']) +
        0.3 * stats['mean_r2']
    )
    print(f"      - Feature {feat_idx}: weight = {weight:.3f}")

# Define design statistics
design_stats = {}
for design in unique_designs:
    design_paths = df[df['design'] == design]
    design_stats[str(design)] = {
        'path_count': len(design_paths),
        'mean_length': float(design_paths['path_length'].mean()),
        'std_length': float(design_paths['path_length'].std()),
        'mean_efficiency': float(design_paths['path_efficiency'].mean()),
        'std_efficiency': float(design_paths['path_efficiency'].std()),
        'mean_directness': float(design_paths['path_directness'].mean()),
        'std_directness': float(design_paths['path_directness'].std())
    }

# Define efficiency statistics (fixed version)
efficiency_stats = {
    'overall': df['path_efficiency'].describe().to_dict(),
    'by_design': {str(design): df[df['design'] == design]['path_efficiency'].describe().to_dict() 
                 for design in unique_designs},
}

# Add length correlation with safety check
if len(df) > 1 and df['path_length'].std() > 0 and df['path_efficiency'].std() > 0:
    efficiency_stats['length_correlation'] = float(np.corrcoef(df['path_length'], df['path_efficiency'])[0,1])
else:
    efficiency_stats['length_correlation'] = 0.0

# Calculate feature correlations safely
feature_correlations = {}
for idx in range(num_features):
    try:
        source_feat = df[f'source_feature_{idx+1}']
        dest_feat = df[f'destination_feature_{idx+1}']
        feat_diff = source_feat - dest_feat
        
        # Check for valid data and non-zero variance
        if len(feat_diff) > 1 and feat_diff.std() > 0 and df['path_efficiency'].std() > 0:
            corr = np.corrcoef(feat_diff, df['path_efficiency'])[0,1]
            feature_correlations[f'feature_{idx+1}'] = float(corr)
        else:
            feature_correlations[f'feature_{idx+1}'] = 0.0
    except (KeyError, ValueError):
        feature_correlations[f'feature_{idx+1}'] = 0.0

efficiency_stats['feature_correlations'] = feature_correlations

# Calculate feature patterns
print("\nCalculating feature patterns...")
feature_patterns = {
    'high_efficiency': {},
    'low_efficiency': {}
}

# Define high and low efficiency thresholds with safety checks
if len(df) > 0 and not df['path_efficiency'].empty:
    try:
        efficiency_threshold_high = np.percentile(df['path_efficiency'].dropna(), 90)
        efficiency_threshold_low = np.percentile(df['path_efficiency'].dropna(), 10)
        
        # Get high and low efficiency paths
        high_efficiency_paths = df[df['path_efficiency'] >= efficiency_threshold_high]
        low_efficiency_paths = df[df['path_efficiency'] <= efficiency_threshold_low]
        
        # Analyze patterns for each feature only if we have valid paths
        if not high_efficiency_paths.empty and not low_efficiency_paths.empty:
            for feat_idx in tqdm(range(num_features), desc="Analyzing feature patterns"):
                try:
                    # Calculate feature differences for high efficiency paths
                    high_eff_diff = np.abs(high_efficiency_paths[f'source_feature_{feat_idx+1}'] - 
                                         high_efficiency_paths[f'destination_feature_{feat_idx+1}'])
                    
                    # Calculate feature differences for low efficiency paths
                    low_eff_diff = np.abs(low_efficiency_paths[f'source_feature_{feat_idx+1}'] - 
                                        low_efficiency_paths[f'destination_feature_{feat_idx+1}'])
                    
                    # Store statistics for high efficiency paths
                    if not high_eff_diff.empty and not high_eff_diff.isna().all():
                        feature_patterns['high_efficiency'][f'feature_{feat_idx+1}'] = {
                            'mean_diff': float(np.mean(high_eff_diff)),
                            'std_diff': float(np.std(high_eff_diff)),
                            'median_diff': float(np.median(high_eff_diff)),
                            'min_diff': float(np.min(high_eff_diff)),
                            'max_diff': float(np.max(high_eff_diff))
                        }
                    
                    # Store statistics for low efficiency paths
                    if not low_eff_diff.empty and not low_eff_diff.isna().all():
                        feature_patterns['low_efficiency'][f'feature_{feat_idx+1}'] = {
                            'mean_diff': float(np.mean(low_eff_diff)),
                            'std_diff': float(np.std(low_eff_diff)),
                            'median_diff': float(np.median(low_eff_diff)),
                            'min_diff': float(np.min(low_eff_diff)),
                            'max_diff': float(np.max(low_eff_diff))
                        }
                except (KeyError, ValueError) as e:
                    print(f"Warning: Could not analyze feature {feat_idx+1}: {e}")
                    continue
    except (IndexError, ValueError) as e:
        print(f"Warning: Could not calculate efficiency thresholds: {e}")
else:
    print("Warning: No valid path efficiency data available for pattern analysis")

# Add pattern analysis summary with safety checks
feature_patterns['analysis_summary'] = {
    'high_efficiency_threshold': float(efficiency_threshold_high) if 'efficiency_threshold_high' in locals() else None,
    'low_efficiency_threshold': float(efficiency_threshold_low) if 'efficiency_threshold_low' in locals() else None,
    'num_high_efficiency_paths': len(high_efficiency_paths) if 'high_efficiency_paths' in locals() else 0,
    'num_low_efficiency_paths': len(low_efficiency_paths) if 'low_efficiency_paths' in locals() else 0
}

# Define recommendations based on feature analysis
recommendations = {
    'primary_features': {},
    'secondary_features': {},
    'weighting_strategy': {}
}

# Add primary features with new weighting formula
for feat_idx, importance in sorted_features[:5]:
    if feat_idx-1 in feature_stats:
        stats = feature_stats[feat_idx-1]
        weight = (
            0.4 * importance +
            0.3 * abs(stats['mean_slope']) +
            0.3 * stats['mean_r2']
        )
        recommendations['primary_features'][str(feat_idx)] = {
            'importance_score': float(importance),
            'gradient_strength': float(stats['mean_slope']),
            'consistency': float(stats['mean_r2']),
            'weight': float(weight)
        }

# Add secondary features with new weighting formula
for feat_idx, importance in sorted_features[5:10]:
    if feat_idx-1 in feature_stats:
        stats = feature_stats[feat_idx-1]
        weight = 0.5 * (
            0.4 * importance +
            0.3 * abs(stats['mean_slope']) +
            0.3 * stats['mean_r2']
        )
        recommendations['secondary_features'][str(feat_idx)] = {
            'importance_score': float(importance),
            'gradient_strength': float(stats['mean_slope']),
            'consistency': float(stats['mean_r2']),
            'weight': float(weight)
        }

# Update weighting strategy explanation
recommendations['weighting_strategy'] = {
    'explanation': """
    The weighting strategy uses an additive combination of three components:
    1. Overall Importance Score (40%): Combines consistency, efficiency correlation, entropy, and stability
    2. Gradient Strength (30%): How strongly the feature changes along paths
    3. Consistency (R²) (30%): How consistently the feature behaves across different paths
    
    Primary weights = 0.4 * importance + 0.3 * |gradient| + 0.3 * R²
    Secondary weights = 0.5 * (0.4 * importance + 0.3 * |gradient| + 0.3 * R²)
    """,
    'components': {
        'importance_weight': 0.4,
        'gradient_weight': 0.3,
        'consistency_weight': 0.3,
        'secondary_scale': 0.5
    }
}

# Now define the results dictionary
results = {
    'design_stats': design_stats,
    'efficiency_stats': efficiency_stats,
    'feature_patterns': feature_patterns,
    'recommendations': recommendations,
    'feature_importance': {f'feature_{i+1}': float(imp) for i, imp in enumerate(feature_importance)},
    'key_features': [int(f) for f in key_features]
}

# Add additional pattern analysis metrics (after feature patterns calculation)
print("\nCalculating additional pattern metrics...")
pattern_metrics = {
    'position_based': {},
    'design_based': {},
    'efficiency_based': {}
}

# 1. Position-based analysis
position_bins = np.linspace(0, 1, 5)  # Split path into quarters
for feat_idx in range(num_features):
    if not np.all(np.isnan(feature_values[feat_idx])):
        pattern_metrics['position_based'][f'feature_{feat_idx+1}'] = {
            'quartile_means': [],
            'quartile_trends': []
        }
        
        for i in range(len(position_bins)-1):
            mask = (all_path_positions >= position_bins[i]) & (all_path_positions < position_bins[i+1])
            quartile_values = feature_values[feat_idx][mask]
            pattern_metrics['position_based'][f'feature_{feat_idx+1}']['quartile_means'].append(
                float(np.nanmean(quartile_values)))
            
            if i > 0:
                trend = pattern_metrics['position_based'][f'feature_{feat_idx+1}']['quartile_means'][-1] - \
                       pattern_metrics['position_based'][f'feature_{feat_idx+1}']['quartile_means'][-2]
                pattern_metrics['position_based'][f'feature_{feat_idx+1}']['quartile_trends'].append(float(trend))

# 2. Design-based analysis
for design in unique_designs:
    pattern_metrics['design_based'][str(design)] = {}
    design_mask = design_values == design
    
    for feat_idx in range(num_features):
        if not np.all(np.isnan(feature_values[feat_idx])):
            design_feat_values = feature_values[feat_idx][design_mask]
            pattern_metrics['design_based'][str(design)][f'feature_{feat_idx+1}'] = {
                'mean': float(np.nanmean(design_feat_values)),
                'std': float(np.nanstd(design_feat_values)),
                'median': float(np.nanmedian(design_feat_values)),
                'iqr': float(np.nanpercentile(design_feat_values, 75) - 
                           np.nanpercentile(design_feat_values, 25))
            }

# 3. Efficiency-based analysis
efficiency_bins = np.linspace(df['path_efficiency'].min(), df['path_efficiency'].max(), 5)
for feat_idx in range(num_features):
    pattern_metrics['efficiency_based'][f'feature_{feat_idx+1}'] = {
        'efficiency_correlations': [],
        'efficiency_bin_means': []
    }
    
    for i in range(len(efficiency_bins)-1):
        mask = (path_efficiencies >= efficiency_bins[i]) & (path_efficiencies < efficiency_bins[i+1])
        bin_values = feature_values[feat_idx][mask]
        pattern_metrics['efficiency_based'][f'feature_{feat_idx+1}']['efficiency_bin_means'].append(
            float(np.nanmean(bin_values)))

# Add pattern metrics to results
results['pattern_metrics'] = pattern_metrics

# Add after the feature importance calculation section:
print("\nCalculating feature entropy patterns...")
entropy_patterns = {}
for feat_idx in tqdm(range(num_features), desc="Calculating entropy patterns"):
    if not np.all(np.isnan(feature_values[feat_idx])):
        entropy_patterns[f'feature_{feat_idx+1}'] = analyze_feature_patterns(
            feature_values[feat_idx],
            all_path_positions,
            design_values,
            path_efficiencies
        )

# Add entropy-based visualizations
print("\nGenerating entropy visualizations...")
os.makedirs('path_analysis_plots/entropy', exist_ok=True)

# 1. Feature entropy heatmap
plt.figure(figsize=(15, 10))
entropy_matrix = np.zeros((num_features, 10))
for feat_idx in range(num_features):
    if f'feature_{feat_idx+1}' in entropy_patterns:
        entropy_matrix[feat_idx] = entropy_patterns[f'feature_{feat_idx+1}']['position_entropy']

sns.heatmap(entropy_matrix, cmap='viridis')
plt.title('Feature Entropy Along Paths')
plt.xlabel('Position in Path (Deciles)')
plt.ylabel('Feature Index')
plt.savefig('path_analysis_plots/entropy/feature_entropy_heatmap.png')
plt.close()

# 2. Feature importance components visualization
print("\nGenerating feature importance component visualizations...")

# Define components first with safety checks
components = {}
if entropy_patterns:
    components = {
        'Consistency (R²)': [patterns['mean_r2'] if patterns and 'mean_r2' in patterns else 0.0 
                            for patterns in entropy_patterns.values()],
        'Efficiency Correlation': [abs(patterns['efficiency_correlation']) if patterns and 'efficiency_correlation' in patterns else 0.0 
                                 for patterns in entropy_patterns.values()],
        'Entropy Score': [1.0 - patterns['entropy'] / (np.log(10) + 1e-10) if patterns and 'entropy' in patterns else 0.0 
                         for patterns in entropy_patterns.values()],
        'Stability': [patterns['stability'] if patterns and 'stability' in patterns else 0.0 
                     for patterns in entropy_patterns.values()]
    }

    # Create stacked bar plot
    plt.figure(figsize=(12, 6))
    df_components = pd.DataFrame(components)
    if not df_components.empty and len(df_components.columns) > 0:
        df_components.plot(kind='bar', stacked=True)
        plt.title('Combined Feature Importance Components')
        plt.xlabel('Feature Index')
        plt.ylabel('Component Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('path_analysis_plots/entropy/combined_importance_components.png')
    plt.close()

    # Plot each component separately
    for name, values in components.items():
        if values and len(values) > 0:  # Check if we have valid values
            plt.figure(figsize=(10, 5))
            plt.bar(range(len(values)), values, color=plt.cm.viridis(0.5))
            plt.title(f'{name} Distribution')
            plt.xlabel('Feature Index')
            plt.ylabel('Score')
            plt.xticks(range(len(values)), [str(i+1) for i in range(len(values))], rotation=45)
            plt.tight_layout()
            # Create a filename-friendly version of the component name
            filename = name.lower().replace(' ', '_').replace('²', '2').replace('(', '').replace(')', '')
            plt.savefig(f'path_analysis_plots/entropy/{filename}_distribution.png')
            plt.close()

    # Create correlation heatmap of components
    plt.figure(figsize=(8, 6))
    component_df = pd.DataFrame(components)
    if not component_df.empty and len(component_df.columns) > 0:
        sns.heatmap(component_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Component Correlation Matrix')
        plt.tight_layout()
        plt.savefig('path_analysis_plots/entropy/component_correlation_matrix.png')
    plt.close()
else:
    print("Warning: No entropy patterns available for visualization")

# Add entropy patterns to results
results['entropy_patterns'] = entropy_patterns

# Add explanatory print statements about feature weights
print("\nFeature Weight Explanation:")
print("Features are weighted based on:")
print("1. Consistency (30%): How predictably the feature changes along paths")
print("2. Efficiency Correlation (30%): How well feature changes correlate with path efficiency")
print("3. Gradient Strength (20%): How strongly the feature changes along paths")
print("4. Stability (20%): How stable the feature values are")
print("\nHigher weights indicate features that are more reliable for guiding efficient paths.")
print("When using these weights in random walks, prefer moves that follow patterns of high-weighted features.")

# Save feature weights for use in weighted random walk
feature_weights = {}
for feat_idx in range(num_features):
    if f'feature_{feat_idx+1}' in entropy_patterns:
        feature_weights[feat_idx] = entropy_patterns[f'feature_{feat_idx+1}']

with open('feature_weights.pkl', 'wb') as f:
    pickle.dump(feature_weights, f)

# Convert numpy types to Python types for JSON serialization
def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    else:
        return serialize_numpy(obj)

serializable_results = convert_to_serializable(results)

with open('path_analysis_results.json', 'w') as f:
    json.dump(serializable_results, f, indent=4)

print("\nVisualization plots have been saved in the 'path_analysis_plots' directory.")
print("\nDetailed analysis complete. Results saved to path_analysis_results.json")