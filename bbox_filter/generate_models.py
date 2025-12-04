"""
Generate scaler and IsolationForest models and export to JSON.

This script trains sklearn models and exports them to JSON files
so they can be loaded by the custom implementations.
"""

import numpy as np
import json
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.ensemble import IsolationForest as SklearnIsolationForest

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample training data
# Features: [aspect_ratio, bbox_area, dist_from_center, cam_height]
X_train = np.random.randn(100, 4) * np.array([0.3, 0.02, 0.1, 30]) + np.array([0.85, 0.04, 0.35, 265])

print("Training StandardScaler...")
scaler = SklearnStandardScaler()
scaler.fit(X_train)

# Export scaler to JSON
scaler_params = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "var": scaler.var_.tolist(),
    "n_features": len(scaler.mean_),
}

with open('scaler_params.json', 'w') as f:
    json.dump(scaler_params, f, indent=2)

print(f"✓ Saved scaler parameters to scaler_params.json")
print(f"  Features: {scaler_params['n_features']}")
print(f"  Mean: {scaler_params['mean']}")

# Train IsolationForest on scaled data
print("\nTraining IsolationForest...")
X_train_scaled = scaler.transform(X_train)
iforest = SklearnIsolationForest(n_estimators=30, random_state=42)
iforest.fit(X_train_scaled)

# Export IsolationForest to JSON
def export_tree(tree):
    """Export a single tree to dictionary."""
    t = tree.tree_
    return {
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "left": t.children_left.tolist(),
        "right": t.children_right.tolist(),
        "n_node_samples": t.n_node_samples.tolist(),
        "node_count": t.node_count,
        "max_depth": t.max_depth,
    }

iforest_params = {
    "n_estimators": len(iforest.estimators_),
    "max_samples": iforest.max_samples_,  # Note: use max_samples_ (fitted value, always int)
    "contamination": iforest.contamination,  # Can be 'auto' or float
    "trees": [export_tree(estimator) for estimator in iforest.estimators_],
}

with open('iforest_params.json', 'w') as f:
    json.dump(iforest_params, f, indent=2)

print(f"✓ Saved IsolationForest parameters to iforest_params.json")
print(f"  Number of trees: {iforest_params['n_estimators']}")
print(f"  Max samples: {iforest_params['max_samples']}")
print(f"  Contamination: {iforest_params['contamination']}")

print("\n" + "="*80)
print("Models successfully exported to JSON files!")
print("="*80)
print("\nYou can now use:")
print("  - scaler_params.json")
print("  - iforest_params.json")
print("\nTo load the custom implementations:")
print("  from custom_scaler import StandardScaler")
print("  from custom_isolation_forest import IsolationForest")
print("  scaler = StandardScaler.from_json('scaler_params.json')")
print("  forest = IsolationForest.from_json('iforest_params.json')")
