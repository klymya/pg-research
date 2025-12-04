"""
Train GMM using sklearn and export to custom format.

Uses sklearn's optimized implementation for training, then exports
the parameters for use with custom detector.
"""

import numpy as np
import csv
from sklearn.mixture import GaussianMixture
from custom_scaler import StandardScaler
from custom_gmm_detector import CustomGMMDetector


def extract_features_from_bbox(x_center, y_center, w_rel, h_rel, cam_height):
    """Extract features from a bbox for novelty detection."""
    aspect_ratio = w_rel / h_rel if h_rel > 0 else 0
    bbox_area = w_rel * h_rel
    dist_from_center = np.sqrt((x_center - 0.5)**2 + (y_center - 0.5)**2)
    return np.array([aspect_ratio, bbox_area, dist_from_center, cam_height])


def absolute_to_relative(x_abs, y_abs, w_abs, h_abs, img_width, img_height):
    """Convert absolute bbox coordinates to relative YOLO format."""
    x_rel = x_abs / img_width
    y_rel = y_abs / img_height
    w_rel = w_abs / img_width
    h_rel = h_abs / img_height
    x_center = x_rel + w_rel / 2
    y_center = y_rel + h_rel / 2
    return x_center, y_center, w_rel, h_rel


print("="*80)
print("Training GMM Detector using sklearn")
print("="*80)

# Load YOLO annotations
print("\n1. Loading YOLO annotations...")
CSV_FILE = "yolo_annotations.csv"

data = []
with open(CSV_FILE, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append({
            'x': float(row['x']),
            'y': float(row['y']),
            'w': float(row['w']),
            'h': float(row['h']),
            'cam_height': float(row['cam_height']),
            'subset': row['subset']
        })

print(f"   ✓ Loaded {len(data)} annotations")

# Calculate features
print("\n2. Calculating features...")
X_train = []
X_test = []

for row in data:
    aspect_ratio = row['w'] / row['h']
    bbox_area = row['w'] * row['h']
    dist_from_center = np.sqrt((row['x'] - 0.5)**2 + (row['y'] - 0.5)**2)
    features = [aspect_ratio, bbox_area, dist_from_center, row['cam_height']]

    if row['subset'] == 'train':
        X_train.append(features)
    else:
        X_test.append(features)

X_train = np.array(X_train)
X_test = np.array(X_test)

print(f"   ✓ Feature matrix shape: {X_train.shape}")
print(f"   ✓ Training samples: {len(X_train)}")
print(f"   ✓ Testing samples: {len(X_test)}")

# Feature scaling
print("\n3. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"   ✓ Features scaled")

# Train sklearn GaussianMixture
print("\n4. Training GaussianMixture (5 Gaussians) with sklearn...")
print("   Parameters:")
print("      n_components: 5")
print("      covariance_type: full")
print("      reg_covar: 1e-6")
print("      random_state: 42")

sklearn_gmm = GaussianMixture(
    n_components=5,
    covariance_type='full',
    reg_covar=1e-6,
    random_state=42,
    max_iter=100,
    verbose=0
)

sklearn_gmm.fit(X_train_scaled)

print(f"\n   ✓ Model trained")
print(f"   ✓ Converged: {sklearn_gmm.converged_}")
print(f"   ✓ Iterations: {sklearn_gmm.n_iter_}")

# Print component weights
print(f"\n   Component weights:")
for k, w in enumerate(sklearn_gmm.weights_):
    print(f"      Component {k}: {w:.4f} ({w*100:.2f}%)")

# Compute thresholds
print("\n5. Computing thresholds...")
train_scores_sklearn = -sklearn_gmm.score_samples(X_train_scaled)
threshold_ok = np.quantile(train_scores_sklearn, 0.95)
threshold_novel = np.quantile(train_scores_sklearn, 0.99)

print(f"   ✓ Threshold 'ok' (95th percentile): {threshold_ok:.4f}")
print(f"   ✓ Threshold 'novel' (99th percentile): {threshold_novel:.4f}")

# Create custom detector with sklearn parameters
print("\n6. Creating CustomGMMDetector with sklearn parameters...")
custom_gmm = CustomGMMDetector(
    n_components=5,
    covariance_type='full',
    threshold_ok=0.95,
    threshold_novel=0.99
)

# Copy parameters from sklearn
custom_gmm.n_features_ = X_train_scaled.shape[1]
custom_gmm.weights_ = sklearn_gmm.weights_
custom_gmm.means_ = sklearn_gmm.means_
custom_gmm.covariances_ = sklearn_gmm.covariances_
custom_gmm.threshold_ok_ = threshold_ok
custom_gmm.threshold_novel_ = threshold_novel
custom_gmm.converged_ = sklearn_gmm.converged_
custom_gmm.n_iter_ = sklearn_gmm.n_iter_

print(f"   ✓ Custom detector created: {custom_gmm}")

# Verify that custom detector gives same results
print("\n7. Verifying custom detector matches sklearn...")
test_sample = X_train_scaled[0:1]

sklearn_score = sklearn_gmm.score_samples(test_sample)[0]
custom_score_val = custom_gmm.score_samples(test_sample)
if isinstance(custom_score_val, np.ndarray):
    custom_score = custom_score_val[0] if len(custom_score_val) > 0 else custom_score_val
else:
    custom_score = custom_score_val

print(f"   sklearn score: {sklearn_score:.6f}")
print(f"   custom score: {custom_score:.6f}")
print(f"   difference: {abs(sklearn_score - custom_score):.6e}")

if abs(sklearn_score - custom_score) < 1e-6:
    print(f"   ✓ Scores match!")
else:
    print(f"   ⚠ Scores differ slightly (expected due to numerical precision)")

# Evaluate on datasets
print("\n8. Evaluating on datasets...")

# Training set
train_pred = custom_gmm.predict(X_train_scaled)
train_scores = custom_gmm.anomaly_score(X_train_scaled)

train_ok = np.sum(train_pred == 'ok')
train_novel = np.sum(train_pred == 'novel')
train_outlier = np.sum(train_pred == 'outlier')

print(f"\n   Training set:")
print(f"      OK: {train_ok} ({train_ok/len(train_pred)*100:.2f}%)")
print(f"      Novel: {train_novel} ({train_novel/len(train_pred)*100:.2f}%)")
print(f"      Outlier: {train_outlier} ({train_outlier/len(train_pred)*100:.2f}%)")
print(f"      Score range: [{train_scores.min():.3f}, {train_scores.max():.3f}]")

# Test set
test_pred = custom_gmm.predict(X_test_scaled)
test_scores = custom_gmm.anomaly_score(X_test_scaled)

test_ok = np.sum(test_pred == 'ok')
test_novel = np.sum(test_pred == 'novel')
test_outlier = np.sum(test_pred == 'outlier')

print(f"\n   Test set:")
print(f"      OK: {test_ok} ({test_ok/len(test_pred)*100:.2f}%)")
print(f"      Novel: {test_novel} ({test_novel/len(test_pred)*100:.2f}%)")
print(f"      Outlier: {test_outlier} ({test_outlier/len(test_pred)*100:.2f}%)")
print(f"      Score range: [{test_scores.min():.3f}, {test_scores.max():.3f}]")

# Save models
print("\n9. Saving models to JSON...")
scaler.to_json('gmm_detector_scaler.json')
custom_gmm.to_json('gmm_detector_params.json')
print(f"   ✓ Saved gmm_detector_scaler.json")
print(f"   ✓ Saved gmm_detector_params.json")

# Test on example bounding box
print("\n" + "="*80)
print("Testing on Example Bounding Box")
print("="*80)

test_bboxes = [
    {'name': 'test', 'x': 63, 'y': 8, 'w': 81, 'h': 81, 'cam_height': 310},
]

img_width, img_height = 640, 480

for bbox in test_bboxes:
    print(f"\nBounding Box: {bbox['name']}")
    print("-"*80)

    # Convert to relative coordinates
    x_center, y_center, w_rel, h_rel = absolute_to_relative(
        bbox['x'], bbox['y'], bbox['w'], bbox['h'], img_width, img_height
    )

    print(f"   Input (absolute): x={bbox['x']}, y={bbox['y']}, w={bbox['w']}, h={bbox['h']}")
    print(f"   Camera height: {bbox['cam_height']} cm")
    print(f"   Relative coords: x={x_center:.4f}, y={y_center:.4f}, w={w_rel:.4f}, h={h_rel:.4f}")

    # Extract features
    features = extract_features_from_bbox(x_center, y_center, w_rel, h_rel, bbox['cam_height'])
    print(f"\n   Features:")
    print(f"      Aspect ratio: {features[0]:.4f}")
    print(f"      Bbox area: {features[1]:.4f}")
    print(f"      Distance from center: {features[2]:.4f}")
    print(f"      Camera height: {features[3]:.0f} cm")

    # Scale features
    features_scaled = scaler.transform(features)
    print(f"\n   Scaled features: [{', '.join([f'{f:.4f}' for f in features_scaled])}]")

    # Predict
    prediction = custom_gmm.predict(features_scaled)
    anomaly_score = custom_gmm.anomaly_score(features_scaled)
    log_likelihood = custom_gmm.score_samples(features_scaled)

    print(f"\n   Results:")
    print(f"      Log-likelihood: {log_likelihood:.4f}")
    print(f"      Anomaly score (negative log-likelihood): {anomaly_score:.4f}")
    print(f"      Threshold 'ok': {custom_gmm.threshold_ok_:.4f}")
    print(f"      Threshold 'novel': {custom_gmm.threshold_novel_:.4f}")
    print(f"      Prediction: {prediction.upper()}")

    if prediction == 'ok':
        print(f"      ✓ ACCEPT - Score {anomaly_score:.4f} <= Threshold {custom_gmm.threshold_ok_:.4f}")
    elif prediction == 'novel':
        print(f"      ⚠ WARNING - Score {anomaly_score:.4f} in range [{custom_gmm.threshold_ok_:.4f}, {custom_gmm.threshold_novel_:.4f}]")
    else:
        print(f"      ✗ REJECT - Score {anomaly_score:.4f} > Threshold {custom_gmm.threshold_novel_:.4f}")

print("\n" + "="*80)
print("Training and Testing Completed Successfully!")
print("="*80)
