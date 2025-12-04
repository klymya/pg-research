"""
Test all novelty detection methods on example bounding box.

Compares:
1. IsolationForest (custom implementation)
2. CustomNoveltyDetector (Mahalanobis distance)
3. CustomGMMDetector (Gaussian Mixture Model with 5 components)
"""

import numpy as np
from custom_scaler import StandardScaler
from custom_isolation_forest import IsolationForest
from custom_novelty_detector import CustomNoveltyDetector
from custom_gmm_detector import CustomGMMDetector


def absolute_to_relative(x_abs, y_abs, w_abs, h_abs, img_width, img_height):
    """Convert absolute bbox coordinates to relative YOLO format."""
    x_rel = x_abs / img_width
    y_rel = y_abs / img_height
    w_rel = w_abs / img_width
    h_rel = h_abs / img_height

    x_center = x_rel + w_rel / 2
    y_center = y_rel + h_rel / 2

    return x_center, y_center, w_rel, h_rel


def extract_features_from_bbox(x_center, y_center, w_rel, h_rel, cam_height):
    """Extract features from a bbox for novelty detection."""
    aspect_ratio = w_rel / h_rel if h_rel > 0 else 0
    bbox_area = w_rel * h_rel
    dist_from_center = np.sqrt((x_center - 0.5)**2 + (y_center - 0.5)**2)

    return np.array([aspect_ratio, bbox_area, dist_from_center, cam_height])


print("="*80)
print("Testing All Novelty Detection Methods")
print("="*80)

# Load models
print("\n1. Loading models...")
print("-"*80)

# IsolationForest
iforest_scaler = StandardScaler.from_json('scaler_params.json')
iforest_model = IsolationForest.from_json('iforest_params.json')
print(f"   ✓ Loaded IsolationForest: {iforest_model}")

# CustomNoveltyDetector
custom_scaler = StandardScaler.from_json('custom_detector_scaler.json')
custom_model = CustomNoveltyDetector.from_json('custom_detector_params.json')
print(f"   ✓ Loaded CustomNoveltyDetector: {custom_model}")

# CustomGMMDetector
gmm_scaler = StandardScaler.from_json('gmm_detector_scaler.json')
gmm_model = CustomGMMDetector.from_json('gmm_detector_params.json')
print(f"   ✓ Loaded CustomGMMDetector: {gmm_model}")

# Test bounding box
print("\n2. Test Bounding Box")
print("-"*80)

test_bbox = {'name': 'test', 'x': 63, 'y': 8, 'w': 81, 'h': 81, 'cam_height': 310}
img_width, img_height = 640, 480

print(f"   Input (absolute): x={test_bbox['x']}, y={test_bbox['y']}, "
      f"w={test_bbox['w']}, h={test_bbox['h']}")
print(f"   Camera height: {test_bbox['cam_height']} cm")
print(f"   Image size: {img_width}x{img_height}")

# Convert to relative coordinates
x_center, y_center, w_rel, h_rel = absolute_to_relative(
    test_bbox['x'], test_bbox['y'], test_bbox['w'], test_bbox['h'],
    img_width, img_height
)

print(f"\n   Relative coords:")
print(f"      Center: ({x_center:.4f}, {y_center:.4f})")
print(f"      Size: w={w_rel:.4f}, h={h_rel:.4f}")

# Extract features
features = extract_features_from_bbox(x_center, y_center, w_rel, h_rel, test_bbox['cam_height'])

print(f"\n   Features:")
print(f"      Aspect ratio: {features[0]:.4f}")
print(f"      Bbox area: {features[1]:.4f}")
print(f"      Distance from center: {features[2]:.4f}")
print(f"      Camera height: {features[3]:.0f} cm")

# Test IsolationForest
print("\n3. IsolationForest Results")
print("-"*80)

iforest_scaled = iforest_scaler.transform(features)
print(f"   Scaled features: [{', '.join([f'{f:.4f}' for f in iforest_scaled])}]")

iforest_score = iforest_model.score_samples(iforest_scaled)
iforest_decision = iforest_model.decision_function(iforest_scaled)
iforest_pred = iforest_model.predict(iforest_scaled)

print(f"\n   Score: {iforest_score:.4f}")
print(f"   Decision function: {iforest_decision:.4f}")
print(f"   Offset: {iforest_model.offset_:.4f}")
print(f"   Prediction: {'NORMAL (1)' if iforest_pred == 1 else 'OUTLIER (-1)'}")

if iforest_pred == 1:
    print(f"   ✓ ACCEPT - Decision {iforest_decision:.4f} >= 0")
else:
    print(f"   ✗ REJECT - Decision {iforest_decision:.4f} < 0")

# Test CustomNoveltyDetector
print("\n4. CustomNoveltyDetector Results (Mahalanobis)")
print("-"*80)

custom_scaled = custom_scaler.transform(features)
print(f"   Scaled features: [{', '.join([f'{f:.4f}' for f in custom_scaled])}]")

custom_score = custom_model.score_samples(custom_scaled)
custom_pred = custom_model.predict(custom_scaled)
mahalanobis_dist = -custom_score

print(f"\n   Mahalanobis distance: {mahalanobis_dist:.4f}")
print(f"   Threshold: {custom_model.threshold_:.4f}")
print(f"   Score: {custom_score:.4f}")
print(f"   Prediction: {'NORMAL (1)' if custom_pred == 1 else 'OUTLIER (-1)'}")

if custom_pred == 1:
    print(f"   ✓ ACCEPT - Distance {mahalanobis_dist:.4f} <= Threshold {custom_model.threshold_:.4f}")
else:
    print(f"   ✗ REJECT - Distance {mahalanobis_dist:.4f} > Threshold {custom_model.threshold_:.4f}")

# Test CustomGMMDetector
print("\n5. CustomGMMDetector Results (GMM with 5 Gaussians)")
print("-"*80)

gmm_scaled = gmm_scaler.transform(features)
print(f"   Scaled features: [{', '.join([f'{f:.4f}' for f in gmm_scaled])}]")

gmm_anomaly = gmm_model.anomaly_score(gmm_scaled)
gmm_log_likelihood = gmm_model.score_samples(gmm_scaled)
gmm_pred_str = gmm_model.predict(gmm_scaled)
gmm_pred_num = gmm_model.predict_numeric(gmm_scaled)

print(f"\n   Log-likelihood: {gmm_log_likelihood:.4f}")
print(f"   Anomaly score (negative log-likelihood): {gmm_anomaly:.4f}")
print(f"   Threshold 'ok': {gmm_model.threshold_ok_:.4f}")
print(f"   Threshold 'novel': {gmm_model.threshold_novel_:.4f}")
print(f"   Prediction: {gmm_pred_str.upper()} (numeric: {gmm_pred_num})")

if gmm_pred_str == 'ok':
    print(f"   ✓ ACCEPT - Score {gmm_anomaly:.4f} <= Threshold {gmm_model.threshold_ok_:.4f}")
elif gmm_pred_str == 'novel':
    print(f"   ⚠ WARNING - Score {gmm_anomaly:.4f} in range [{gmm_model.threshold_ok_:.4f}, {gmm_model.threshold_novel_:.4f}]")
else:
    print(f"   ✗ REJECT - Score {gmm_anomaly:.4f} > Threshold {gmm_model.threshold_novel_:.4f}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

methods = [
    ('IsolationForest', iforest_pred, iforest_score),
    ('CustomNoveltyDetector', custom_pred, custom_score),
    ('CustomGMMDetector', gmm_pred_num, gmm_anomaly),
]

print(f"\n{'Method':<25} {'Prediction':<15} {'Score':<15} {'Status'}")
print("-"*80)

for method_name, pred, score in methods:
    if method_name == 'CustomGMMDetector':
        # GMM has 3 states: ok (1), novel (0), outlier (-1)
        if pred == 1:
            status = 'OK'
            symbol = '✓'
        elif pred == 0:
            status = 'NOVEL'
            symbol = '⚠'
        else:
            status = 'OUTLIER'
            symbol = '✗'
    else:
        status = 'NORMAL' if pred == 1 else 'OUTLIER'
        symbol = '✓' if pred == 1 else '✗'
    print(f"{method_name:<25} {status:<15} {score:<15.4f} {symbol}")

print("-"*80)

# Consensus
accept_count = 0
reject_count = 0
warning_count = 0

for method_name, pred, _ in methods:
    if method_name == 'CustomGMMDetector':
        if pred == 1:
            accept_count += 1
        elif pred == 0:
            warning_count += 1
        else:
            reject_count += 1
    else:
        if pred == 1:
            accept_count += 1
        else:
            reject_count += 1

print(f"\nConsensus:")
print(f"   Accept: {accept_count}/{len(methods)} methods")
print(f"   Warning: {warning_count}/{len(methods)} methods")
print(f"   Reject: {reject_count}/{len(methods)} methods")

if accept_count == len(methods):
    print("   → ACCEPT (all methods agree)")
elif reject_count == len(methods):
    print("   → REJECT (all methods agree)")
elif accept_count >= len(methods) / 2:
    print("   → ACCEPT (majority vote)")
else:
    print("   → WARNING (no clear consensus - manual review recommended)")

print("\n" + "="*80)
print("Testing Completed!")
print("="*80)
