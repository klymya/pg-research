"""
Example usage of custom IsolationForest and StandardScaler implementations.

This script demonstrates how to:
1. Load models from JSON files
2. Process single samples
3. Process batches of samples
4. Complete bbox novelty detection pipeline
"""

import numpy as np
from custom_scaler import StandardScaler
from custom_isolation_forest import IsolationForest


def main():
    print("="*80)
    print("Custom IsolationForest and StandardScaler - Example Usage")
    print("="*80)

    # Load models from JSON
    print("\n1. Loading models from JSON files...")
    scaler = StandardScaler.from_json('scaler_params.json')
    forest = IsolationForest.from_json('iforest_params.json')
    print(f"   ✓ Loaded {scaler}")
    print(f"   ✓ Loaded {forest}")

    # Single sample prediction
    print("\n2. Single sample prediction:")
    print("-" * 80)
    x = np.array([0.75, 0.03, 0.14, 250])
    print(f"   Input features: {x}")
    print(f"   (aspect_ratio, bbox_area, dist_from_center, cam_height)")

    x_scaled = scaler.transform(x)
    print(f"   Scaled features: {x_scaled}")

    score = forest.score_samples(x_scaled)
    prediction = forest.predict(x_scaled)

    print(f"   Anomaly score: {score:.4f}")
    print(f"   Prediction: {'NORMAL' if prediction == 1 else 'OUTLIER'}")

    # Batch processing
    print("\n3. Batch processing:")
    print("-" * 80)
    X_batch = np.array([
        [0.75, 0.03, 0.14, 250],  # Normal
        [2.5, 0.01, 0.4, 270],     # Outlier: extreme aspect ratio
        [0.9, 0.7, 0.0, 450],      # Outlier: large area + unusual height
        [0.8, 0.04, 0.35, 260],    # Normal
    ])

    X_scaled = scaler.transform(X_batch)
    scores = forest.score_samples(X_scaled)
    predictions = forest.predict(X_scaled)

    print(f"   Processing {len(X_batch)} samples:\n")
    for i, (x, score, pred) in enumerate(zip(X_batch, scores, predictions)):
        status = "NORMAL" if pred == 1 else "OUTLIER"
        print(f"   Sample {i+1}: {status:8s} | score: {score:7.4f} | features: {x}")

    # Complete bbox pipeline
    print("\n4. Complete bbox novelty detection pipeline:")
    print("-" * 80)

    def check_bbox(x, y, w, h, cam_height, img_width=640, img_height=480):
        """Check if a bounding box is anomalous."""
        # Convert to relative coordinates
        x_center = (x + w/2) / img_width
        y_center = (y + h/2) / img_height
        w_rel = w / img_width
        h_rel = h / img_height

        # Extract features
        aspect_ratio = w_rel / h_rel
        bbox_area = w_rel * h_rel
        dist_from_center = np.sqrt((x_center - 0.5)**2 + (y_center - 0.5)**2)
        features = np.array([aspect_ratio, bbox_area, dist_from_center, cam_height])

        # Scale and predict
        features_scaled = scaler.transform(features)
        prediction = forest.predict(features_scaled)
        score = forest.score_samples(features_scaled)

        return {
            'prediction': 'NORMAL' if prediction == 1 else 'OUTLIER',
            'score': float(score),
            'bbox': {'x': x, 'y': y, 'w': w, 'h': h, 'cam_height': cam_height},
            'features': {
                'aspect_ratio': aspect_ratio,
                'bbox_area': bbox_area,
                'dist_from_center': dist_from_center,
            }
        }

    # Test bboxes
    # test_bboxes = [
    #     {'name': 'Normal bbox', 'x': 200, 'y': 150, 'w': 80, 'h': 120, 'cam_height': 250},
    #     {'name': 'Large bbox', 'x': 50, 'y': 50, 'w': 500, 'h': 400, 'cam_height': 260},
    #     {'name': 'Wide bbox', 'x': 100, 'y': 200, 'w': 400, 'h': 30, 'cam_height': 250},
    #     {'name': 'Small bbox', 'x': 300, 'y': 200, 'w': 30, 'h': 45, 'cam_height': 270},
    # ]
    test_bboxes = [
        {'name': 'test', 'x': 63, 'y': 8, 'w': 81, 'h': 81, 'cam_height': 310},
    ]

    print(f"   Testing {len(test_bboxes)} bounding boxes:\n")
    for bbox_info in test_bboxes:
        name = bbox_info.pop('name')
        result = check_bbox(**bbox_info)

        print(f"   {name}:")
        print(f"      Input: x={result['bbox']['x']}, y={result['bbox']['y']}, "
              f"w={result['bbox']['w']}, h={result['bbox']['h']}, "
              f"cam_height={result['bbox']['cam_height']} cm")
        print(f"      Features: aspect_ratio={result['features']['aspect_ratio']:.3f}, "
              f"area={result['features']['bbox_area']:.4f}, "
              f"dist_from_center={result['features']['dist_from_center']:.3f}")
        print(f"      Result: {result['prediction']} (score: {result['score']:.4f})")
        print()

    # Demonstrate model properties
    print("5. Model properties:")
    print("-" * 80)
    print(f"   Scaler:")
    print(f"      Number of features: {scaler.n_features_}")
    print(f"      Feature means: {scaler.mean_}")
    print(f"      Feature scales: {scaler.scale_}")

    print(f"\n   IsolationForest:")
    print(f"      Number of trees: {forest.n_estimators_}")
    print(f"      Max samples per tree: {forest.max_samples_}")
    print(f"      Contamination: {forest.contamination_}")
    print(f"      Offset (normalization): {forest.offset_:.4f}")

    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
