"""
Train CustomNoveltyDetector on YOLO annotations and test it.

Pipeline:
1. Load YOLO annotations from CSV
2. Build feature matrices for train/test
3. Scale features with StandardScaler (custom_scaler.py)
4. Train CustomNoveltyDetector (Mahalanobis-based novelty detector)
5. Evaluate on train/test
6. Export scaler and detector to JSON
7. Run a small manual test on example bbox
"""

import os
import numpy as np
import csv
import sys
from pathlib import Path

from custom_scaler import StandardScaler
from custom_novelty_detector import CustomNoveltyDetector


# -----------------------------------------------------------------------------
# Feature extraction helpers
# -----------------------------------------------------------------------------
def extract_features_from_bbox(x_center, y_center, w_rel, h_rel, cam_height):
    """
    Extract features from a bbox for novelty detection.

    Args:
        x_center: normalized x center coordinate [0, 1]
        y_center: normalized y center coordinate [0, 1]
        w_rel: normalized width [0, 1]
        h_rel: normalized height [0, 1]
        cam_height: camera height in cm

    Returns:
        numpy array: feature vector [aspect_ratio, bbox_area, dist_from_center, cam_height]
    """
    aspect_ratio = w_rel / h_rel if h_rel > 0 else 0.0
    bbox_area = w_rel * h_rel
    dist_from_center = float(np.sqrt((x_center - 0.5) ** 2 + (y_center - 0.5) ** 2))

    return np.array([aspect_ratio, bbox_area, dist_from_center, cam_height], dtype=float)


def absolute_to_relative(x_abs, y_abs, w_abs, h_abs, img_width, img_height):
    """
    Convert absolute bbox coordinates to relative (normalized) YOLO-like format.

    Args:
        x_abs: absolute x coordinate (top-left corner)
        y_abs: absolute y coordinate (top-left corner)
        w_abs: absolute width
        h_abs: absolute height
        img_width: image width in pixels
        img_height: image height in pixels

    Returns:
        tuple: (x_center, y_center, w_rel, h_rel) in normalized coordinates [0, 1]
    """
    x_rel = x_abs / img_width
    y_rel = y_abs / img_height
    w_rel = w_abs / img_width
    h_rel = h_abs / img_height

    x_center = x_rel + w_rel / 2.0
    y_center = y_rel + h_rel / 2.0

    return x_center, y_center, w_rel, h_rel


# -----------------------------------------------------------------------------
# Main training / evaluation script
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("Training Custom Novelty Detector (Mahalanobis)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Load YOLO annotations
    # -------------------------------------------------------------------------
    print("\n1. Loading YOLO annotations...")
    CSV_FILE = "yolo_annotations.csv"
    csv_path = Path(CSV_FILE)

    if not csv_path.exists():
        print(f"   ✗ CSV file not found: {csv_path}")
        sys.exit(1)

    data = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data.append({
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "w": float(row["w"]),
                    "h": float(row["h"]),
                    "cam_height": float(row["cam_height"]),
                    "subset": row.get("subset", "train"),
                })
            except (KeyError, ValueError) as e:
                print(f"   ! Skipping bad row: {row} ({e})")

    if not data:
        print("   ✗ No valid rows loaded from CSV.")
        sys.exit(1)

    print(f"   ✓ Loaded {len(data)} annotations")

    # -------------------------------------------------------------------------
    # 2. Build feature matrices
    # -------------------------------------------------------------------------
    print("\n2. Calculating features...")
    X_train = []
    X_test = []

    for row in data:
        # Here x,y,w,h assumed already in [0,1] (YOLO-style)
        aspect_ratio = row["w"] / row["h"] if row["h"] > 0 else 0.0
        bbox_area = row["w"] * row["h"]
        dist_from_center = float(np.sqrt((row["x"] - 0.5) ** 2 + (row["y"] - 0.5) ** 2))
        features = [aspect_ratio, bbox_area, dist_from_center, row["cam_height"]]

        if row["subset"] == "train":
            X_train.append(features)
        else:
            X_test.append(features)

    if not X_train:
        print("   ✗ No training samples found (subset != 'train').")
        sys.exit(1)

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float) if X_test else None

    print(f"   ✓ Feature matrix shape (train): {X_train.shape}")
    if X_test is not None:
        print(f"   ✓ Feature matrix shape (test):  {X_test.shape}")
    else:
        print("   ! No test samples found (all rows marked 'train').")

    if not np.isfinite(X_train).all():
        print("   ✗ Training features contain NaN or Inf. Check input CSV / feature logic.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 3. Scaling features with StandardScaler (custom_scaler.py)
    # -------------------------------------------------------------------------
    print("\n3. Scaling features with StandardScaler...")
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None and len(X_test) > 0:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = None

    if not np.isfinite(X_train_scaled).all():
        print("   ✗ Scaled training features contain NaN or Inf. Check scaler/data.")
        sys.exit(1)

    print("   ✓ Features scaled (mean≈0, std≈1)")

    # -------------------------------------------------------------------------
    # 4. Train CustomNoveltyDetector (Mahalanobis)
    # -------------------------------------------------------------------------
    print("\n4. Training CustomNoveltyDetector (Mahalanobis)...")
    contamination = 0.15
    print(f"   Parameters:")
    print(f"      contamination: {contamination:.2f}")

    detector = CustomNoveltyDetector(contamination=contamination)
    detector.fit(X_train_scaled)

    print(f"   ✓ Model trained: {detector}")
    print(f"   ✓ Threshold (distance): {detector.threshold_:.4f}")

    # -------------------------------------------------------------------------
    # 5. Evaluate on training & test
    # -------------------------------------------------------------------------
    print("\n5. Evaluating on datasets...")

    # Training set
    train_pred = detector.predict(X_train_scaled)
    train_scores = detector.score_samples(X_train_scaled)  # negative distances
    train_distances = -train_scores

    n_train_in = (train_pred == 1).sum()
    n_train_out = (train_pred == -1).sum()

    print(f"\n   Training set:")
    print(f"      Samples:  {len(train_pred)}")
    print(f"      Inliers:  {n_train_in} ({n_train_in / len(train_pred) * 100:.2f}%)")
    print(f"      Outliers: {n_train_out} ({n_train_out / len(train_pred) * 100:.2f}%)")
    print(f"      Distance range: [{train_distances.min():.3f}, {train_distances.max():.3f}]")
    print(f"      Score range (neg. dist): [{train_scores.min():.3f}, {train_scores.max():.3f}]")

    # Test set
    if X_test_scaled is not None and len(X_test_scaled) > 0:
        test_pred = detector.predict(X_test_scaled)
        test_scores = detector.score_samples(X_test_scaled)
        test_distances = -test_scores

        n_test_in = (test_pred == 1).sum()
        n_test_out = (test_pred == -1).sum()

        print(f"\n   Test set:")
        print(f"      Samples:  {len(test_pred)}")
        print(f"      Inliers:  {n_test_in} ({n_test_in / len(test_pred) * 100:.2f}%)")
        print(f"      Outliers: {n_test_out} ({n_test_out / len(test_pred) * 100:.2f}%)")
        print(f"      Distance range: [{test_distances.min():.3f}, {test_distances.max():.3f}]")
        print(f"      Score range (neg. dist): [{test_scores.min():.3f}, {test_scores.max():.3f}]")
    else:
        print("\n   Test set: (none)")

    # -------------------------------------------------------------------------
    # 6. Save models
    # -------------------------------------------------------------------------
    print("\n6. Saving models to JSON...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "novelty_detector_model")
    os.makedirs(model_dir, exist_ok=True)
    scaler.to_json(os.path.join(model_dir, "custom_detector_scaler.json"))
    detector.to_json(os.path.join(model_dir, "custom_detector_params.json"))
    print("   ✓ Saved custom_detector_scaler.json")
    print("   ✓ Saved custom_detector_params.json")

    # -------------------------------------------------------------------------
    # 7. Manual test on example bounding box
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Testing on Example Bounding Box")
    print("=" * 80)

    test_bboxes = [
        {"name": "test", "x": 63, "y": 8, "w": 81, "h": 81, "cam_height": 310},
    ]

    img_width, img_height = 640, 480

    for bbox in test_bboxes:
        print(f"\nBounding Box: {bbox['name']}")
        print("-" * 80)

        # Absolute → relative
        x_center, y_center, w_rel, h_rel = absolute_to_relative(
            bbox["x"], bbox["y"], bbox["w"], bbox["h"], img_width, img_height
        )

        print(
            f"   Input (absolute): x={bbox['x']}, y={bbox['y']}, "
            f"w={bbox['w']}, h={bbox['h']}"
        )
        print(f"   Camera height: {bbox['cam_height']} cm")
        print(
            f"   Relative coords: x={x_center:.4f}, y={y_center:.4f}, "
            f"w={w_rel:.4f}, h={h_rel:.4f}"
        )

        # Features
        features = extract_features_from_bbox(
            x_center, y_center, w_rel, h_rel, bbox["cam_height"]
        )
        print("\n   Features:")
        print(f"      Aspect ratio:        {features[0]:.4f}")
        print(f"      Bbox area:           {features[1]:.4f}")
        print(f"      Distance from center:{features[2]:.4f}")
        print(f"      Camera height:       {features[3]:.0f} cm")

        # Scale features (StandardScaler from custom_scaler.py)
        features_scaled = scaler.transform(features)
        print(
            "\n   Scaled features: ["
            + ", ".join(f"{f:.4f}" for f in np.asarray(features_scaled))
            + "]"
        )

        # Predict
        prediction = detector.predict(features_scaled)
        score = detector.score_samples(features_scaled)   # negative distance
        mahalanobis_dist = -float(score)

        print("\n   Results:")
        print(f"      Mahalanobis distance: {mahalanobis_dist:.4f}")
        print(f"      Threshold:           {detector.threshold_:.4f}")
        print(f"      Anomaly score:       {float(score):.4f} (negative distance)")
        print(f"      Prediction:          {'NORMAL' if prediction == 1 else 'OUTLIER'}")

        if prediction == 1:
            print(
                f"      ✓ ACCEPT - Distance {mahalanobis_dist:.4f} "
                f"<= Threshold {detector.threshold_:.4f}"
            )
        else:
            print(
                f"      ✗ REJECT - Distance {mahalanobis_dist:.4f} "
                f"> Threshold {detector.threshold_:.4f}"
            )

    print("\n" + "=" * 80)
    print("Training and Testing Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
