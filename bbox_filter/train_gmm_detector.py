"""
Train CustomGMMDetector on YOLO annotations and test it.

Pipeline:
1. Load YOLO annotations from CSV
2. Build feature matrices for train/test
3. Scale features with StandardScaler (custom_scaler.py)
4. Train CustomGMMDetector (GMM-based novelty detector)
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
from custom_gmm_detector import CustomGMMDetector


# -----------------------------------------------------------------------------
# Feature extraction helpers
# -----------------------------------------------------------------------------
def extract_features_from_bbox(x_center, y_center, w_rel, h_rel, cam_height):
    """
    Extract features from a bbox for novelty detection.

    Features:
      - aspect_ratio = w / h
      - bbox_area = w * h
      - dist_from_center = distance of bbox center from (0.5, 0.5) in normalized coords
      - cam_height (as-is, will be scaled later)
    """
    if h_rel <= 0:
        aspect_ratio = 0.0
    else:
        aspect_ratio = w_rel / h_rel

    bbox_area = w_rel * h_rel
    dist_from_center = float(np.sqrt((x_center - 0.5) ** 2 + (y_center - 0.5) ** 2))

    return np.array([aspect_ratio, bbox_area, dist_from_center, cam_height], dtype=float)


def absolute_to_relative(x_abs, y_abs, w_abs, h_abs, img_width, img_height):
    """
    Convert absolute bbox coordinates to relative YOLO-like format.

      x_rel, y_rel – top-left corner in [0,1]
      w_rel, h_rel – width/height in [0,1]
      x_center, y_center – bbox center in [0,1]
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
    print("Training Custom GMM Detector (GMM-based novelty detector)")
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

    # Basic sanity check for weird values
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
        print("   ✗ Scaled training features contain NaN or Inf. Check scaler and data.")
        sys.exit(1)

    print("   ✓ Features scaled (train mean≈0, std≈1)")

    # -------------------------------------------------------------------------
    # 4. Train CustomGMMDetector
    # -------------------------------------------------------------------------
    print("\n4. Training CustomGMMDetector (GMM)...")

    # You can tweak these from the top if needed
    n_components = 5
    covariance_type = "full"       # 'full', 'diag', 'spherical'
    threshold_ok = 0.90            # quantile on anomaly scores
    threshold_novel = 0.95
    max_iter = 100
    reg_covar = 1e-6
    random_state = 42

    print("   Parameters:")
    print(f"      n_components:    {n_components}")
    print(f"      covariance_type: {covariance_type}")
    print(f"      threshold_ok:    {threshold_ok:.2f} (quantile of anomaly score)")
    print(f"      threshold_novel: {threshold_novel:.2f} (quantile of anomaly score)")
    print(f"      max_iter:        {max_iter}")
    print(f"      reg_covar:       {reg_covar:e}")
    print(f"      random_state:    {random_state}")

    if X_train_scaled.shape[0] < n_components:
        print(
            f"   ✗ Not enough training samples ({X_train_scaled.shape[0]}) "
            f"for n_components = {n_components}"
        )
        sys.exit(1)

    gmm_detector = CustomGMMDetector(
        n_components=n_components,
        covariance_type=covariance_type,
        threshold_ok=threshold_ok,
        threshold_novel=threshold_novel,
        max_iter=max_iter,
        reg_covar=reg_covar,
        random_state=random_state,
    )

    gmm_detector.fit(X_train_scaled)

    print(f"\n   ✓ Model trained: {gmm_detector}")
    print(f"   ✓ Converged:     {gmm_detector.converged_}")
    print(f"   ✓ Iterations:    {gmm_detector.n_iter_}")
    print(f"   ✓ Threshold 'ok':    {gmm_detector.threshold_ok_:.4f}")
    print(f"   ✓ Threshold 'novel': {gmm_detector.threshold_novel_:.4f}")

    # Component weights
    print(f"\n   Component weights:")
    for k, w in enumerate(gmm_detector.weights_):
        print(f"      Component {k}: {w:.4f} ({w * 100:.2f}%)")

    # -------------------------------------------------------------------------
    # 5. Evaluate on training & test
    # -------------------------------------------------------------------------
    print("\n5. Evaluating on datasets...")

    # Training set
    train_pred = gmm_detector.predict(X_train_scaled)
    train_scores = gmm_detector.anomaly_score(X_train_scaled)  # higher = more anomalous

    train_ok = np.sum(train_pred == "ok")
    train_novel = np.sum(train_pred == "novel")
    train_outlier = np.sum(train_pred == "outlier")

    print(f"\n   Training set:")
    print(f"      Samples: {len(train_pred)}")
    print(f"      OK:      {train_ok} ({train_ok / len(train_pred) * 100:.2f}%)")
    print(f"      Novel:   {train_novel} ({train_novel / len(train_pred) * 100:.2f}%)")
    print(f"      Outlier: {train_outlier} ({train_outlier / len(train_pred) * 100:.2f}%)")
    print(f"      Score range (anomaly): [{train_scores.min():.3f}, {train_scores.max():.3f}]")

    # Test set (if present)
    if X_test_scaled is not None and len(X_test_scaled) > 0:
        test_pred = gmm_detector.predict(X_test_scaled)
        test_scores = gmm_detector.anomaly_score(X_test_scaled)

        test_ok = np.sum(test_pred == "ok")
        test_novel = np.sum(test_pred == "novel")
        test_outlier = np.sum(test_pred == "outlier")

        print(f"\n   Test set:")
        print(f"      Samples: {len(test_pred)}")
        print(f"      OK:      {test_ok} ({test_ok / len(test_pred) * 100:.2f}%)")
        print(f"      Novel:   {test_novel} ({test_novel / len(test_pred) * 100:.2f}%)")
        print(f"      Outlier: {test_outlier} ({test_outlier / len(test_pred) * 100:.2f}%)")
        print(f"      Score range (anomaly): [{test_scores.min():.3f}, {test_scores.max():.3f}]")
    else:
        print("\n   Test set: (none)")

    # -------------------------------------------------------------------------
    # 6. Save models (for C++ inference)
    # -------------------------------------------------------------------------
    print("\n6. Saving models to JSON...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "gmm_detector_model")
    os.makedirs(model_dir, exist_ok=True)
    scaler.to_json(os.path.join(model_dir, "gmm_detector_scaler.json"))
    gmm_detector.to_json(os.path.join(model_dir, "gmm_detector_params.json"))
    print("   ✓ Saved gmm_detector_scaler.json")
    print("   ✓ Saved gmm_detector_params.json")

    # -------------------------------------------------------------------------
    # 7. Manual test on example bbox
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

        # Convert to relative coordinates
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

        # Extract features
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
        features_scaled_arr = np.asarray(features_scaled, dtype=float)
        print(
            "\n   Scaled features: ["
            + ", ".join(f"{f:.4f}" for f in features_scaled_arr)
            + "]"
        )

        # Predict
        prediction = gmm_detector.predict(features_scaled_arr)
        anomaly_score = float(gmm_detector.anomaly_score(features_scaled_arr))
        log_likelihood = float(gmm_detector.score_samples(features_scaled_arr))

        print("\n   Results:")
        print(f"      Log-likelihood: {log_likelihood:.4f}")
        print(f"      Anomaly score:  {anomaly_score:.4f} (higher = more anomalous)")
        print(f"      Threshold 'ok':    {gmm_detector.threshold_ok_:.4f}")
        print(f"      Threshold 'novel': {gmm_detector.threshold_novel_:.4f}")
        print(f"      Prediction: {prediction.upper()}")

        if prediction == "ok":
            print(
                f"      ✓ ACCEPT - Score {anomaly_score:.4f} "
                f"<= Threshold {gmm_detector.threshold_ok_:.4f}"
            )
        elif prediction == "novel":
            print(
                f"      ⚠ WARNING - Score {anomaly_score:.4f} in range "
                f"[{gmm_detector.threshold_ok_:.4f}, {gmm_detector.threshold_novel_:.4f}]"
            )
        else:
            print(
                f"      ✗ REJECT - Score {anomaly_score:.4f} "
                f"> Threshold {gmm_detector.threshold_novel_:.4f}"
            )

    print("\n" + "=" * 80)
    print("Training and Testing Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
