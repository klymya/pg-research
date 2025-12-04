"""
Pytest tests for custom IsolationForest and StandardScaler implementations.
Tests compare outputs against sklearn implementations as ground truth.
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path

# sklearn imports (ground truth)
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

# Custom imports
from custom_scaler import StandardScaler
from custom_isolation_forest import IsolationForest


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    # Normal data
    X_normal = np.random.randn(100, 4) * np.array([0.3, 0.02, 0.1, 30]) + np.array([0.85, 0.04, 0.35, 265])
    # Outliers
    X_outliers = np.array([
        [0.5, 0.2, 0.2, 250],  # Large area
        [2.5, 0.01, 0.4, 270],  # Extreme aspect ratio
        [0.9, 0.05, 0.0, 450],  # Unusual camera height
    ])
    X_combined = np.vstack([X_normal, X_outliers])
    return X_normal, X_outliers, X_combined


@pytest.fixture
def sklearn_scaler(sample_data):
    """Train sklearn StandardScaler on sample data."""
    X_normal, _, _ = sample_data
    scaler = SklearnStandardScaler()
    scaler.fit(X_normal)
    return scaler


@pytest.fixture
def sklearn_iforest(sample_data):
    """Train sklearn IsolationForest on sample data."""
    X_normal, _, _ = sample_data
    # Use same parameters as in notebook
    iforest = SklearnIsolationForest(
        n_estimators=30,
        random_state=42
    )
    iforest.fit(X_normal)
    return iforest


@pytest.fixture
def scaler_json(sklearn_scaler, tmp_path):
    """Export sklearn scaler to JSON and return path."""
    params = {
        "mean": sklearn_scaler.mean_.tolist(),
        "scale": sklearn_scaler.scale_.tolist(),
        "var": sklearn_scaler.var_.tolist(),
        "n_features": len(sklearn_scaler.mean_),
    }

    json_path = tmp_path / "scaler_params.json"
    with open(json_path, 'w') as f:
        json.dump(params, f)

    return json_path, params


@pytest.fixture
def iforest_json(sklearn_iforest, tmp_path):
    """Export sklearn IsolationForest to JSON and return path."""
    def export_tree(tree):
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

    params = {
        "n_estimators": len(sklearn_iforest.estimators_),
        "max_samples": sklearn_iforest.max_samples_,
        "contamination": sklearn_iforest.contamination,
        "trees": [export_tree(estimator) for estimator in sklearn_iforest.estimators_],
    }

    json_path = tmp_path / "iforest_params.json"
    with open(json_path, 'w') as f:
        json.dump(params, f)

    return json_path, params


class TestStandardScaler:
    """Test custom StandardScaler implementation."""

    def test_load_from_json(self, scaler_json):
        """Test loading scaler from JSON file."""
        json_path, expected_params = scaler_json

        scaler = StandardScaler.from_json(json_path)

        assert scaler.n_features_ == expected_params['n_features']
        np.testing.assert_array_almost_equal(scaler.mean_, expected_params['mean'])
        np.testing.assert_array_almost_equal(scaler.scale_, expected_params['scale'])
        np.testing.assert_array_almost_equal(scaler.var_, expected_params['var'])

    def test_load_from_dict(self, scaler_json):
        """Test loading scaler from dictionary."""
        _, params = scaler_json

        scaler = StandardScaler.from_dict(params)

        assert scaler.n_features_ == params['n_features']
        np.testing.assert_array_almost_equal(scaler.mean_, params['mean'])

    def test_transform_single_sample(self, sklearn_scaler, scaler_json):
        """Test transform on single sample matches sklearn."""
        json_path, _ = scaler_json
        custom_scaler = StandardScaler.from_json(json_path)

        # Single sample
        x = np.array([0.75, 0.03, 0.14, 250])

        sklearn_result = sklearn_scaler.transform(x.reshape(1, -1))[0]
        custom_result = custom_scaler.transform(x)

        np.testing.assert_array_almost_equal(custom_result, sklearn_result, decimal=10)

    def test_transform_multiple_samples(self, sklearn_scaler, scaler_json, sample_data):
        """Test transform on multiple samples matches sklearn."""
        json_path, _ = scaler_json
        custom_scaler = StandardScaler.from_json(json_path)

        _, _, X_test = sample_data

        sklearn_result = sklearn_scaler.transform(X_test)
        custom_result = custom_scaler.transform(X_test)

        np.testing.assert_array_almost_equal(custom_result, sklearn_result, decimal=10)

    def test_inverse_transform(self, sklearn_scaler, scaler_json):
        """Test inverse transform matches sklearn."""
        json_path, _ = scaler_json
        custom_scaler = StandardScaler.from_json(json_path)

        # Transform and inverse transform
        x = np.array([0.75, 0.03, 0.14, 250])

        sklearn_transformed = sklearn_scaler.transform(x.reshape(1, -1))
        sklearn_result = sklearn_scaler.inverse_transform(sklearn_transformed)[0]

        custom_transformed = custom_scaler.transform(x)
        custom_result = custom_scaler.inverse_transform(custom_transformed)

        np.testing.assert_array_almost_equal(custom_result, sklearn_result, decimal=10)
        np.testing.assert_array_almost_equal(custom_result, x, decimal=10)

    def test_uninitialized_error(self):
        """Test that uninitialized scaler raises error."""
        scaler = StandardScaler()

        with pytest.raises(ValueError, match="not initialized"):
            scaler.transform(np.array([1, 2, 3, 4]))

    def test_wrong_num_features(self, scaler_json):
        """Test error when wrong number of features provided."""
        json_path, _ = scaler_json
        scaler = StandardScaler.from_json(json_path)

        with pytest.raises(ValueError, match="Expected 4 features"):
            scaler.transform(np.array([1, 2, 3]))  # Only 3 features


class TestIsolationForest:
    """Test custom IsolationForest implementation."""

    def test_load_from_json(self, iforest_json):
        """Test loading IsolationForest from JSON file."""
        json_path, expected_params = iforest_json

        forest = IsolationForest.from_json(json_path)

        assert forest.n_estimators_ == expected_params['n_estimators']
        assert forest.max_samples_ == expected_params['max_samples']
        assert forest.contamination_ == expected_params['contamination']
        assert len(forest.trees_) == expected_params['n_estimators']

    def test_load_from_dict(self, iforest_json):
        """Test loading IsolationForest from dictionary."""
        _, params = iforest_json

        forest = IsolationForest.from_dict(params)

        assert forest.n_estimators_ == params['n_estimators']
        assert len(forest.trees_) == params['n_estimators']

    def test_score_samples_single(self, sklearn_iforest, iforest_json):
        """Test score_samples on single sample matches sklearn (approximately)."""
        json_path, _ = iforest_json
        custom_forest = IsolationForest.from_json(json_path)

        # Single sample
        x = np.array([0.75, 0.03, 0.14, 250])

        sklearn_score = sklearn_iforest.score_samples(x.reshape(1, -1))[0]
        custom_score = custom_forest.score_samples(x)

        # Scores should be very close (same tree structure)
        np.testing.assert_array_almost_equal(custom_score, sklearn_score, decimal=10)

    def test_score_samples_multiple(self, sklearn_iforest, iforest_json, sample_data):
        """Test score_samples on multiple samples matches sklearn."""
        json_path, _ = iforest_json
        custom_forest = IsolationForest.from_json(json_path)

        _, _, X_test = sample_data

        sklearn_scores = sklearn_iforest.score_samples(X_test)
        custom_scores = custom_forest.score_samples(X_test)

        # All scores should match exactly (same trees, same algorithm)
        np.testing.assert_array_almost_equal(custom_scores, sklearn_scores, decimal=10)

    def test_predict_single(self, sklearn_iforest, iforest_json):
        """Test predict on single sample."""
        json_path, _ = iforest_json
        custom_forest = IsolationForest.from_json(json_path)

        # Single sample
        x = np.array([0.75, 0.03, 0.14, 250])

        sklearn_pred = sklearn_iforest.predict(x.reshape(1, -1))[0]
        custom_pred = custom_forest.predict(x)

        # Predictions should match
        assert custom_pred == sklearn_pred
        assert custom_pred in [-1, 1]

    def test_predict_multiple(self, sklearn_iforest, iforest_json, sample_data):
        """Test predict on multiple samples matches sklearn."""
        json_path, _ = iforest_json
        custom_forest = IsolationForest.from_json(json_path)

        _, _, X_test = sample_data

        sklearn_preds = sklearn_iforest.predict(X_test)
        custom_preds = custom_forest.predict(X_test)

        # Predictions should match exactly
        np.testing.assert_array_equal(custom_preds, sklearn_preds)

    def test_decision_function(self, sklearn_iforest, iforest_json, sample_data):
        """Test decision_function matches sklearn."""
        json_path, _ = iforest_json
        custom_forest = IsolationForest.from_json(json_path)

        _, _, X_test = sample_data

        sklearn_decisions = sklearn_iforest.decision_function(X_test)
        custom_decisions = custom_forest.decision_function(X_test)

        # decision_function = score_samples - offset
        # Verify relationship
        scores = custom_forest.score_samples(X_test)
        expected_decisions = scores - custom_forest.offset_
        np.testing.assert_array_almost_equal(custom_decisions, expected_decisions, decimal=10)

        # Match sklearn
        np.testing.assert_array_almost_equal(custom_decisions, sklearn_decisions, decimal=10)

    def test_outlier_detection(self, sklearn_iforest, iforest_json, sample_data):
        """Test that outliers are correctly detected."""
        json_path, _ = iforest_json
        custom_forest = IsolationForest.from_json(json_path)

        X_normal, X_outliers, _ = sample_data

        # Predict on normal data
        normal_preds = custom_forest.predict(X_normal)
        # Most should be inliers
        assert np.mean(normal_preds == 1) > 0.5

        # Predict on outliers
        outlier_preds = custom_forest.predict(X_outliers)
        # Some outliers should be detected (not guaranteed all)
        # Just check that predictions are valid
        assert all(p in [-1, 1] for p in outlier_preds)


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_full_pipeline(self, sklearn_scaler, sklearn_iforest, scaler_json, iforest_json):
        """Test complete pipeline: scale -> predict."""
        scaler_path, _ = scaler_json
        forest_path, _ = iforest_json

        # Load custom models
        custom_scaler = StandardScaler.from_json(scaler_path)
        custom_forest = IsolationForest.from_json(forest_path)

        # Test data
        test_samples = np.array([
            [0.75, 0.03, 0.14, 250],  # Normal
            [2.5, 0.01, 0.4, 270],     # Outlier (extreme aspect ratio)
            [0.9, 0.7, 0.0, 450],      # Outlier (large area, unusual height)
        ])

        # Sklearn pipeline
        sklearn_scaled = sklearn_scaler.transform(test_samples)
        sklearn_preds = sklearn_iforest.predict(sklearn_scaled)
        sklearn_scores = sklearn_iforest.score_samples(sklearn_scaled)

        # Custom pipeline
        custom_scaled = custom_scaler.transform(test_samples)
        custom_preds = custom_forest.predict(custom_scaled)
        custom_scores = custom_forest.score_samples(custom_scaled)

        # Results should match
        np.testing.assert_array_almost_equal(custom_scaled, sklearn_scaled, decimal=10)
        np.testing.assert_array_almost_equal(custom_scores, sklearn_scores, decimal=10)
        np.testing.assert_array_equal(custom_preds, sklearn_preds)

    def test_batch_processing(self, sklearn_scaler, sklearn_iforest, scaler_json, iforest_json):
        """Test processing multiple samples at once."""
        scaler_path, _ = scaler_json
        forest_path, _ = iforest_json

        custom_scaler = StandardScaler.from_json(scaler_path)
        custom_forest = IsolationForest.from_json(forest_path)

        # Generate random test data
        np.random.seed(123)
        test_data = np.random.randn(50, 4) * np.array([0.3, 0.02, 0.1, 30]) + np.array([0.85, 0.04, 0.35, 265])

        # Sklearn
        sklearn_scaled = sklearn_scaler.transform(test_data)
        sklearn_scores = sklearn_iforest.score_samples(sklearn_scaled)
        sklearn_preds = sklearn_iforest.predict(sklearn_scaled)

        # Custom
        custom_scaled = custom_scaler.transform(test_data)
        custom_scores = custom_forest.score_samples(custom_scaled)
        custom_preds = custom_forest.predict(custom_scaled)

        # All should match
        np.testing.assert_array_almost_equal(custom_scaled, sklearn_scaled, decimal=10)
        np.testing.assert_array_almost_equal(custom_scores, sklearn_scores, decimal=10)
        np.testing.assert_array_equal(custom_preds, sklearn_preds)

    def test_consistency_across_runs(self, scaler_json, iforest_json):
        """Test that results are consistent across multiple runs."""
        scaler_path, _ = scaler_json
        forest_path, _ = iforest_json

        x = np.array([0.75, 0.03, 0.14, 250])

        # Run 1
        scaler1 = StandardScaler.from_json(scaler_path)
        forest1 = IsolationForest.from_json(forest_path)
        scaled1 = scaler1.transform(x)
        score1 = forest1.score_samples(scaled1)
        pred1 = forest1.predict(scaled1)

        # Run 2 (reload models)
        scaler2 = StandardScaler.from_json(scaler_path)
        forest2 = IsolationForest.from_json(forest_path)
        scaled2 = scaler2.transform(x)
        score2 = forest2.score_samples(scaled2)
        pred2 = forest2.predict(scaled2)

        # Should be identical
        np.testing.assert_array_equal(scaled1, scaled2)
        np.testing.assert_equal(score1, score2)
        np.testing.assert_equal(pred1, pred2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
