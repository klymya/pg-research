"""
Custom Novelty Detector for Bounding Box Filtering.

This implementation uses statistical methods to detect anomalous bounding boxes
based on Mahalanobis distance and percentile-based thresholding.
"""

import numpy as np
import json


class CustomNoveltyDetector:
    """
    Custom novelty detector using Mahalanobis distance and statistical thresholds.

    This detector computes the Mahalanobis distance of samples from the training
    distribution and uses percentile-based thresholds for anomaly detection.
    """

    def __init__(self, contamination=0.1):
        """
        Initialize the novelty detector.

        Args:
            contamination: Expected proportion of outliers (0 < contamination <= 0.5)
        """
        self.contamination = contamination
        self.mean_ = None
        self.cov_ = None
        self.inv_cov_ = None
        self.threshold_ = None
        self.n_features_ = None

    def fit(self, X):
        """
        Fit the novelty detector to training data.

        Args:
            X: Training data of shape (n_samples, n_features)
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_ = X.shape[1]

        # Calculate mean and covariance
        self.mean_ = np.mean(X, axis=0)
        self.cov_ = np.cov(X, rowvar=False)

        # Add small regularization to ensure invertibility
        if self.n_features_ > 1:
            self.cov_ += np.eye(self.n_features_) * 1e-6
        else:
            self.cov_ = np.array([[self.cov_ + 1e-6]])

        # Compute inverse covariance
        self.inv_cov_ = np.linalg.inv(self.cov_)

        # Calculate Mahalanobis distances for training data
        distances = self._mahalanobis_distance(X)

        # Set threshold based on contamination percentile
        percentile = 100 * (1 - self.contamination)
        self.threshold_ = np.percentile(distances, percentile)

        return self

    def _mahalanobis_distance(self, X):
        """
        Calculate Mahalanobis distance for samples.

        Args:
            X: Data of shape (n_samples, n_features) or (n_features,)

        Returns:
            Array of Mahalanobis distances
        """
        X = np.array(X)

        # Handle single sample
        if X.ndim == 1:
            diff = X - self.mean_
            distance = np.sqrt(diff @ self.inv_cov_ @ diff)
            return distance

        # Handle multiple samples
        diff = X - self.mean_
        distances = np.sqrt(np.sum(diff @ self.inv_cov_ * diff, axis=1))
        return distances

    def decision_function(self, X):
        """
        Calculate decision function (negative Mahalanobis distance).

        Larger (less negative) values indicate more normal samples.

        Args:
            X: Data of shape (n_samples, n_features) or (n_features,)

        Returns:
            Array of decision scores
        """
        if self.mean_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        distances = self._mahalanobis_distance(X)
        # Return negative distance (sklearn convention: higher = more normal)
        return -distances

    def score_samples(self, X):
        """
        Calculate anomaly scores (negative Mahalanobis distance).

        This is an alias for decision_function for consistency with sklearn.

        Args:
            X: Data of shape (n_samples, n_features) or (n_features,)

        Returns:
            Array of anomaly scores
        """
        return self.decision_function(X)

    def predict(self, X):
        """
        Predict if samples are outliers or inliers.

        Args:
            X: Data of shape (n_samples, n_features) or (n_features,)

        Returns:
            Array of predictions:
                1 for inliers
                -1 for outliers
        """
        if self.mean_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        distances = self._mahalanobis_distance(X)

        # Single sample
        if isinstance(distances, (int, float, np.number)):
            return 1 if distances <= self.threshold_ else -1

        # Multiple samples
        return np.where(distances <= self.threshold_, 1, -1)

    def to_dict(self):
        """
        Export model parameters to dictionary.

        Returns:
            Dictionary with model parameters
        """
        if self.mean_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return {
            "contamination": self.contamination,
            "mean": self.mean_.tolist(),
            "cov": self.cov_.tolist(),
            "inv_cov": self.inv_cov_.tolist(),
            "threshold": float(self.threshold_),
            "n_features": int(self.n_features_)
        }

    def to_json(self, json_path):
        """
        Save model parameters to JSON file.

        Args:
            json_path: Path to save JSON file
        """
        params = self.to_dict()
        with open(json_path, 'w') as f:
            json.dump(params, f, indent=2)

    @classmethod
    def from_dict(cls, params):
        """
        Load model from dictionary.

        Args:
            params: Dictionary with model parameters

        Returns:
            CustomNoveltyDetector instance
        """
        detector = cls(contamination=params['contamination'])
        detector.mean_ = np.array(params['mean'])
        detector.cov_ = np.array(params['cov'])
        detector.inv_cov_ = np.array(params['inv_cov'])
        detector.threshold_ = params['threshold']
        detector.n_features_ = params['n_features']
        return detector

    @classmethod
    def from_json(cls, json_path):
        """
        Load model from JSON file.

        Args:
            json_path: Path to JSON file

        Returns:
            CustomNoveltyDetector instance
        """
        with open(json_path, 'r') as f:
            params = json.load(f)
        return cls.from_dict(params)

    def __repr__(self):
        if self.mean_ is None:
            return "CustomNoveltyDetector(not fitted)"
        return f"CustomNoveltyDetector(contamination={self.contamination}, n_features={self.n_features_})"
