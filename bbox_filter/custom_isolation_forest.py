"""
Custom IsolationForest implementation using pure Python + NumPy.
Can be initialized from JSON parameters exported from sklearn.
"""

import numpy as np
import json


class IsolationTree:
    """Single isolation tree for the IsolationForest."""

    def __init__(self, feature, threshold, left, right, n_node_samples, node_count, max_depth):
        """
        Initialize isolation tree from exported parameters.

        Args:
            feature: array of feature indices for each node
            threshold: array of split thresholds for each node
            left: array of left child indices
            right: array of right child indices
            n_node_samples: array of sample counts at each node
            node_count: total number of nodes
            max_depth: maximum depth of tree
        """
        self.feature = np.array(feature)
        self.threshold = np.array(threshold)
        self.left = np.array(left)
        self.right = np.array(right)
        self.n_node_samples = np.array(n_node_samples)
        self.node_count = node_count
        self.max_depth = max_depth

    def _path_length(self, x, node_id=0, current_depth=0):
        """
        Calculate path length for a single sample through the tree.

        Args:
            x: feature vector (1D numpy array)
            node_id: current node index
            current_depth: current depth in tree

        Returns:
            Path length from root to leaf
        """
        # Leaf node (no split)
        if self.left[node_id] == self.right[node_id]:  # -1 == -1 means leaf
            # Adjust path length with average path length of unsuccessful search in BST
            n = self.n_node_samples[node_id]
            if n <= 1:
                return current_depth
            else:
                # Average path length of unsuccessful search in BST
                c = self._c(n)
                return current_depth + c

        # Internal node
        feature_idx = self.feature[node_id]
        threshold_val = self.threshold[node_id]

        # Go left or right based on feature value
        if x[feature_idx] <= threshold_val:
            return self._path_length(x, self.left[node_id], current_depth + 1)
        else:
            return self._path_length(x, self.right[node_id], current_depth + 1)

    @staticmethod
    def _c(n):
        """
        Average path length of unsuccessful search in BST.

        This matches sklearn's _average_path_length function exactly.

        Formula from paper:
        c(n) = 2*H(n-1) - (2*(n-1)/n)
        where H(n) is harmonic number ≈ ln(n) + euler_gamma

        Special cases:
        - n <= 1: return 0.0
        - n == 2: return 1.0
        - n > 2: use formula

        Args:
            n: number of samples

        Returns:
            Average path length
        """
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        # Use Euler's constant (euler_gamma)
        euler_gamma = 0.5772156649015329
        harmonic = np.log(n - 1.0) + euler_gamma
        return 2.0 * harmonic - (2.0 * (n - 1.0) / n)


class IsolationForest:
    """
    IsolationForest for anomaly detection using isolation trees.

    Can be initialized from JSON parameters exported from sklearn's IsolationForest.
    """

    def __init__(self):
        """Initialize empty IsolationForest."""
        self.n_estimators_ = None
        self.max_samples_ = None
        self.contamination_ = None
        self.trees_ = []
        self.offset_ = None
        self.threshold_ = None

    @classmethod
    def from_json(cls, json_path):
        """
        Load IsolationForest parameters from JSON file.

        Args:
            json_path: Path to JSON file with IsolationForest parameters

        Returns:
            IsolationForest instance initialized with loaded parameters
        """
        with open(json_path, 'r') as f:
            params = json.load(f)

        return cls.from_dict(params)

    @classmethod
    def from_dict(cls, params):
        """
        Load IsolationForest parameters from dictionary.

        Args:
            params: Dictionary with IsolationForest parameters

        Returns:
            IsolationForest instance initialized with loaded parameters
        """
        forest = cls()
        forest.n_estimators_ = params['n_estimators']

        # Handle max_samples - ensure it's an integer
        max_samples = params['max_samples']
        if isinstance(max_samples, str):
            # If it's a string like "auto", use the number of samples from first tree
            # or infer from the tree structure
            if params['trees']:
                # Use max n_node_samples from root nodes as approximation
                max_samples = max(tree['n_node_samples'][0] for tree in params['trees'])
        forest.max_samples_ = int(max_samples)

        # Handle contamination
        contamination = params['contamination']
        if isinstance(contamination, str) and contamination == 'auto':
            forest.contamination_ = 'auto'
        else:
            forest.contamination_ = float(contamination)

        # Load trees
        forest.trees_ = []
        for tree_params in params['trees']:
            tree = IsolationTree(
                feature=tree_params['feature'],
                threshold=tree_params['threshold'],
                left=tree_params['left'],
                right=tree_params['right'],
                n_node_samples=tree_params['n_node_samples'],
                node_count=tree_params['node_count'],
                max_depth=tree_params['max_depth']
            )
            forest.trees_.append(tree)

        # Calculate normalization factor (c(max_samples))
        # This matches sklearn's average_path_length_max_samples
        if forest.max_samples_ > 1:
            forest._c_max_samples = IsolationTree._c(forest.max_samples_)
        else:
            forest._c_max_samples = 0.0

        # sklearn offset is always -0.5
        forest.offset_ = -0.5

        return forest

    def _anomaly_score(self, x):
        """
        Calculate anomaly score for a single sample.

        Score is based on average path length across all trees.
        Lower path length = more anomalous.

        Args:
            x: feature vector (1D numpy array)

        Returns:
            Anomaly score in range [0, 1]
            Higher score = more anomalous
        """
        # Get average path length across all trees
        path_lengths = [tree._path_length(x) for tree in self.trees_]
        avg_path_length = np.mean(path_lengths)

        # Normalize by expected average path length
        if self.offset_ > 0:
            score = 2.0 ** (-avg_path_length / self.offset_)
        else:
            score = 1.0

        return score

    def score_samples(self, X):
        """
        Calculate anomaly scores for samples.

        The anomaly score is computed as:
        score = -2^(-depths / (n_estimators * c(max_samples)))

        Where depths is the sum of path lengths across all trees,
        and c(n) is the average path length of unsuccessful search in BST.

        This matches sklearn's implementation exactly.

        Args:
            X: numpy array of shape (n_samples, n_features) or (n_features,)

        Returns:
            Array of anomaly scores (more negative = more anomalous)
        """
        X = np.array(X)

        # Handle single sample
        if X.ndim == 1:
            # Sum path lengths across all trees
            depths = sum(tree._path_length(X) for tree in self.trees_)
            # Normalize by n_estimators * c(max_samples)
            denominator = self.n_estimators_ * self._c_max_samples
            if denominator != 0:
                normalized_depth = depths / denominator
            else:
                normalized_depth = 0
            # sklearn formula: -2^(-normalized_depth)
            return -(2 ** (-normalized_depth))

        # Handle multiple samples
        scores = []
        for x in X:
            depths = sum(tree._path_length(x) for tree in self.trees_)
            denominator = self.n_estimators_ * self._c_max_samples
            if denominator != 0:
                normalized_depth = depths / denominator
            else:
                normalized_depth = 0
            scores.append(-(2 ** (-normalized_depth)))

        return np.array(scores)

    def predict(self, X):
        """
        Predict if samples are outliers or inliers.

        In sklearn, the prediction uses decision_function:
        - decision_function = score_samples - offset
        - If decision_function < 0: outlier (-1)
        - If decision_function >= 0: inlier (1)

        Args:
            X: numpy array of shape (n_samples, n_features) or (n_features,)

        Returns:
            Array of predictions:
                1 for inliers
                -1 for outliers
        """
        scores = self.score_samples(X)

        # sklearn uses decision_function = score_samples - offset
        # Then predicts: decision_function < 0 => outlier (-1), else inlier (1)
        decision = scores - self.offset_

        # Return predictions
        if isinstance(decision, np.ndarray):
            return np.where(decision < 0, -1, 1)
        else:
            return -1 if decision < 0 else 1

    def decision_function(self, X):
        """
        Shifted anomaly score: score_samples - offset.

        In sklearn, decision_function = score_samples - offset.
        This shifted score is used for prediction:
        - decision_function < 0: outlier
        - decision_function >= 0: inlier

        Args:
            X: numpy array of shape (n_samples, n_features) or (n_features,)

        Returns:
            Array of decision scores (score_samples - offset)
        """
        return self.score_samples(X) - self.offset_

    def __repr__(self):
        if self.n_estimators_ is None:
            return "IsolationForest(not initialized)"
        return f"IsolationForest(n_estimators={self.n_estimators_}, contamination={self.contamination_})"
