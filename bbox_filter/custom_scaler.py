import numpy as np
import json


class StandardScaler:
    """
    Simple StandardScaler (mean/std) for compatibility with C++ inference.
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_features_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0  # avoid division by zero
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise ValueError("Scaler not fitted yet")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            return (X - self.mean_) / self.scale_

        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    # -----------------------------
    # JSON EXPORT FOR C++
    # -----------------------------
    def to_dict(self):
        if self.mean_ is None:
            raise ValueError("Scaler not fitted")

        return {
            "n_features": int(self.n_features_),
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist()
        }

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    # -----------------------------
    # LOAD BACK IF NEEDED
    # -----------------------------
    @classmethod
    def from_json(cls, path):
        with open(path, "r") as f:
            params = json.load(f)

        scaler = cls()
        scaler.n_features_ = params["n_features"]
        scaler.mean_ = np.array(params["mean"], dtype=float)
        scaler.scale_ = np.array(params["scale"], dtype=float)
        return scaler

    def __repr__(self):
        if self.mean_ is None:
            return "StandardScaler(not fitted)"
        return f"StandardScaler(n_features={self.n_features_})"
