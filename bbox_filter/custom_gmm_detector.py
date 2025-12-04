import numpy as np
import json
import math


class CustomGMMDetector:
    """
    Custom GMM-based novelty detector.

    Models the training data as a mixture of K Gaussian distributions
    and uses negative log-likelihood as anomaly score.
    """

    def __init__(
        self,
        n_components=3,
        covariance_type="full",           # 'full', 'diag', 'spherical'
        threshold_ok=0.95,                # quantile for "ok"
        threshold_novel=0.99,             # quantile for "novel"
        max_iter=100,
        tol=1e-3,
        reg_covar=1e-6,
        random_state=None,
    ):
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.threshold_ok = float(threshold_ok)
        self.threshold_novel = float(threshold_novel)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.reg_covar = float(reg_covar)
        self.random_state = random_state

        # learned params
        self.weights_ = None      # shape (K,)
        self.means_ = None        # shape (K, D)
        self.covariances_ = None  # shape depends on covariance_type
        self.n_features_ = None

        # thresholds on anomaly score
        self.threshold_ok_ = None
        self.threshold_novel_ = None

        # convergence info
        self.converged_ = False
        self.n_iter_ = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _check_covariance_type(self):
        if self.covariance_type not in ("full", "diag", "spherical"):
            raise ValueError(
                f"Unsupported covariance_type: {self.covariance_type}. "
                f"Use 'full', 'diag', or 'spherical'."
            )

    def _initialize_parameters(self, X):
        """Random init: sample means from data, shared covariance."""
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        rng = np.random.RandomState(self.random_state)

        # means: choose K distinct samples
        indices = rng.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[indices].copy()

        # initial weights: uniform
        self.weights_ = np.full(self.n_components, 1.0 / self.n_components, dtype=float)

        # global empirical covariance
        emp_cov = np.cov(X, rowvar=False)
        emp_cov += self.reg_covar * np.eye(n_features)

        if self.covariance_type == "full":
            # one full matrix per component
            self.covariances_ = np.repeat(emp_cov[None, :, :], self.n_components, axis=0)
        elif self.covariance_type == "diag":
            # one diagonal vector per component
            diag = np.diag(emp_cov)
            self.covariances_ = np.repeat(diag[None, :], self.n_components, axis=0)
        elif self.covariance_type == "spherical":
            # one scalar variance per component
            var = np.trace(emp_cov) / n_features
            self.covariances_ = np.full(self.n_components, var, dtype=float)

    def _estimate_log_gaussian_prob(self, X):
        """
        Compute matrix of log p(X | component k), shape (n_samples, K).
        """
        n_samples, n_features = X.shape
        log_prob = np.empty((n_samples, self.n_components), dtype=float)

        for k in range(self.n_components):
            mean = self.means_[k]
            cov = self.covariances_[k]
            diff = X - mean

            if self.covariance_type == "full":
                # ensure positive definite
                cov_k = cov + self.reg_covar * np.eye(n_features)
                sign, logdet = np.linalg.slogdet(cov_k)
                if sign <= 0:
                    cov_k = cov_k + 1e-6 * np.eye(n_features)
                    sign, logdet = np.linalg.slogdet(cov_k)
                inv_cov = np.linalg.inv(cov_k)
                # Mahalanobis distance
                m = np.sum(diff @ inv_cov * diff, axis=1)

            elif self.covariance_type == "diag":
                var = cov + self.reg_covar
                logdet = np.sum(np.log(var))
                m = np.sum(diff * diff / var, axis=1)

            elif self.covariance_type == "spherical":
                var = cov + self.reg_covar
                logdet = n_features * math.log(var)
                m = np.sum(diff * diff, axis=1) / var

            else:
                raise ValueError(f"Unsupported covariance_type: {self.covariance_type}")

            log_prob[:, k] = -0.5 * (n_features * math.log(2.0 * math.pi) + logdet + m)

        return log_prob

    def _estimate_log_prob_resp(self, X):
        """
        Compute:
          - per-sample log p(x_i)
          - responsibilities gamma_ik
          - total log-likelihood sum_i log p(x_i)
        """
        # log p(x | k)
        log_gauss = self._estimate_log_gaussian_prob(X)         # (N, K)
        log_weights = np.log(self.weights_ + 1e-16)             # (K,)

        # log p(x, k) = log pi_k + log N(x | mu_k, Sigma_k)
        log_prob = log_gauss + log_weights                      # (N, K)

        # log p(x) = logsumexp over components
        log_prob_max = log_prob.max(axis=1, keepdims=True)      # (N, 1)
        log_prob_shifted = log_prob - log_prob_max
        np.exp(log_prob_shifted, out=log_prob_shifted)          # now exp(log_prob - max)
        sum_exp = log_prob_shifted.sum(axis=1, keepdims=True)   # (N, 1)
        log_prob_norm = log_prob_max + np.log(sum_exp)          # (N, 1)
        log_likelihood = float(log_prob_norm.sum())

        # responsibilities gamma_ik = p(k | x_i)
        resp = log_prob_shifted / sum_exp                       # (N, K)

        return log_prob_norm.squeeze(axis=1), resp, log_likelihood

    def _m_step(self, X, resp):
        """
        Update weights, means, covariances using responsibilities.
        """
        n_samples, n_features = X.shape
        nk = resp.sum(axis=0) + 10 * np.finfo(float).eps   # shape (K,)

        # weights
        self.weights_ = nk / n_samples

        # means
        self.means_ = (resp.T @ X) / nk[:, None]

        # covariances
        if self.covariance_type == "full":
            covs = np.empty((self.n_components, n_features, n_features), dtype=float)
            for k in range(self.n_components):
                diff = X - self.means_[k]
                weighted = diff * resp[:, k][:, None]       # (N, D)
                cov_k = (weighted.T @ diff) / nk[k]         # (D, D)
                cov_k.flat[:: n_features + 1] += self.reg_covar
                covs[k] = cov_k
            self.covariances_ = covs

        elif self.covariance_type == "diag":
            covs = np.empty((self.n_components, n_features), dtype=float)
            for k in range(self.n_components):
                diff = X - self.means_[k]
                cov_k = (resp[:, k][:, None] * diff * diff).sum(axis=0) / nk[k]
                cov_k += self.reg_covar
                covs[k] = cov_k
            self.covariances_ = covs

        elif self.covariance_type == "spherical":
            covs = np.empty(self.n_components, dtype=float)
            for k in range(self.n_components):
                diff = X - self.means_[k]
                cov_k = (resp[:, k][:, None] * diff * diff).sum() / (nk[k] * n_features)
                cov_k += self.reg_covar
                covs[k] = cov_k
            self.covariances_ = covs

        else:
            raise ValueError(f"Unsupported covariance_type: {self.covariance_type}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X):
        """
        Fit GMM to training data using EM algorithm.

        X: array-like, shape (n_samples, n_features)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self._check_covariance_type()

        if n_samples < self.n_components:
            raise ValueError("n_samples < n_components: cannot fit GMM")

        # initialize
        self._initialize_parameters(X)

        prev_log_likelihood = -np.inf

        for i in range(self.max_iter):
            # E-step
            _, resp, log_likelihood = self._estimate_log_prob_resp(X)

            if np.isnan(log_likelihood):
                # numerical issue: abort early
                break

            # M-step
            self._m_step(X, resp)

            # convergence check
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                self.converged_ = True
                self.n_iter_ = i + 1
                break

            prev_log_likelihood = log_likelihood

        if not self.converged_:
            self.n_iter_ = self.max_iter

        # thresholds computed on training anomaly scores
        train_scores = self.anomaly_score(X)
        self.threshold_ok_ = float(np.quantile(train_scores, self.threshold_ok))
        self.threshold_novel_ = float(np.quantile(train_scores, self.threshold_novel))

        return self

    def score_samples(self, X):
        """
        Per-sample log-likelihood log p(x).

        Returns: array, shape (n_samples,)
        """
        if self.means_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        X = np.asarray(X, dtype=float)
        single = False
        if X.ndim == 1:
            X = X.reshape(1, -1)
            single = True

        self._check_covariance_type()

        log_prob_norm, _, _ = self._estimate_log_prob_resp(X)
        if single:
            return float(log_prob_norm[0])
        return log_prob_norm

    def anomaly_score(self, X):
        """
        Negative log-likelihood = anomaly score.
        Higher score => more anomalous.
        """
        return -self.score_samples(X)

    def predict(self, X):
        """
        Predict 'ok', 'novel', or 'outlier' based on thresholds.
        """
        if self.threshold_ok_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        scores = np.asarray(self.anomaly_score(X))
        single = False
        if scores.ndim == 0:
            scores = scores[None]
            single = True

        labels = np.empty(scores.shape[0], dtype=object)
        labels[scores <= self.threshold_ok_] = "ok"
        labels[(scores > self.threshold_ok_) & (scores <= self.threshold_novel_)] = "novel"
        labels[scores > self.threshold_novel_] = "outlier"

        return labels[0] if single else labels

    def predict_numeric(self, X):
        """
        Numeric labels: 1 = ok, 0 = novel, -1 = outlier
        """
        pred = self.predict(X)
        if isinstance(pred, str):
            return 1 if pred == "ok" else (0 if pred == "novel" else -1)

        out = np.zeros(len(pred), dtype=int)
        out[pred == "ok"] = 1
        out[pred == "novel"] = 0
        out[pred == "outlier"] = -1
        return out

    # ------------------------------------------------------------------
    # Export / import for C++ deployment
    # ------------------------------------------------------------------
    def to_dict(self):
        """Export model parameters to a plain dict (JSON serializable)."""
        if self.means_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return {
            "n_components": int(self.n_components),
            "covariance_type": self.covariance_type,
            "threshold_ok": float(self.threshold_ok),
            "threshold_novel": float(self.threshold_novel),
            "n_features": int(self.n_features_),
            "weights": self.weights_.tolist(),
            "means": self.means_.tolist(),
            "covariances": self.covariances_.tolist(),
            "threshold_ok_value": float(self.threshold_ok_),
            "threshold_novel_value": float(self.threshold_novel_),
            "converged": bool(self.converged_),
            "n_iter": int(self.n_iter_),
        }

    def to_json(self, json_path):
        """Save model parameters to JSON file."""
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, params):
        """Load model from dict produced by to_dict()."""
        obj = cls(
            n_components=params["n_components"],
            covariance_type=params["covariance_type"],
            threshold_ok=params["threshold_ok"],
            threshold_novel=params["threshold_novel"],
        )
        obj.n_features_ = params["n_features"]
        obj.weights_ = np.array(params["weights"], dtype=float)
        obj.means_ = np.array(params["means"], dtype=float)
        obj.covariances_ = np.array(params["covariances"], dtype=float)
        obj.threshold_ok_ = float(params["threshold_ok_value"])
        obj.threshold_novel_ = float(params["threshold_novel_value"])
        obj.converged_ = bool(params.get("converged", True))
        obj.n_iter_ = int(params.get("n_iter", 0))
        return obj

    @classmethod
    def from_json(cls, json_path):
        """Load model parameters from JSON file."""
        with open(json_path, "r") as f:
            params = json.load(f)
        return cls.from_dict(params)

    def __repr__(self):
        if self.means_ is None:
            return f"CustomGMMDetector(n_components={self.n_components}, not fitted)"
        return (
            f"CustomGMMDetector(n_components={self.n_components}, "
            f"n_features={self.n_features_}, converged={self.converged_})"
        )
