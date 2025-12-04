#pragma once

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <limits>

namespace novelty {

using json = nlohmann::json;
using Eigen::VectorXd;
using Eigen::MatrixXd;

// ============================================================================
// StandardScaler (inference only)
// Expected Python JSON (custom_scaler.to_dict()):
// {
//   "n_features": <int>,
//   "mean": [...],
//   "scale": [...]
// }
// ============================================================================
class StandardScaler {
public:
    StandardScaler() = default;

    void load_from_json_file(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open scaler JSON: " + path);
        }
        json j;
        f >> j;
        load_from_json(j);
    }

    void load_from_json(const json& j) {
        // You may need to adjust these keys if your custom_scaler differs
        std::vector<double> mean_vec = j.at("mean").get<std::vector<double>>();
        std::vector<double> scale_vec = j.at("scale").get<std::vector<double>>();

        n_features_ = static_cast<int>(mean_vec.size());
        if (scale_vec.size() != static_cast<size_t>(n_features_)) {
            throw std::runtime_error("StandardScaler: mean/scale size mismatch");
        }

        mean_.resize(n_features_);
        scale_.resize(n_features_);
        for (int i = 0; i < n_features_; ++i) {
            mean_(i) = mean_vec[i];
            scale_(i) = scale_vec[i];
        }
    }

    int n_features() const { return n_features_; }

    // Transform single sample: x_scaled = (x - mean) / scale
    VectorXd transform(const VectorXd& x) const {
        if (x.size() != n_features_) {
            throw std::runtime_error("StandardScaler::transform: dimension mismatch");
        }
        VectorXd y = x;
        for (int i = 0; i < n_features_; ++i) {
            y(i) = (y(i) - mean_(i)) / scale_(i);
        }
        return y;
    }

private:
    int n_features_{0};
    VectorXd mean_;
    VectorXd scale_;
};

// ============================================================================
// CustomNoveltyDetector (Mahalanobis-based)
// Matches your Python CustomNoveltyDetector JSON:
// {
//   "contamination": <double>,
//   "mean": [...],
//   "cov": [[...], ...],
//   "inv_cov": [[...], ...],
//   "threshold": <double>,
//   "n_features": <int>
// }
// ============================================================================
class CustomMahalanobisDetector {
public:
    CustomMahalanobisDetector() = default;

    void load_from_json_file(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open Mahalanobis JSON: " + path);
        }
        json j;
        f >> j;
        load_from_json(j);
    }

    void load_from_json(const json& j) {
        contamination_ = j.at("contamination").get<double>();
        threshold_ = j.at("threshold").get<double>();
        n_features_ = j.at("n_features").get<int>();

        // mean
        {
            std::vector<double> mean_vec = j.at("mean").get<std::vector<double>>();
            if (static_cast<int>(mean_vec.size()) != n_features_) {
                throw std::runtime_error("MahalanobisDetector: mean size mismatch");
            }
            mean_.resize(n_features_);
            for (int i = 0; i < n_features_; ++i) {
                mean_(i) = mean_vec[i];
            }
        }

        // cov (not strictly needed for inference, but load anyway)
        {
            cov_.resize(n_features_, n_features_);
            auto cov_j = j.at("cov");
            if (static_cast<int>(cov_j.size()) != n_features_) {
                throw std::runtime_error("MahalanobisDetector: cov rows mismatch");
            }
            for (int r = 0; r < n_features_; ++r) {
                auto row = cov_j.at(r);
                if (static_cast<int>(row.size()) != n_features_) {
                    throw std::runtime_error("MahalanobisDetector: cov cols mismatch");
                }
                for (int c = 0; c < n_features_; ++c) {
                    cov_(r, c) = row.at(c).get<double>();
                }
            }
        }

        // inv_cov
        {
            inv_cov_.resize(n_features_, n_features_);
            auto inv_j = j.at("inv_cov");
            if (static_cast<int>(inv_j.size()) != n_features_) {
                throw std::runtime_error("MahalanobisDetector: inv_cov rows mismatch");
            }
            for (int r = 0; r < n_features_; ++r) {
                auto row = inv_j.at(r);
                if (static_cast<int>(row.size()) != n_features_) {
                    throw std::runtime_error("MahalanobisDetector: inv_cov cols mismatch");
                }
                for (int c = 0; c < n_features_; ++c) {
                    inv_cov_(r, c) = row.at(c).get<double>();
                }
            }
        }
    }

    int n_features() const { return n_features_; }

    // Mahalanobis distance
    double mahalanobis(const VectorXd& x) const {
        if (x.size() != n_features_) {
            throw std::runtime_error("MahalanobisDetector::mahalanobis: dimension mismatch");
        }
        VectorXd diff = x - mean_;
        double d2 = diff.transpose() * inv_cov_ * diff;
        if (d2 < 0.0) d2 = 0.0; // numerical guard
        return std::sqrt(d2);
    }

    // decision_function: negative distance (higher = more normal)
    double decision_function(const VectorXd& x) const {
        return -mahalanobis(x);
    }

    // anomaly score: negative decision_function (or distance itself)
    double anomaly_score(const VectorXd& x) const {
        // You can choose to use distance directly;
        // Python uses negative distance as "score_samples".
        return mahalanobis(x);
    }

    // predict: 1 = inlier, -1 = outlier
    int predict(const VectorXd& x) const {
        double d = mahalanobis(x);
        return (d <= threshold_) ? 1 : -1;
    }

private:
    double contamination_{0.1};
    double threshold_{0.0};
    int n_features_{0};
    VectorXd mean_;
    MatrixXd cov_;
    MatrixXd inv_cov_;
};

// ============================================================================
// CustomGMMDetector (inference-only)
// Matches your Python CustomGMMDetector JSON:
// {
//   "n_components": <int>,
//   "covariance_type": "full"/"diag"/"spherical",
//   "threshold_ok": <double>,
//   "threshold_novel": <double>,
//   "n_features": <int>,
//   "weights": [...],
//   "means": [[...], ...],
//   "covariances": ... (shape depends on covariance_type),
//   "threshold_ok_value": <double>,
//   "threshold_novel_value": <double>,
//   "converged": <bool>,
//   "n_iter": <int>
// }
// ============================================================================
class CustomGMMDetectorCpp {
public:
    enum class CovarianceType {
        FULL,
        DIAG,
        SPHERICAL
    };

    CustomGMMDetectorCpp() = default;

    void load_from_json_file(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open GMM JSON: " + path);
        }
        json j;
        f >> j;
        load_from_json(j);
    }

    void load_from_json(const json& j) {
        n_components_ = j.at("n_components").get<int>();
        std::string cov_type_str = j.at("covariance_type").get<std::string>();
        if (cov_type_str == "full") {
            cov_type_ = CovarianceType::FULL;
        } else if (cov_type_str == "diag") {
            cov_type_ = CovarianceType::DIAG;
        } else if (cov_type_str == "spherical") {
            cov_type_ = CovarianceType::SPHERICAL;
        } else {
            throw std::runtime_error("CustomGMMDetectorCpp: unsupported covariance_type: " + cov_type_str);
        }

        threshold_ok_quantile_ = j.at("threshold_ok").get<double>();
        threshold_novel_quantile_ = j.at("threshold_novel").get<double>();
        n_features_ = j.at("n_features").get<int>();
        threshold_ok_ = j.at("threshold_ok_value").get<double>();
        threshold_novel_ = j.at("threshold_novel_value").get<double>();
        converged_ = j.value("converged", true);
        n_iter_ = j.value("n_iter", 0);

        // weights
        {
            std::vector<double> w_vec = j.at("weights").get<std::vector<double>>();
            if (static_cast<int>(w_vec.size()) != n_components_) {
                throw std::runtime_error("GMM: weights size mismatch");
            }
            weights_.resize(n_components_);
            for (int k = 0; k < n_components_; ++k)
                weights_(k) = w_vec[k];
        }

        // means (K x D)
        {
            auto means_j = j.at("means");
            if (static_cast<int>(means_j.size()) != n_components_) {
                throw std::runtime_error("GMM: means rows mismatch");
            }
            means_.resize(n_components_, n_features_);
            for (int k = 0; k < n_components_; ++k) {
                auto row = means_j.at(k);
                if (static_cast<int>(row.size()) != n_features_) {
                    throw std::runtime_error("GMM: means cols mismatch");
                }
                for (int d = 0; d < n_features_; ++d) {
                    means_(k, d) = row.at(d).get<double>();
                }
            }
        }

        // covariances
        auto covs_j = j.at("covariances");
        if (cov_type_ == CovarianceType::FULL) {
            // covs_j is K x D x D
            if (static_cast<int>(covs_j.size()) != n_components_) {
                throw std::runtime_error("GMM: full cov rows mismatch");
            }
            cov_full_.resize(n_components_);
            inv_cov_full_.resize(n_components_);
            logdet_full_.resize(n_components_);
            for (int k = 0; k < n_components_; ++k) {
                auto covk_j = covs_j.at(k);
                if (static_cast<int>(covk_j.size()) != n_features_) {
                    throw std::runtime_error("GMM: full cov[k] rows mismatch");
                }
                MatrixXd cov_k(n_features_, n_features_);
                for (int r = 0; r < n_features_; ++r) {
                    auto row = covk_j.at(r);
                    if (static_cast<int>(row.size()) != n_features_) {
                        throw std::runtime_error("GMM: full cov[k] cols mismatch");
                    }
                    for (int c = 0; c < n_features_; ++c) {
                        cov_k(r, c) = row.at(c).get<double>();
                    }
                }
                cov_full_[k] = cov_k;

                // Precompute inv_cov and logdet using Cholesky (assuming SPD)
                Eigen::LLT<MatrixXd> llt(cov_k);
                if (llt.info() != Eigen::Success) {
                    // Try adding a bit of regularization
                    MatrixXd cov_reg = cov_k + 1e-9 * MatrixXd::Identity(n_features_, n_features_);
                    llt.compute(cov_reg);
                    if (llt.info() != Eigen::Success) {
                        throw std::runtime_error("GMM: LLT failed for covariance matrix");
                    }
                }
                MatrixXd L = llt.matrixL();
                double logdet = 0.0;
                for (int d = 0; d < n_features_; ++d) {
                    double diag = L(d, d);
                    if (diag <= 0.0) {
                        throw std::runtime_error("GMM: non-positive diagonal in Cholesky");
                    }
                    logdet += std::log(diag);
                }
                logdet *= 2.0;
                logdet_full_[k] = logdet;
                inv_cov_full_[k] = llt.solve(MatrixXd::Identity(n_features_, n_features_));
            }
        } else if (cov_type_ == CovarianceType::DIAG) {
            // covs_j is K x D
            if (static_cast<int>(covs_j.size()) != n_components_) {
                throw std::runtime_error("GMM: diag cov rows mismatch");
            }
            cov_diag_.resize(n_components_, n_features_);
            for (int k = 0; k < n_components_; ++k) {
                auto row = covs_j.at(k);
                if (static_cast<int>(row.size()) != n_features_) {
                    throw std::runtime_error("GMM: diag cov cols mismatch");
                }
                for (int d = 0; d < n_features_; ++d) {
                    cov_diag_(k, d) = row.at(d).get<double>();
                }
            }
        } else if (cov_type_ == CovarianceType::SPHERICAL) {
            // covs_j is K
            if (static_cast<int>(covs_j.size()) != n_components_) {
                throw std::runtime_error("GMM: spherical cov size mismatch");
            }
            cov_spherical_.resize(n_components_);
            for (int k = 0; k < n_components_; ++k) {
                cov_spherical_(k) = covs_j.at(k).get<double>();
            }
        }
    }

    int n_features() const { return n_features_; }
    int n_components() const { return n_components_; }

    // log p(x | component k)
    double log_gaussian_component(int k, const VectorXd& x) const {
        if (x.size() != n_features_) {
            throw std::runtime_error("GMM::log_gaussian_component: dimension mismatch");
        }
        VectorXd diff = x - means_.row(k).transpose();

        const double log2pi = std::log(2.0 * M_PI);

        if (cov_type_ == CovarianceType::FULL) {
            const MatrixXd& inv_cov = inv_cov_full_[k];
            double logdet = logdet_full_[k];
            double m = diff.transpose() * inv_cov * diff; // Mahalanobis
            return -0.5 * (n_features_ * log2pi + logdet + m);
        } else if (cov_type_ == CovarianceType::DIAG) {
            Eigen::RowVectorXd var_row = cov_diag_.row(k);
            // log |Sigma|
            double logdet = var_row.array().log().sum();
            // Mahalanobis
            Eigen::ArrayXd diff2 = diff.array().square();
            double m = (diff2 / var_row.array()).sum();
            return -0.5 * (n_features_ * log2pi + logdet + m);
        } else { // SPHERICAL
            double var = cov_spherical_(k);
            double logdet = n_features_ * std::log(var);
            double m = diff.squaredNorm() / var;
            return -0.5 * (n_features_ * log2pi + logdet + m);
        }
    }

    // log p(x) = log sum_k [ pi_k * N(x | mu_k, Sigma_k) ]
    double score_log(const VectorXd& x) const {
        if (x.size() != n_features_) {
            throw std::runtime_error("GMM::score_log: dimension mismatch");
        }

        std::vector<double> log_probs;
        log_probs.reserve(n_components_);

        for (int k = 0; k < n_components_; ++k) {
            double lp = std::log(weights_(k)) + log_gaussian_component(k, x);
            log_probs.push_back(lp);
        }

        // log-sum-exp
        double max_log = -std::numeric_limits<double>::infinity();
        for (double v : log_probs) {
            if (v > max_log) max_log = v;
        }

        double sum_exp = 0.0;
        for (double v : log_probs) {
            sum_exp += std::exp(v - max_log);
        }

        return max_log + std::log(sum_exp);
    }

    // Python's score_samples: log-likelihood
    double score_sample(const VectorXd& x) const {
        return score_log(x);
    }

    // anomaly score: negative log-likelihood
    double anomaly_score(const VectorXd& x) const {
        return -score_log(x);
    }

    // Predict "ok", "novel", "outlier" based on anomaly score thresholds
    std::string predict_label(const VectorXd& x) const {
        double s = anomaly_score(x);
        if (s <= threshold_ok_) {
            return "ok";
        } else if (s <= threshold_novel_) {
            return "novel";
        }
        return "outlier";
    }

private:
    int n_components_{0};
    int n_features_{0};
    CovarianceType cov_type_{CovarianceType::FULL};

    VectorXd weights_;      // (K,)
    MatrixXd means_;        // (K, D)

    // FULL
    std::vector<MatrixXd> cov_full_;
    std::vector<MatrixXd> inv_cov_full_;
    std::vector<double>   logdet_full_;

    // DIAG
    MatrixXd cov_diag_;     // (K, D)

    // SPHERICAL
    VectorXd cov_spherical_; // (K,)

    // thresholds
    double threshold_ok_quantile_{0.95};
    double threshold_novel_quantile_{0.99};
    double threshold_ok_{0.0};
    double threshold_novel_{0.0};

    // meta
    bool converged_{true};
    int  n_iter_{0};
};

} // namespace novelty
