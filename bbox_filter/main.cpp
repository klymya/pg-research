#include "novelty_detectors.hpp"
#include <iostream>

int main() {
    try {
        // 1. Load scaler and GMM (paths must match your Python script)
        novelty::StandardScaler scaler;
        scaler.load_from_json_file("gmm_detector_scaler.json");

        novelty::CustomGMMDetectorCpp gmm;
        gmm.load_from_json_file("gmm_detector_params.json");

        // 2. Build raw feature vector [aspect_ratio, bbox_area, dist_from_center, cam_height]
        Eigen::VectorXd x_raw(4);
        x_raw << 1.0, 0.02, 0.15, 310.0;

        // 3. Scale features
        Eigen::VectorXd x_scaled = scaler.transform(x_raw);

        // 4. Run detector
        double logp = gmm.score_sample(x_scaled);
        double score = gmm.anomaly_score(x_scaled);
        std::string label = gmm.predict_label(x_scaled);

        std::cout << "log p(x)        = " << logp   << "\n";
        std::cout << "anomaly score   = " << score << "\n";
        std::cout << "predicted label = " << label << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
