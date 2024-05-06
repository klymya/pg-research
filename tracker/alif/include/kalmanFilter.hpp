#pragma once

#include <memory>
#include "dataType.hpp"

namespace byte_kalman
{
	class KalmanFilter
	{
	public:
		static const double chi2inv95[10];
		KalmanFilter();
		void initiate(const DETECTBOX& measurement, KAL_MEAN& mean, KAL_COVA& var);
		void predict(KAL_MEAN& mean, KAL_COVA& covariance);
		void project(const KAL_MEAN& mean, const KAL_COVA& covariance, KAL_HMEAN& projected_mean, KAL_HCOVA& projected_cov);
		void update(const KAL_MEAN& mean,
			const KAL_COVA& covariance,
			const DETECTBOX& measurement,
			KAL_MEAN& new_mean,
			KAL_COVA& new_var);

		void gating_distance(
			const KAL_MEAN& mean,
			const KAL_COVA& covariance,
			const std::vector<DETECTBOX>& measurements,
			Eigen::Matrix<float, 1, -1>& square_maha,
			bool only_position = false);

	private:
		std::shared_ptr<KAL_COVA> _motion_mat;
		std::shared_ptr<KAL_OBSERVATION> _update_mat;

		float _std_weight_position;
		float _std_weight_velocity;
	};
}