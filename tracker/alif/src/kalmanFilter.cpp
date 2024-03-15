#include "kalmanFilter.h"
#include <Eigen/Cholesky>

namespace byte_kalman
{
	const double KalmanFilter::chi2inv95[10] = {
	0,
	3.8415,
	5.9915,
	7.8147,
	9.4877,
	11.070,
	12.592,
	14.067,
	15.507,
	16.919
	};
	KalmanFilter::KalmanFilter()
	{
		double dt = 1.;

		_motion_mat = Eigen::MatrixXf::Identity(KALMAN_N_VAR, KALMAN_N_VAR);
		for (int i = 0; i < KALMAN_N_DIM; i++) {
			_motion_mat(i, KALMAN_N_DIM + i) = dt;
		}
		_update_mat = Eigen::MatrixXf::Identity(KALMAN_N_DIM, KALMAN_N_VAR);

		// this->_std_weight_position = 1. / 20;
		// this->_std_weight_velocity = 1. / 160;
		this->_std_weight_position = 1. / 10;
		this->_std_weight_velocity = 1. / 80;
	}

	KAL_DATA KalmanFilter::initiate(const DETECTBOX &measurement)
	{
		DETECTBOX mean_pos = measurement;
		DETECTBOX mean_vel;
		for (int i = 0; i < KALMAN_N_DIM; i++) mean_vel(i) = 0;

		KAL_MEAN mean;
		for (int i = 0; i < KALMAN_N_VAR; i++) {
			if (i < KALMAN_N_DIM) mean(i) = mean_pos(i);
			else mean(i) = mean_vel(i - KALMAN_N_DIM);
		}

		KAL_MEAN std;
		for (int i = 0; i < KALMAN_N_DIM; i++) {
			std(i) = 2 * _std_weight_position * measurement[3];
		}
		for (int i = 0; i < KALMAN_N_DIM; i++) {
			std(i + KALMAN_N_DIM) = 10 * _std_weight_velocity * measurement[3];
		}

		KAL_MEAN tmp = std.array().square();
		KAL_COVA var = tmp.asDiagonal();
		return std::make_pair(mean, var);
	}

	void KalmanFilter::predict(KAL_MEAN &mean, KAL_COVA &covariance)
	{
		//revise the data;
		DETECTBOX std_pos;
		std_pos << _std_weight_position * mean(3),
			_std_weight_position * mean(3),
			_std_weight_position * mean(3),
			_std_weight_position * mean(3),
			_std_weight_position * mean(3),
			_std_weight_position * mean(3);
		DETECTBOX std_vel;
		std_vel << _std_weight_velocity * mean(3),
			_std_weight_velocity * mean(3),
			_std_weight_velocity * mean(3),
			_std_weight_velocity * mean(3),
			_std_weight_velocity * mean(3),
			_std_weight_velocity * mean(3);
		KAL_MEAN tmp;
		tmp.block<1, KALMAN_N_DIM>(0, 0) = std_pos;
		tmp.block<1, KALMAN_N_DIM>(0, KALMAN_N_DIM) = std_vel;
		tmp = tmp.array().square();
		KAL_COVA motion_cov = tmp.asDiagonal();
		KAL_MEAN mean1 = this->_motion_mat * mean.transpose();
		KAL_COVA covariance1 = this->_motion_mat * covariance *(_motion_mat.transpose());
		covariance1 += motion_cov;

		mean = mean1;
		covariance = covariance1;
	}

	KAL_HDATA KalmanFilter::project(const KAL_MEAN &mean, const KAL_COVA &covariance)
	{
		DETECTBOX std;
		std << _std_weight_position * mean(3), _std_weight_position * mean(3),
			_std_weight_position * mean(3), _std_weight_position * mean(3),
			_std_weight_position * mean(3), _std_weight_position * mean(3);
		KAL_HMEAN mean1 = _update_mat * mean.transpose();
		KAL_HCOVA covariance1 = _update_mat * covariance * (_update_mat.transpose());
		Eigen::Matrix<float, KALMAN_N_DIM, KALMAN_N_DIM> diag = std.asDiagonal();
		diag = diag.array().square().matrix();
		covariance1 += diag;
		//    covariance1.diagonal() << diag;
		return std::make_pair(mean1, covariance1);
	}

	KAL_DATA
		KalmanFilter::update(
			const KAL_MEAN &mean,
			const KAL_COVA &covariance,
			const DETECTBOX &measurement)
	{
		KAL_HDATA pa = project(mean, covariance);
		KAL_HMEAN projected_mean = pa.first;
		KAL_HCOVA projected_cov = pa.second;

		//chol_factor, lower =
		//scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
		//kalmain_gain =
		//scipy.linalg.cho_solve((cho_factor, lower),
		//np.dot(covariance, self._upadte_mat.T).T,
		//check_finite=False).T
		Eigen::Matrix<float, KALMAN_N_DIM, KALMAN_N_VAR> B = (covariance * (_update_mat.transpose())).transpose();
		Eigen::Matrix<float, KALMAN_N_VAR, KALMAN_N_DIM> kalman_gain = (projected_cov.llt().solve(B)).transpose(); // eg.8x4
		Eigen::Matrix<float, 1, KALMAN_N_DIM> innovation = measurement - projected_mean; //eg.1x4
		auto tmp = innovation * (kalman_gain.transpose());
		KAL_MEAN new_mean = (mean.array() + tmp.array()).matrix();
		KAL_COVA new_covariance = covariance - kalman_gain * projected_cov*(kalman_gain.transpose());
		return std::make_pair(new_mean, new_covariance);
	}

	Eigen::Matrix<float, 1, -1>
		KalmanFilter::gating_distance(
			const KAL_MEAN &mean,
			const KAL_COVA &covariance,
			const std::vector<DETECTBOX> &measurements,
			bool only_position)
	{
		KAL_HDATA pa = this->project(mean, covariance);
		if (only_position) {
			printf("not implement!");
			exit(0);
		}
		KAL_HMEAN mean1 = pa.first;
		KAL_HCOVA covariance1 = pa.second;

		//    Eigen::Matrix<float, -1, 4, Eigen::RowMajor> d(size, 4);
		DETECTBOXSS d(measurements.size(), KALMAN_N_DIM);
		int pos = 0;
		for (DETECTBOX box : measurements) {
			d.row(pos++) = box - mean1;
		}
		Eigen::Matrix<float, -1, -1, Eigen::RowMajor> factor = covariance1.llt().matrixL();
		Eigen::Matrix<float, -1, -1> z = factor.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(d).transpose();
		auto zz = ((z.array())*(z.array())).matrix();
		auto square_maha = zz.colwise().sum();
		return square_maha;
	}
}