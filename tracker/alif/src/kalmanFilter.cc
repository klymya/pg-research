#include "kalmanFilter.hpp"
#include <Eigen/Cholesky>
#include <iostream>

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
		_motion_mat = std::shared_ptr<KAL_COVA>(new KAL_COVA());
		_update_mat = std::shared_ptr<KAL_OBSERVATION>(new KAL_OBSERVATION());

		*_motion_mat = Eigen::MatrixXf::Identity(KALMAN_N_VAR, KALMAN_N_VAR);
		for (int i = 0; i < KALMAN_N_DIM; i++) {
			(*_motion_mat)(i, KALMAN_N_DIM + i) = dt;
		}
		*_update_mat = Eigen::MatrixXf::Identity(KALMAN_N_DIM, KALMAN_N_VAR);

		// this->_std_weight_position = 1. / 20;
		// this->_std_weight_velocity = 1. / 160;
		this->_std_weight_position = 1. / 10;
		this->_std_weight_velocity = 1. / 80;
	}

	void KalmanFilter::initiate(const DETECTBOX &measurement, KAL_MEAN &mean, KAL_COVA &var)
	{
		std::unique_ptr<DETECTBOX> mean_vel(new DETECTBOX());
		for (int i = 0; i < KALMAN_N_DIM; i++) (*mean_vel)(i) = 0;

		for (int i = 0; i < KALMAN_N_VAR; i++) {
			if (i < KALMAN_N_DIM) mean(i) = measurement(i);
			else mean(i) = (*mean_vel)(i - KALMAN_N_DIM);
		}

		std::unique_ptr<KAL_MEAN> std1(new KAL_MEAN());
		for (int i = 0; i < KALMAN_N_DIM; i++) {
			(*std1)(i) = 2 * _std_weight_position * measurement[3];
		}
		for (int i = 0; i < KALMAN_N_DIM; i++) {
			(*std1)(i + KALMAN_N_DIM) = 10 * _std_weight_velocity * measurement[3];
		}

		var = ((KAL_MEAN)std1->array().square()).asDiagonal();
	}

	void KalmanFilter::predict(KAL_MEAN &mean, KAL_COVA &covariance)
	{
		//revise the data;
		std::unique_ptr<DETECTBOX> std_pos(new DETECTBOX());
		std::unique_ptr<DETECTBOX> std_vel(new DETECTBOX());
		std::unique_ptr<KAL_MEAN> tmp(new KAL_MEAN());

		*std_pos << _std_weight_position * mean(3),
			_std_weight_position * mean(3),
			_std_weight_position * mean(3),
			_std_weight_position * mean(3),
			_std_weight_position * mean(3),
			_std_weight_position * mean(3);

		*std_vel << _std_weight_velocity * mean(3),
			_std_weight_velocity * mean(3),
			_std_weight_velocity * mean(3),
			_std_weight_velocity * mean(3),
			_std_weight_velocity * mean(3),
			_std_weight_velocity * mean(3);

		tmp->block<1, KALMAN_N_DIM>(0, 0) = *std_pos;
		tmp->block<1, KALMAN_N_DIM>(0, KALMAN_N_DIM) = *std_vel;
		*tmp = tmp->array().square();

		mean = *this->_motion_mat * mean.transpose();
		covariance = *this->_motion_mat * covariance *(_motion_mat->transpose()) + (KAL_COVA)tmp->asDiagonal();
	}

	void KalmanFilter::project(const KAL_MEAN &mean, const KAL_COVA &covariance, KAL_HMEAN &projected_mean, KAL_HCOVA &projected_cov)
	{
		std::unique_ptr<DETECTBOX> std1(new DETECTBOX());
		std::unique_ptr<KAL_HCOVA> diag(new KAL_HCOVA());

		*std1 << _std_weight_position * mean(3), _std_weight_position * mean(3),
			_std_weight_position * mean(3), _std_weight_position * mean(3),
			_std_weight_position * mean(3), _std_weight_position * mean(3);

		*diag = std1->asDiagonal();
		*diag = diag->array().square().matrix();

		projected_mean = *_update_mat * mean.transpose();
		projected_cov = *_update_mat * covariance * (_update_mat->transpose());
		projected_cov += *diag;
	}

	void KalmanFilter::update(
		const KAL_MEAN &mean,
		const KAL_COVA &covariance,
		const DETECTBOX &measurement,
		KAL_MEAN &new_mean,
		KAL_COVA &new_covariance)
	{	
		std::unique_ptr<KAL_HMEAN> projected_mean(new KAL_HMEAN());
		std::unique_ptr<KAL_HCOVA> projected_cov(new KAL_HCOVA());
		std::unique_ptr<KAL_OBSERVATION> B(new KAL_OBSERVATION());
		std::unique_ptr<Eigen::Matrix<float, KALMAN_N_VAR, KALMAN_N_DIM>> kalman_gain(new Eigen::Matrix<float, KALMAN_N_VAR, KALMAN_N_DIM>());
		std::unique_ptr<Eigen::Matrix<float, 1, KALMAN_N_DIM>> innovation(new Eigen::Matrix<float, 1, KALMAN_N_DIM>());
		
		project(mean, covariance, *projected_mean, *projected_cov);

		*B = (covariance * (_update_mat->transpose())).transpose();
		*kalman_gain = (projected_cov->llt().solve(*B)).transpose(); // eg.8x4
		*innovation = measurement - (*projected_mean); //eg.1x4

		new_mean = (mean.array() + ((*innovation) * (kalman_gain->transpose())).array()).matrix();
		new_covariance = covariance - (*kalman_gain) * (*projected_cov)*(kalman_gain->transpose());
	}

	void KalmanFilter::gating_distance(
		const KAL_MEAN &mean,
		const KAL_COVA &covariance,
		const std::vector<DETECTBOX> &measurements,
		Eigen::Matrix<float, 1, -1> &square_maha,
		bool only_position)
	{
		std::unique_ptr<KAL_HMEAN> mean1(new KAL_HMEAN());
		std::unique_ptr<KAL_HCOVA> covariance1(new KAL_HCOVA());
		project(mean, covariance, *mean1, *covariance1);
		if (only_position) {
			printf("not implement!");
			exit(0);
		}

		//    Eigen::Matrix<float, -1, 4, Eigen::RowMajor> d(size, 4);
		std::unique_ptr<DETECTBOXSS> d(new DETECTBOXSS(measurements.size(), KALMAN_N_DIM));
		std::unique_ptr<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>> factor(new Eigen::Matrix<float, -1, -1, Eigen::RowMajor>);
		std::unique_ptr<Eigen::Matrix<float, -1, -1>> z(new Eigen::Matrix<float, -1, -1>);

		int pos = 0;
		for (DETECTBOX box : measurements) {
			d->row(pos++) = box - (*mean1);
		}
		*factor = covariance1->llt().matrixL();
		*z = factor->triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(*d).transpose();
		square_maha = ((z->array())*(z->array())).matrix().colwise().sum();
	}
}