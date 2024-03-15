#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

const int KALMAN_N_DIM = 6;
const int KALMAN_N_VAR = KALMAN_N_DIM * 2;

typedef Eigen::Matrix<float, 1, KALMAN_N_DIM, Eigen::RowMajor> DETECTBOX;
typedef Eigen::Matrix<float, -1, KALMAN_N_DIM, Eigen::RowMajor> DETECTBOXSS;
typedef Eigen::Matrix<float, 1, 128, Eigen::RowMajor> FEATURE;
typedef Eigen::Matrix<float, Eigen::Dynamic, 128, Eigen::RowMajor> FEATURESS;
//typedef std::vector<FEATURE> FEATURESS;

//Kalmanfilter
//typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_FILTER;
typedef Eigen::Matrix<float, 1, KALMAN_N_VAR, Eigen::RowMajor> KAL_MEAN;
typedef Eigen::Matrix<float, KALMAN_N_VAR, KALMAN_N_VAR, Eigen::RowMajor> KAL_COVA;
typedef Eigen::Matrix<float, 1, KALMAN_N_DIM, Eigen::RowMajor> KAL_HMEAN;
typedef Eigen::Matrix<float, KALMAN_N_DIM, KALMAN_N_DIM, Eigen::RowMajor> KAL_HCOVA;
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

//main
using RESULT_DATA = std::pair<int, DETECTBOX>;

//tracker:
using TRACKER_DATA = std::pair<int, FEATURESS>;
using MATCH_DATA = std::pair<int, int>;
typedef struct t {
	std::vector<MATCH_DATA> matches;
	std::vector<int> unmatched_tracks;
	std::vector<int> unmatched_detections;
}TRACHER_MATCHD;

//linear_assignment:
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> DYNAMICM;