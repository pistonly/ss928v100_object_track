//
// Created by oem on 24-3-25.
//


#pragma once
#include "dataType.hpp"


class KalmanFilter {
public:
    KalmanFilter();
    KAL_DATA initiate(const TRACKTBOX& measurement);
    KAL_DATA predict(KAL_MEAN& mean,KAL_COVA& covariance);
    KAL_HDATA project(const KAL_MEAN& mean, const KAL_COVA& covariance);
    KAL_DATA update(const KAL_MEAN& mean,
        const KAL_COVA& covariance,
        const TRACKTBOX& measurement);


private:
    Eigen::Matrix<float,8,8, Eigen::RowMajor> _motion_mat;
    Eigen::Matrix<float,4,8,Eigen::RowMajor> _update_mat;
    float _std_weight_position;
    float _std_weight_velocity;
    int ndim;
    int dt;



};
