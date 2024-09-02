//
// Created by oem on 24-3-25.
//

#include "kalmanFilter.hpp"



KalmanFilter::KalmanFilter() {
    this->ndim = 4;
    this->dt = 1;
    _motion_mat = Eigen::MatrixXf::Identity(2*ndim,2*ndim);
    for (int i=0;i<ndim;i++) {
        _motion_mat(i,ndim+i) = dt;
    }
    _update_mat = Eigen::MatrixXf::Identity(ndim,2*ndim);

    _std_weight_position = 1. / 6;
    _std_weight_velocity = 1. / 90;
    // _std_weight_position = 1. / 20;
    // _std_weight_velocity = 1. / 160;
}

KAL_DATA KalmanFilter::initiate(const TRACKTBOX &measurement) {

    // 初始化卡尔曼滤波
    TRACKTBOX mean_pos = measurement;
    TRACKTBOX mean_vel;
    for (int i =0; i < ndim; i++) {
        mean_vel(i) = 0;
    }


    KAL_MEAN mean;
    for (int i=0; i < 2* ndim; i++) {
        if (i<ndim) mean(i) = mean_pos(i);
        else mean(i) = mean_vel(i-4);
    }

    KAL_MEAN std_;
    std_(0) = 2 * _std_weight_position * measurement[2];
    std_(1) = 2 * _std_weight_position * measurement[3];
    std_(2) = 2 * _std_weight_position * measurement[2];
    std_(3) = 2 * _std_weight_position * measurement[3];
    std_(4)= 10 * _std_weight_velocity * measurement[2];
    std_(5)= 10 * _std_weight_velocity * measurement[3];
    std_(6)= 10 * _std_weight_velocity * measurement[2];
    std_(7)= 10 * _std_weight_velocity * measurement[3];

    KAL_MEAN tmp = std_.array().square();
    KAL_COVA var = tmp.asDiagonal();

    return std::make_pair(mean,var);
}

KAL_DATA KalmanFilter::predict(KAL_MEAN &mean, KAL_COVA &covariance) {

    TRACKTBOX std_pos;
    std_pos << _std_weight_position * mean(2),
    _std_weight_position * mean(3),
    _std_weight_position * mean(2),
    _std_weight_position * mean(3);

    TRACKTBOX std_vel;
    std_vel << _std_weight_velocity * mean(2),
    _std_weight_velocity * mean(3),
    _std_weight_velocity * mean(2),
    _std_weight_velocity * mean(3);

    KAL_MEAN tmp;
    tmp.block<1,4>(0,0) = std_pos;
    tmp.block<1,4>(0,4) = std_vel;

    tmp = tmp.array().square();
    KAL_COVA motion_cov = tmp.asDiagonal();
    KAL_MEAN mean1 = _motion_mat * mean.transpose();
    KAL_COVA covariance1 = _motion_mat * covariance *(_motion_mat.transpose());
    covariance1 += motion_cov;
    // mean = mean1;
    // covariance = covariance1;
    return std::make_pair(mean1,covariance1);

}


KAL_HDATA KalmanFilter::project(const KAL_MEAN &mean, const KAL_COVA &covariance) {

    TRACKTBOX std_;
    std_ << _std_weight_position * mean(2),
    _std_weight_position * mean(3),
    _std_weight_position * mean(2),
    _std_weight_position * mean(3);

    KAL_HMEAN mean1 = _update_mat * mean.transpose();
    KAL_HCOVA covariance1 = _update_mat * covariance * (_update_mat.transpose());

    Eigen::Matrix<float,4,4>diag = std_.asDiagonal();
    diag = diag.array().square().matrix();
    covariance1 += diag;
    return std::make_pair(mean1,covariance1);

}


KAL_DATA KalmanFilter::update(
    const KAL_MEAN &mean,
    const KAL_COVA &covariance,
    const TRACKTBOX &measurement) {

    KAL_HDATA pa = project(mean, covariance);
    KAL_HMEAN projected_mean = pa.first;
    KAL_HCOVA projected_cov = pa.second;


    Eigen::Matrix<float, 4, 8> B = (covariance * (_update_mat.transpose())).transpose();
    Eigen::Matrix<float, 8, 4> kalman_gain = (projected_cov.llt().solve(B)).transpose(); // eg.8x4
    Eigen::Matrix<float, 1, 4> innovation = measurement - projected_mean; //eg.1x4
    auto tmp = innovation * (kalman_gain.transpose());
    KAL_MEAN new_mean = (mean.array() + tmp.array()).matrix();
    KAL_COVA new_covariance = covariance - kalman_gain * projected_cov*(kalman_gain.transpose());
    return std::make_pair(new_mean, new_covariance);
}





