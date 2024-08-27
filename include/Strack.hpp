//
// Created by oem on 24-3-25.
//

#pragma once
#include <unordered_map>

# include "kalmanFilter.hpp"


class STrack {
public:
    STrack(std::vector<float> tlwh_);
    ~STrack();
    void activate(KalmanFilter &kalman_filter);
    void static multi_predict(std::unordered_map<int,STrack> &stracks,KalmanFilter &kalman_filter);
    void update(std::vector<float> tlwh_);

    std::vector<float> tlwh_to_xywh(std::vector<float> tlwn_tmp);
    STrack() = default;



public:

    std::vector<float> _tlwh;
    std::vector<float> tlwh;
    std::vector<float> tlbr;
    KAL_MEAN mean;
    KAL_COVA covariance;

private:
    KalmanFilter kalman_filter;
};
