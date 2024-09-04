#include "Strack.hpp"
#include <unordered_map>

STrack::STrack(std::vector<float> tlwh_, bool using_kal) :mb_using_kal(using_kal){
    _tlwh.resize(4);
    _tlwh.assign(tlwh_.begin(), tlwh_.end());
    if (using_kal)
      activate();
}

STrack::~STrack() {
}

// void STrack::activate(KalmanFilter &kalman_filter) {
//     this->kalman_filter = kalman_filter;
//     std::vector<float> xywh = tlwh_to_xywh(_tlwh);
//     TRACKTBOX xywh_box;
//     xywh_box[0] = xywh[0];
//     xywh_box[1] = xywh[1];
//     xywh_box[2] = xywh[2];
//     xywh_box[3] = xywh[3];
//     auto mc = this->kalman_filter.initiate(xywh_box);
//     this->mean = mc.first;
//     this->covariance = mc.second;
// }

void STrack::activate() {
  if (mb_using_kal){
    std::vector<float> xywh = tlwh_to_xywh(_tlwh);
    TRACKTBOX xywh_box;
    xywh_box[0] = xywh[0];
    xywh_box[1] = xywh[1];
    xywh_box[2] = xywh[2];
    xywh_box[3] = xywh[3];
    auto mc = this->kalman_filter.initiate(xywh_box);
    this->mean = mc.first;
    this->covariance = mc.second;
  }
}

void STrack::update(std::vector<float> tlwh_) {
  if (mb_using_kal) {
    std::vector<float> xywh = tlwh_to_xywh(tlwh_);
    TRACKTBOX xywh_box;
    xywh_box[0] = xywh[0];
    xywh_box[1] = xywh[1];
    xywh_box[2] = xywh[2];
    xywh_box[3] = xywh[3];
    auto mc =
        this->kalman_filter.update(this->mean, this->covariance, xywh_box);
    this->mean = mc.first;
    this->covariance = mc.second;
    float x = mean[0], y = mean[1], w = mean[2], h = mean[3];
    _tlwh[0] = x - w / 2;
    _tlwh[1] = y - h / 2;
    _tlwh[2] = w;
    _tlwh[3] = h;
  } else {
    _tlwh[0] = tlwh_[0];
    _tlwh[1] = tlwh_[1];
    _tlwh[2] = tlwh_[2];
    _tlwh[3] = tlwh_[3];
  }
}

KAL_DATA STrack::predict() {
    return kalman_filter.predict(mean, covariance);
}

void STrack::multi_predict(std::unordered_map<int, STrack> &stracks) {
    for (auto it = stracks.begin(); it != stracks.end(); ++it) {
        KAL_DATA results = it->second.predict();
        it->second.mean = results.first;
        it->second.covariance = results.second;
    }
}

std::vector<float> STrack::tlwh_to_xywh(std::vector<float> tlwh_tmp) {
    std::vector<float> xywh_output(4);
    xywh_output.assign(tlwh_tmp.begin(), tlwh_tmp.end());
    xywh_output[0] += xywh_output[2] / 2;
    xywh_output[1] += xywh_output[3] / 2;
    return xywh_output;
}

