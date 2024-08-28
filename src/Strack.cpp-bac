//
// Created by oem on 24-3-25.
//

#include "Strack.h"

#include <unordered_map>


STrack::STrack(std::vector<float> tlwh_) {

    _tlwh.resize(4);
    _tlwh.assign(tlwh_.begin(),tlwh_.end());

}

STrack::~STrack() {
}

void STrack::activate(KalmanFilter &kalman_filter) {
    this->kalman_filter = kalman_filter;
    std::vector<float> xywh = tlwh_to_xywh(_tlwh);
    TRACKTBOX xywh_box;
    xywh_box[0] = xywh[0];
    xywh_box[1] = xywh[1];
    xywh_box[2] = xywh[2];
    xywh_box[3] = xywh[3];
    auto mc = this->kalman_filter.initiate(xywh_box);
    this->mean = mc.first;
    this->covariance = mc.second;
    int a =1;
}


void STrack::update(std::vector<float> tlwh_) {

//    tlwh_ = {716.5677909851074, 1550.4425792694092, 114.02051544189453, 58.98624038696289};
    std::vector<float> xywh = tlwh_to_xywh(tlwh_);
    TRACKTBOX xywh_box;
    xywh_box[0] = xywh[0];
    xywh_box[1] = xywh[1];
    xywh_box[2] = xywh[2];
    xywh_box[3] = xywh[3];
    auto mc = this->kalman_filter.update(this->mean,this->covariance,xywh_box);
    this->mean = mc.first;
    this->covariance = mc.second;


}


void STrack::multi_predict(std::unordered_map<int,STrack> &stracks, KalmanFilter &kalman_filter) {
    for (auto it = stracks.begin(); it != stracks.end(); ++it) {
        int track_id = it->first;
        KAL_DATA results;

        results = kalman_filter.predict(stracks[track_id].mean,stracks[track_id].covariance);

        stracks[track_id].mean = results.first;
        stracks[track_id].covariance = results.second;
    }


}




std::vector<float> STrack::tlwh_to_xywh(std::vector<float> tlwn_tmp) {
    std::vector<float> xywh_output;

    xywh_output.resize(4);
    xywh_output.assign(tlwn_tmp.begin(),tlwn_tmp.end());

    xywh_output[0] += xywh_output[2] /2;
    xywh_output[1] += xywh_output[3] /2;

    return xywh_output;

}

