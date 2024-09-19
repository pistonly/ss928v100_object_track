#pragma once
#include "kalmanFilter.hpp"
#include <unordered_map>
#include "ost_utils.hpp"

class STrack {
public:
  explicit STrack(std::vector<float> tlwh_, bool using_kal);
  ~STrack();
  // void activate(KalmanFilter &kalman_filter);
  static void multi_predict(std::unordered_map<int, STrack> &stracks);
  KAL_DATA predict();
  void update(std::vector<float> tlwh_);

  std::vector<float> tlwh_to_xywh(std::vector<float> tlwh_tmp);

  OSTTemplateData template_packet;

public:
  std::vector<float> _tlwh;
  std::vector<float> tlbr;
  KAL_MEAN mean;
  KAL_COVA covariance;

private:
  bool mb_using_kal;
  KalmanFilter kalman_filter;
  void activate();
};
