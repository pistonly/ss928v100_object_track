//
// Created by oem on 24-3-25.
//

#pragma once


#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


struct DrBBox {
    float x0;
    float y0;
    float x1;
    float y1;
};

struct DrOBB {
    DrBBox box;
    float score;
    int class_id;
};

struct LensTrackingResults {
    std::vector<DrBBox> results;
    std::vector<std::string> times;
};

struct SwitchInfo {
    std::string to_cam;
    std::vector<float> new_init_box;  // tlwh
};

struct LensInfo {
    cv::Mat template_img;
    std::string id_in_cam;
    std::string from_cam_id;
    std::string out_of_bounds = "0";
    LensTrackingResults tracking_results;
    SwitchInfo switch_info;
};


struct TrackingResults {
    cv::Mat template_img;
    DrBBox result;
    std::string time;
    std::string from_cam_id;
    std::string out_of_bounds = "0";
    SwitchInfo switch_info;

};


struct ReceiveInfo {
    cv::Mat template_img;
    SwitchInfo switch_info;
};


struct TotalInfo {
    std::vector<std::string> id_in_cam;
    cv::Mat template_img;
    std::string start_cam ;
    std::string current_cam;
    std::string start_time;
    std::string end_time ;
    std::string cam_trajectory = "";
    std::string out_of_bounds ;
    std::unordered_map<std::string, LensTrackingResults> tracking_results;
};




//struct TotalInfo {
//    std::vector<std::string> id_in_cam = {"99-1", "100-1"};
//    cv::Mat template_img;
//    std::string start_cam = "99";
//    std::string current_cam = "100";
//    std::string start_time = "2024-03-08 10:54:23.164";
//    std::string end_time = "2024-03-08 10:58:52.112";
//    std::string cam_trajectory = "99 100";
//    std::string out_of_bounds = "0";
//    std::unordered_map<std::string, LensTrackingResults> tracking_results;
//
//};


//Kalmanfilter
//typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_FILTER;

typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> TRACKTBOX;
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

typedef std::unordered_map<std::string, TrackingResults> TRACKING_RESULTS;
typedef std::unordered_map<int, TotalInfo> TOTAL_DICT;




