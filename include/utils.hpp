//
// Created by oem on 24-3-28.
//

#pragma once

# include <iostream>
# include <vector>
# include "dataType.h"
#include <opencv2/imgproc.hpp>
#include "opencv2/core.hpp"
#include <unordered_map>



std::vector<std::pair<int, int>> find_coordinates(const std::vector<std::vector<int>>& matrix, int target);
std::vector<std::pair<int, int>> find_coordinates(const std::vector<std::vector<std::string>>& matrix, const std::string& target);
std::pair<int, int> find_value_index(const std::vector<std::vector<std::string>>& matrix, const std::string& target_value) ;


std::vector<float> tranpose_boundary_box(const KAL_MEAN& tbox,
                                                    const cv::Mat& Hmatrix_from,
                                                    const cv::Mat& Hmatrix_to,
                                                    const std::string& tag_from,
                                                    const std::string& tag_to);

void generate_DrOBB(std::vector<std::vector<std::vector<cv::Rect>>> &initial_box_matrix_, std::vector<std::vector<std::vector<DrOBB>>> &initial_box_matrix);

std::vector<std::unordered_map<std::string,ReceiveInfo>> process_total_dict(const TRACKING_RESULTS& tracking_results,
                        TOTAL_DICT &total_dict);