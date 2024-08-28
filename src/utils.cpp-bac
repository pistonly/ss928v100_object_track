//
// Created by oem on 24-3-28.
//

#include "utils.h"


std::unordered_map<std::string, std::vector<int>> shift_pixels = {
    {"11", {900, 1600, 2700, 3800}},
    {"12", {900, 1600, 2000, 3100}},
    {"13", {900, 1600, 1500, 2600}},
    {"14", {900, 1600, 1100, 2200}},
    {"15", {900, 1600, 600, 1700}},
    {"16", {900, 1600, 0, 1100}},
    {"21", {600, 1300, 2700, 3800}},
    {"22", {600, 1300, 2000, 3100}},
    {"23", {600, 1300, 1500, 2600}},
    {"24", {600, 1300, 1100, 2200}},
    {"25", {600, 1300, 600, 1700}},
    {"26", {600, 1300, 0, 1100}},
    {"31", {300, 1000, 2700, 3800}},
    {"32", {300, 1000, 2000, 3100}},
    {"33", {300, 1000, 1500, 2600}},
    {"34", {300, 1000, 1100, 2200}},
    {"35", {300, 1000, 600, 1700}},
    {"36", {300, 1000, 0, 1100}}
};


std::vector<std::pair<int, int>> find_coordinates(const std::vector<std::vector<int>>& matrix, const int target) {
    std::vector<std::pair<int, int>> result;

    for (int i = 0; i < matrix.size(); ++i) {
        for (int j = 0; j < matrix[i].size(); ++j) {
            if (matrix[i][j] == target) {
                result.push_back(std::make_pair(i, j));
            }
        }
    }

    return result;
}


std::vector<std::pair<int, int>> find_coordinates(const std::vector<std::vector<std::string>>& matrix, const std::string& target) {
    std::vector<std::pair<int, int>> result;

    for (int i = 0; i < matrix.size(); ++i) {
        for (int j = 0; j < matrix[i].size(); ++j) {
            if (matrix[i][j] == target) {
                result.push_back(std::make_pair(i, j));
            }
        }
    }

    return result;
}

std::pair<int, int> find_value_index(const std::vector<std::vector<std::string>>& matrix, const std::string& target_value) {
    for (int i = 0; i < matrix.size(); ++i) {
        const auto& row = matrix[i];
        for (int j = 0; j < row.size(); ++j) {
            if (row[j] == target_value) {
                return std::make_pair(i, j);
            }
        }
    }
    return std::make_pair(-1, -1);
}



std::vector<float> tranpose_boundary_box(const KAL_MEAN& tbox,
                                                    const cv::Mat& Hmatrix_from,
                                                    const cv::Mat& Hmatrix_to,
                                                    const std::string& tag_from,
                                                    const std::string& tag_to) {

    std::vector<int> xxyy_from = shift_pixels[tag_from];
    std::vector<int> xxyy_to = shift_pixels[tag_to];


    std::vector<cv::Point2f> cxy;
    float w = tbox[2];
    float h = tbox[3];

    cxy.push_back(cv::Point2f(tbox[0],  tbox[1]));

    std::vector<cv::Point2f> trans_dim;


    cv::perspectiveTransform(cxy, trans_dim, Hmatrix_from);


    trans_dim[0].x += xxyy_from[2];
    trans_dim[0].y += xxyy_from[0];

    trans_dim[0].x -= xxyy_to[2];
    trans_dim[0].y -= xxyy_to[0];
    cv::Mat inv_Hmatrix_to;
    double determinant = cv::invert(Hmatrix_to, inv_Hmatrix_to);

    std::vector<cv::Point2f> box_to_dim;
    cv::perspectiveTransform(trans_dim, box_to_dim, inv_Hmatrix_to);


    std::vector<float> box_to(4);

    box_to[0] = box_to_dim[0].x - w /2;
    box_to[1] = box_to_dim[0].y - h /2;
    box_to[2] = w;
    box_to[3] = h;

    return box_to;
}


void generate_DrOBB(std::vector<std::vector<std::vector<cv::Rect>>> &initial_box_matrix_,std::vector<std::vector<std::vector<DrOBB>>> &initial_box_matrix){
    DrOBB bbox;
    for (int i=0; i< initial_box_matrix_.size(); i++){
        for (int j=0;j<initial_box_matrix_[i].size();j++){
            for (int k=0;k<initial_box_matrix_[i][j].size();k++){

                cv::Rect trackWindow = initial_box_matrix_[i][j][k];
                bbox.box.x0 = trackWindow.x;
                bbox.box.x1 = trackWindow.x+trackWindow.width;
                bbox.box.y0 = trackWindow.y;
                bbox.box.y1 = trackWindow.y+trackWindow.height;
                initial_box_matrix[i][j].push_back(bbox);

            }
        }
    }

}



std::vector<std::unordered_map<std::string,ReceiveInfo>> process_total_dict(const TRACKING_RESULTS& tracking_results,
        TOTAL_DICT &total_dict){


    std::vector<std::unordered_map<std::string,ReceiveInfo>> switch_info_list;
    std::vector<std::string> id_in_cam_list;

    for (const auto& it:tracking_results){
        std::string id_in_cam = it.first;
        auto track_data = it.second;
        if (track_data.from_cam_id !=""){
            std::string from_cam_id = track_data.from_cam_id;
            for (auto& it2:total_dict){
                if (from_cam_id == it2.second.id_in_cam.back()){
                    it2.second.id_in_cam.push_back(id_in_cam);
                }
            }
        }
    }
    if (!total_dict.empty()){
        for (const auto& it2:total_dict){
            id_in_cam_list.push_back(it2.second.id_in_cam.back());
        }
    }


    for (const auto& it:tracking_results) {
        std::string id_in_cam = it.first;
        auto track_data = it.second;

        size_t pos = id_in_cam.find("-");
        std::string cam_id = id_in_cam.substr(0, pos);
        int id_in_host;
        auto it3 = std::find(id_in_cam_list.begin(), id_in_cam_list.end(), id_in_cam);
        if (it3 != id_in_cam_list.end()) {
            id_in_host = std::distance(id_in_cam_list.begin(), it3);

            total_dict[id_in_host].tracking_results[cam_id].results.push_back(track_data.result);
            total_dict[id_in_host].tracking_results[cam_id].times.push_back(track_data.time);
            total_dict[id_in_host].current_cam = cam_id;

            if (track_data.out_of_bounds=="1"){
                total_dict[id_in_host].end_time = track_data.time;
                total_dict[id_in_host].out_of_bounds = track_data.out_of_bounds;
            }
            if (!track_data.switch_info.to_cam.empty()){
                std::string to_cam = track_data.switch_info.to_cam;
                auto new_init_box = track_data.switch_info.new_init_box;
                DrBBox DRnew_init_box;
                DRnew_init_box.x0 = new_init_box[0];
                DRnew_init_box.y1 = new_init_box[1];
                DRnew_init_box.x1 = new_init_box[0] + new_init_box[2];
                DRnew_init_box.y1 = new_init_box[1] + new_init_box[3];
                total_dict[id_in_host].tracking_results[to_cam].results.push_back(DRnew_init_box);
                total_dict[id_in_host].tracking_results[to_cam].times.push_back(track_data.time);
                total_dict[id_in_host].cam_trajectory += " " + to_cam;

                std::unordered_map<std::string,ReceiveInfo> switch_info;
                switch_info[id_in_cam] = {
                        .template_img=total_dict[id_in_host].template_img,
                        .switch_info = {
                                .to_cam=to_cam,
                                .new_init_box=new_init_box
                        }
                };
                switch_info_list.push_back(switch_info);

            }

        }
        else{

            int new_id_in_host=0;
            int max_id = 0;
            if (!total_dict.empty()) {
                // 获取最大键值

                for (const auto& pair:total_dict){
                    if (pair.first > max_id){
                        max_id = pair.first;
                    }
                }
                new_id_in_host = max_id +1;
            }
            total_dict[new_id_in_host].id_in_cam.push_back(id_in_cam);
            total_dict[new_id_in_host].template_img = track_data.template_img;
            total_dict[new_id_in_host].start_cam = cam_id;
            total_dict[new_id_in_host].current_cam = cam_id;
            total_dict[new_id_in_host].start_time = track_data.time;
            total_dict[new_id_in_host].end_time = "";
            total_dict[new_id_in_host].out_of_bounds = "0";
            total_dict[new_id_in_host].cam_trajectory = cam_id;
            total_dict[new_id_in_host].tracking_results[cam_id].results.push_back(track_data.result);
            total_dict[new_id_in_host].tracking_results[cam_id].times.push_back(track_data.time);
        }
    }
    return  switch_info_list;




}