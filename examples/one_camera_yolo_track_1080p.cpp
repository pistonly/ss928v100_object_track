#include "Strack.hpp"
#include "ffmpeg_vdec_vpss.hpp"
#include "nnn_ostrack_callback.hpp"
#include "post_process_tools.hpp"
#include "utils.hpp"
#include "yolov8.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <fstream>
#include <half.hpp>
#include <iterator>
#include <nlohmann/json.hpp>
#include <ost_utils.hpp>
#include <sstream>
#include <string>
#include <vector>

using half_float::half;
using json = nlohmann::json;
extern Logger logger;

#define IMAGE_HEIGHT 1152
#define IMAGE_WIDTH 1920
#define OFFSET_W 0
#define OFFSET_H 36

std::atomic<bool> running(true);
void signal_handler(int signum) { running = false; }
int g_track_id = 0;

void processTrackers(std::unordered_map<int, STrack> &trackers,
                     NNN_Ostrack_Callback &ostModel,
                     const std::vector<unsigned char> &img, int imageW,
                     int imageH, int imageId, std::ofstream &real_result_f,
                     bool save_result) {
  for (auto it = trackers.begin(); it != trackers.end();) {
    int trackerId = it->first;
    auto &tr = it->second;
    auto &tlwh = tr._tlwh;
    int t_x0 = static_cast<int>(tlwh[0]);
    int t_y0 = static_cast<int>(tlwh[1]);
    int t_w = static_cast<int>(tlwh[2]);
    int t_h = static_cast<int>(tlwh[3]);

    float search_resize_factor;
    int search_crop_x0, search_crop_y0;
    std::vector<float> tlwh_new;
    int thres = 5;
    int edge_thres_x = thres;
    int edge_thres_y = OFFSET_H + thres;

    {
      Timer timer("model duration");
      if (ostModel.preprocess(img.data(), imageW, imageH, t_x0, t_y0, t_w, t_h,
                              search_resize_factor, search_crop_x0,
                              search_crop_y0, tr.template_packet) != SUCCESS) {
        ++it;
        continue;
      }

      if (ostModel.ExecuteRPN_Async() != SUCCESS ||
          ostModel.SynchronizeStream() != SUCCESS) {
        ++it;
        continue;
      }

      if (ostModel.postprocess(search_crop_x0, search_crop_y0,
                               search_resize_factor, tlwh_new) != SUCCESS) {
        ++it;
        continue;
      }

      tr.update(tlwh_new);
    }

    if (save_result && real_result_f.is_open()) {
      real_result_f << imageId << ", " << trackerId << ", " << tr._tlwh[0]
                    << ", " << tr._tlwh[1] << ", " << tr._tlwh[2] << ", "
                    << tr._tlwh[3] << std::endl;
    }

    // 检查目标是否在图像边缘或尺寸是否超过640x640，如果是则移除该追踪器
    if (isAtImageEdge(tr._tlwh, edge_thres_x, edge_thres_y, IMAGE_HEIGHT,
                      IMAGE_WIDTH) ||
        tr._tlwh[2] > 640 || tr._tlwh[3] > 640) {
      it = trackers.erase(it);
    } else {
      ++it;
    }
  }
}

float cal_privilege(int cls_order, bool in_privileged_region, float conf,
                    float cls_coef, float region_coef, float conf_coef) {
  float region_val;
  if (in_privileged_region)
    region_val = 1;
  else
    region_val = 0;

  return (10 - cls_order) * cls_coef + region_val * region_coef +
         conf * conf_coef;
}

void add_tracks_from_dets(std::unordered_map<int, STrack> &tracks,
                          std::vector<std::vector<std::vector<half>>> &det_bbox,
                          std::vector<std::vector<half>> det_conf,
                          std::vector<std::vector<half>> &cls,
                          bool using_kal_filter, std::vector<int> selected_ids,
                          int track_max_num = 6, float skip_iou_thres = 0.5,
                          float delete_iou_thres = 0.3, int privileged_x0 = 0,
                          int privileged_x1 = IMAGE_WIDTH,
                          int privileged_y0 = 0,
                          int privileged_y1 = IMAGE_HEIGHT, float cls_coef = 10,
                          float region_coef = 100, float conf_coef = 1.f) {
  const std::vector<std::vector<half>> &det_bbox_batch0 = det_bbox[0];
  const std::vector<half> &det_conf_batch0 = det_conf[0];
  const std::vector<half> &cls_batch0 = cls[0];
  const auto det_num = det_bbox_batch0.size();
  // max iou of each tracker
  std::unordered_map<int, float> track_ious;
  for (const auto &tr : tracks) {
    track_ious.emplace(tr.first, 0.f);
  }

  std::vector<std::pair<float, std::vector<float>>> to_be_added_dets_with_score;

  float privilege_score = 0.f;
  int index_ord = 0;
  bool in_privileged_region = false;
  for (auto i = 0; i < det_num; ++i) {
    int cls_i = static_cast<int>(cls_batch0[i]);

    auto index_it = std::find(selected_ids.begin(), selected_ids.end(), cls_i);
    if (index_it == selected_ids.end())
      continue;
    else {
      index_ord = std::distance(selected_ids.begin(), index_it);
    }

    float iou = 0;
    const std::vector<half> &xyxy = det_bbox_batch0[i];

    for (const auto &tr : tracks) {
      const std::vector<float> &xyxy_tr = tlwh2xyxy(tr.second._tlwh);
      float iou_tmp = cal_iou(xyxy.data(), xyxy_tr.data());
      if (iou_tmp > iou)
        iou = iou_tmp;
      if (iou_tmp > track_ious[tr.first])
        track_ious[tr.first] = iou_tmp;
    }
    if (iou > skip_iou_thres) {
      logger.log(DEBUG, "skip det in tracker-pool");
      continue;
    } else {
      logger.log(DEBUG, "to be add det: ", xyxy[0], ", ", xyxy[1], ", ",
                 xyxy[2], ", ", xyxy[3]);
      std::vector<float> tlwh = xyxy2tlwh(xyxy);
      // whether in privileged region
      if (tlwh[0] < privileged_x1 && tlwh[0] >= privileged_x0 &&
          tlwh[1] < privileged_y1 && tlwh[1] >= privileged_y0) {
        in_privileged_region = true;
      } else {
        in_privileged_region = false;
      }

      privilege_score =
          cal_privilege(index_ord, in_privileged_region, det_conf_batch0[i],
                        cls_coef, region_coef, conf_coef);
      //
      to_be_added_dets_with_score.push_back(
          std::make_pair(privilege_score, std::move(tlwh)));
    }
  }

  // remove tracker who is not in detections
  for (auto it = tracks.begin(); it != tracks.end();) {
    const auto tId = it->first;
    const auto iou = track_ious[tId];
    if (iou < delete_iou_thres) {
      it = tracks.erase(it);
      logger.log(DEBUG, "Delete tracker: ", tId, " iou: ", iou);
    } else {
      ++it;
      logger.log(DEBUG, "Confirm tracker: ", tId, " iou: ", iou);
    }
  }

  // sort dets by privilege_score;
  std::sort(to_be_added_dets_with_score.begin(),
            to_be_added_dets_with_score.end(),
            [](const std::pair<float, std::vector<float>> &a,
               const std::pair<float, std::vector<float>> &b) {
              return a.first > b.first;
            });

  // add trackers
  if (tracks.size() < track_max_num) {
    int needed_track_num = track_max_num - tracks.size();
    int added_num = 0;
    for (const auto &tlwh_pair : to_be_added_dets_with_score) {
      logger.log(DEBUG, "Add new tracker: ", g_track_id,
                 " tlwh: ", tlwh_pair.second[0], ", ", tlwh_pair.second[1],
                 ", ", tlwh_pair.second[2], ", ", tlwh_pair.second[3]);
      tracks.emplace(g_track_id++, STrack(tlwh_pair.second, using_kal_filter));
      added_num++;
      if (added_num == needed_track_num)
        break;
    }
  }
  logger.log(INFO, "current tracker num: ", tracks.size());
}

int main(int argc, char *argv[]) {
  // OST model params
  std::cout << "Usage: " << argv[0] << " <config_path>" << std::endl;
  std::string configure_path = "../data/configure_1080p.json";

  if (argc > 1)
    configure_path = argv[1];

  // read configure
  std::ifstream config_file(configure_path);
  if (!config_file.is_open()) {
    logger.log(ERROR, "Can't open configure file: ", configure_path);
    return 1;
  }

  json config_data;
  try {
    config_file >> config_data;
  } catch (json::parse_error &e) {
    logger.log(ERROR, "JSON parse error: ", e.what());
    return 1;
  }

  // 设置日志级别
  if (config_data.contains("log_level")) {
    std::string level = config_data["log_level"];
    if (level == "DEBUG") {
      logger.setLogLevel(DEBUG);
    } else if (level == "INFO") {
      logger.setLogLevel(INFO);
    } else if (level == "WARNING") {
      logger.setLogLevel(WARNING);
    } else if (level == "ERROR") {
      logger.setLogLevel(ERROR);
    } else {
      logger.log(WARNING, "Unknown log level: ", level, ", using INFO level.");
      logger.setLogLevel(INFO);
    }
  }

  std::vector<std::string> required_keys = {
      "rtsp_url", "om_path",    "yolov8_om_path", "tcp_ip",
      "tcp_port", "output_dir", "save_result",    "decode_step_mode"};
  for (const auto &key : required_keys) {
    if (!config_data.contains(key)) {
      logger.log(ERROR, "Can't find key: ", key);
      return 1;
    }
  }

  std::string omPath = config_data["om_path"];
  std::string yolov8ModelPath = config_data["yolov8_om_path"];
  float template_factor = config_data["template_factor"];
  float search_area_factor = config_data["search_area_factor"];
  int template_size = config_data["template_size"];
  int search_size = config_data["search_size"];
  bool save_result = config_data["save_result"];
  std::string output_dir = config_data["output_dir"];

  int yolov8_time_interval_s = config_data["yolov8_time_interval"]; // s
  std::chrono::microseconds yolov8_time_interval(yolov8_time_interval_s *
                                                 1000000); // 微秒

  std::vector<int> selected_det_ids = config_data["selected_det_ids"].get<std::vector<int>>();
  int max_tracker_num = config_data["max_tracker_num"];

  // VDEC source
  std::string rtsp_url = config_data["rtsp_url"];
  const int imageH = IMAGE_HEIGHT;
  const int imageW = IMAGE_WIDTH;
  const int IMAGE_SIZE = imageH * imageW * 1.5;

  // Initialize decoder
  HardwareDecoder decoder(rtsp_url, true);
  decoder.start_decode();

  // yolov8
  const float conf_thres = config_data["conf_thres"];
  const float iou_thres = config_data["iou_thres"];
  const int max_det = config_data["max_det"];
  YOLOV8 yolov8(yolov8ModelPath, output_dir);
  yolov8.set_postprocess_parameters(conf_thres, iou_thres, max_det);
  int batch_num = yolov8.mv_outputs_dim[0][0];
  std::vector<std::vector<std::vector<half>>> det_bbox(batch_num);
  std::vector<std::vector<half>> det_conf(batch_num);
  std::vector<std::vector<half>> det_cls(batch_num);

  // Initialize OST model
  NNN_Ostrack_Callback ostModel(omPath, template_factor, search_area_factor,
                                template_size, search_size);

  // Initialize tracking
  bool using_kal_filter = false;
  std::unordered_map<int, STrack> trackers;

  // pre-allocate buffers
  std::vector<unsigned char> img(IMAGE_SIZE);
  // fill YUV to gray image
  const int Y_size = imageH * imageW;
  const int UV_size = Y_size / 2;
  std::fill(img.begin(), img.begin() + Y_size, 114);
  std::fill(img.begin() + Y_size, img.end(), 128);

  signal(SIGINT, signal_handler); // Capture Ctrl+C

  // Save results
  std::ofstream real_result_f = create_file_from_pts(output_dir, "results.csv", output_dir);
  if (!real_result_f) {
    logger.log(ERROR, "opening file for writing: ", output_dir + "results.csv");
  } else {
    real_result_f << "imageId,trackerId,l,t,w,h" << std::endl;
  }

  // 初始化 last_yolov8_time 为当前时间
  auto last_yolov8_time = std::chrono::steady_clock::now();
  int imageId = 0;
  while (running && !decoder.is_ffmpeg_exit()) {
    {
      Timer timer("process one frame");
      if (decoder.get_frame_without_release()) {
        logger.log(DEBUG, "Got one frame");
        copy_yuv420_from_frame(reinterpret_cast<char *>(img.data()),
                               &decoder.frame_L, IMAGE_HEIGHT, IMAGE_WIDTH,
                               OFFSET_H, OFFSET_W);
        // // debug
        // std::stringstream ss;
        // ss << "/mnt/data/sot/frame_" << imageId << ".jpg";
        // saveBinaryFile(img, ss.str());

        // 获取当前时间
        auto now = std::chrono::steady_clock::now();
        // 计算距离上一次执行 yolov8 的时间差
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            now - last_yolov8_time);

        if (elapsed >= yolov8_time_interval && trackers.size() < 6) {
          Timer timer("yolov8 processing ...");
          // add new trackers
          // std::cout << "yolov8 processing ..." << std::endl;
          yolov8.process_one_image(img, det_bbox, det_conf, det_cls);
          std::cout << "add tracks ... " << std::endl;
          add_tracks_from_dets(trackers, det_bbox, det_conf, det_cls,
                               using_kal_filter, selected_det_ids,
                               max_tracker_num);
          last_yolov8_time = now;
          // save yolov8 results
          std::string det_file_name = getCurrentTimeWithMilliseconds() + "_" + std::to_string(det_bbox[0].size()) + ".csv";
          save_detect_results_csv(det_bbox, det_conf, det_cls, output_dir, det_file_name);

        }

        // Use Kalman filter if enabled
        if (using_kal_filter) {
          STrack::multi_predict(trackers);
        }

        processTrackers(trackers, ostModel, img, imageW, imageH, imageId++,
                        real_result_f, save_result);

      } else {
        break;
      }

      decoder.release_frames();
    }
  }

  if (save_result && real_result_f.is_open()) {
    real_result_f.close();
  }
}
