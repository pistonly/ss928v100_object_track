#include "Strack.hpp"
#include "nnn_ostrack_callback.hpp"
#include "post_process_tools.hpp"
#include "ss_mpi_vpss.h"
#include "utils.hpp"
#include "yolov8.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <fstream>
#include <half.hpp>
#include <nlohmann/json.hpp>
#include <ost_utils.hpp>
#include <string>
#include <vector>

using half_float::half;
using json = nlohmann::json;
extern Logger logger;

#define IMAGE_HEIGHT 2160
#define IMAGE_WIDTH 3840

std::atomic<bool> running(true);
void signal_handler(int signum) { running = false; }
int track_id = 0;

// 提取通用的错误处理函数
bool handle_error(const char *action, int vpss_grp, int vpss_chn, int ret) {
  if (ret != TD_SUCCESS) {
    logger.log(ERROR, action, " error, grp: ", vpss_grp, " chn: ", vpss_chn,
               " Err code: ", ret);
    return false;
  }
  return true;
}

// 获取或释放帧数据
bool process_frames(ot_video_frame_info &frame, int chn, bool release = false) {
  std::vector<std::pair<td_s32, td_s32>> grp_chns{{0, 0}, {2, 2}};

  if (chn >= grp_chns.size()) {
    logger.log(ERROR, "chn should be 0 or 1");
    return false;
  } else {
    td_s32 &vpss_grp = grp_chns[chn].first;
    td_s32 &vpss_chn = grp_chns[chn].second;

    if (release) {
      int ret = ss_mpi_vpss_release_chn_frame(vpss_grp, vpss_chn, &frame);
      if (!handle_error("Release vpss chn", vpss_grp, vpss_chn, ret))
        return false;
    } else {
      int ret = ss_mpi_vpss_get_chn_frame(vpss_grp, vpss_chn, &frame, 100);
      if (!handle_error("Get vpss chn", vpss_grp, vpss_chn, ret))
        return false;
    }
  }
  return true;
}

bool isAtImageEdge(std::vector<float> tlwh, int threshold = 5) {
  const float x0 = tlwh[0];
  const float y0 = tlwh[1];
  const float x1 = x0 + tlwh[2];
  const float y1 = y0 + tlwh[3];
  if (x0 < threshold || y0 < threshold || x1 > (IMAGE_WIDTH - threshold) ||
      y1 > (IMAGE_HEIGHT - threshold))
    return true;
  return false;
}

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

    // 检查目标是否在图像边缘，如果是则移除该追踪器
    if (isAtImageEdge(tr._tlwh)) {
      it = trackers.erase(it);
    } else {
      ++it;
    }
  }
}

void add_tracks_from_dets(std::unordered_map<int, STrack> &tracks,
                          std::vector<std::vector<std::vector<half>>> &det_bbox,
                          std::vector<std::vector<half>> &cls,
                          bool using_kal_filter, int track_max_num = 6,
                          int selected_id = 1) {
  int needed_track_num = track_max_num - tracks.size();
  int added_num = 0;
  const std::vector<std::vector<half>> &det_bbox_batch0 = det_bbox[0];
  const std::vector<half> &cls_batch0 = cls[0];
  const auto det_num = det_bbox_batch0.size();
  for (auto i = 0; i < det_num; ++i) {
    int cls_i = static_cast<int>(cls_batch0[i]);
    if (cls_i != selected_id)
      continue;
    float iou = 0;
    const std::vector<half> &xyxy = det_bbox_batch0[i];
    for (const auto &tr : tracks) {
      const std::vector<float> &xyxy_tr = tlwh2xyxy(tr.second._tlwh);
      float iou_tmp = cal_iou(xyxy.data(), xyxy_tr.data());
      if (iou_tmp > iou)
        iou = iou_tmp;
    }
    if (iou > 0.1)
      continue;
    else {
      std::vector<float> tlwh = xyxy2tlwh(xyxy);
      tracks.emplace(track_id++, STrack(tlwh, using_kal_filter));
      added_num++;
      if (added_num == needed_track_num)
        break;
    }
  }
}

int main(int argc, char *argv[]) {
  // OST model params
  std::cout << "Usage: " << argv[0] << " <config_path>" << std::endl;
  std::string configure_path = "../data/configure.json";

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
      "rtsp_url",        "om_path",        "yolov8_om_path", "tcp_id",
      "tcp_port",        "output_dir",     "save_result",    "decode_step_mode",
      "yolov8_roi_left", "yolov8_roi_top", "yolov8_scale"};
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
  int yolov8_time_interval =
    config_data["yolov8_time_interval"]; // s
  yolov8_time_interval *= 1000000; // us
  int selected_det_id = config_data["selected_det_id"];
  int max_tracker_num = config_data["max_tracker_num"];

  // VDEC source
  std::string rtsp_url = config_data["rtsp_url"];
  const int imageH = IMAGE_HEIGHT;
  const int imageW = IMAGE_WIDTH;
  const int IMAGE_SIZE = imageH * imageW * 1.5;

  // yolov8
  const int yolov8_roi_left = config_data["yolov8_roi_left"];
  const int yolov8_roi_top = config_data["yolov8_roi_top"];
  const float yolov8_scale = config_data["yolov8_scale"];
  const float conf_thres = 0.5;
  const float iou_thres = 0.6;
  const int max_det = config_data["max_det"];
  YOLOV8 yolov8(yolov8ModelPath, output_dir);
  yolov8.set_roi_parameters(yolov8_roi_left, yolov8_roi_top, yolov8_scale);
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
  std::unordered_map<int, STrack> trackers_ch0;
  std::unordered_map<int, STrack> trackers_ch1;

  // pre-allocate buffers
  std::vector<unsigned char> img(IMAGE_SIZE);
  ot_video_frame_info frame_ch0;
  ot_video_frame_info frame_ch1;

  signal(SIGINT, signal_handler); // Capture Ctrl+C

  // Save results
  std::ofstream real_result_f(output_dir + "results.csv");
  if (!real_result_f) {
    logger.log(ERROR, "opening file for writing: ", output_dir + "results.csv");
  } else {
    real_result_f << "imageId,trackerId,l,t,w,h" << std::endl;
  }

  int imageId = 0;
  uint64_t last_yolo_ts_ch0 = 0, last_yolo_ts_ch1 = 0;
  while (running) {
    {
      Timer timer("process one frame of chn-0");
      // process first channel
      // get one frame
      int current_ch = 0;
      if (!process_frames(frame_ch0, current_ch)) {
        break;
      }

      copy_yuv420_from_frame(reinterpret_cast<char *>(img.data()), &frame_ch0);

      if (trackers_ch0.size() < 6 &&
          (frame_ch0.video_frame.pts - last_yolo_ts_ch0) >
              yolov8_time_interval) {
        logger.log(INFO, "yolov8 processing ...");
        yolov8.process_one_image(reinterpret_cast<unsigned char *>(img.data()),
                                 imageW, imageH, det_bbox, det_conf, det_cls);
        logger.log(INFO, "add tracks ...");
        add_tracks_from_dets(trackers_ch0, det_bbox, det_cls, using_kal_filter,
                             max_tracker_num, selected_det_id);
      }

      // Use Kalman filter if enabled
      if (using_kal_filter) {
        STrack::multi_predict(trackers_ch0);
      }

      processTrackers(trackers_ch0, ostModel, img, imageW, imageH, imageId++,
                      real_result_f, save_result);

      process_frames(frame_ch0, current_ch, true);
    }
    {
      Timer timer("process one frame of chn-0");
      // process first channel
      // get one frame
      int current_ch = 1;
      if (!process_frames(frame_ch0, current_ch)) {
        break;
      }

      copy_yuv420_from_frame(reinterpret_cast<char *>(img.data()), &frame_ch1);

      if (trackers_ch1.size() < 6 &&
          (frame_ch0.video_frame.pts - last_yolo_ts_ch0) >
              yolov8_time_interval) {
        logger.log(INFO, "yolov8 processing ...");
        yolov8.process_one_image(reinterpret_cast<unsigned char *>(img.data()),
                                 imageW, imageH, det_bbox, det_conf, det_cls);
        logger.log(INFO, "add tracks ...");
        add_tracks_from_dets(trackers_ch1, det_bbox, det_cls, using_kal_filter,
                             max_tracker_num, selected_det_id);
      }

      // Use Kalman filter if enabled
      if (using_kal_filter) {
        STrack::multi_predict(trackers_ch1);
      }

      processTrackers(trackers_ch1, ostModel, img, imageW, imageH, imageId++,
                      real_result_f, save_result);

      process_frames(frame_ch1, current_ch, true);
    }
  }

  if (save_result && real_result_f.is_open()) {
    real_result_f.close();
  }
}
