#include "Strack.hpp"
#include "ffmpeg_vdec_vpss.hpp"
#include "nnn_ostrack_callback.hpp"
#include "post_process_tools.hpp"
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
                     int imageH, int imageId, bool &template_initialized,
                     std::ofstream &real_result_f, bool save_result) {
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

  // VDEC source
  std::string rtsp_url = config_data["rtsp_url"];
  const int imageH = IMAGE_HEIGHT;
  const int imageW = IMAGE_WIDTH;
  const int IMAGE_SIZE = imageH * imageW * 1.5;

  // Initialize decoder
  HardwareDecoder decoder(rtsp_url, true);
  decoder.start_decode();

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
  std::unordered_map<int, STrack> trackers;

  std::vector<unsigned char> img(IMAGE_SIZE);
  bool template_initialized = false;
  signal(SIGINT, signal_handler); // Capture Ctrl+C

  // Save results
  std::ofstream real_result_f(output_dir + "results.csv");
  if (!real_result_f) {
    logger.log(ERROR, "opening file for writing: ", output_dir + "results.csv");
  } else {
    real_result_f << "imageId,trackerId,l,t,w,h" << std::endl;
  }

  int imageId = 0;
  while (running && !decoder.is_ffmpeg_exit()) {
    {
      Timer timer("process one frame");
      if (decoder.get_frame_without_release()) {
        std::cout << "Got one frame" << std::endl;
        copy_yuv420_from_frame(reinterpret_cast<char *>(img.data()),
                               &decoder.frame_H);

        if (imageId % 10 == 0 && trackers.size() < 6) {
          // add new trackers
          std::cout << "yolov8 processing ..." << std::endl;
          yolov8.process_one_image(
              reinterpret_cast<unsigned char *>(img.data()), imageW, imageH,
              det_bbox, det_conf, det_cls);
          std::cout << "add tracks ... " << std::endl;
          add_tracks_from_dets(trackers, det_bbox, det_cls, using_kal_filter, 6,
                               1);
        }

        // Use Kalman filter if enabled
        if (using_kal_filter) {
          STrack::multi_predict(trackers);
        }

        processTrackers(trackers, ostModel, img, imageW, imageH, imageId++,
                        template_initialized, real_result_f, save_result);

      } else {
        break;
      }

      decoder.release_frames();
    }

    template_initialized = false;
  }

  if (save_result && real_result_f.is_open()) {
    real_result_f.close();
  }
}
