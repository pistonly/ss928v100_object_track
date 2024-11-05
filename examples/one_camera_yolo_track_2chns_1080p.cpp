#include "Strack.hpp"
#include "nnn_ostrack_callback.hpp"
#include "post_process_tools.hpp"
#include "ss_mpi_vpss.h"
#include "tcp_tools.hpp"
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

#define IMAGE_HEIGHT 1152
#define IMAGE_WIDTH 1920
#define OFFSET_W 0
#define OFFSET_H 36

std::atomic<bool> running(true);
void signal_handler(int signum) { running = false; }
int track_id = 0;

static TCP tcp_obj;

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
  std::vector<std::pair<td_s32, td_s32>> grp_chns{{0, 1}, {2, 3}};

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

void processTrackers(std::unordered_map<int, STrack> &trackers,
                     NNN_Ostrack_Callback &ostModel,
                     const std::vector<unsigned char> &img, int imageW,
                     int imageH, uint8_t cameraId, uint64_t timestamp,
                     std::ofstream &real_result_f, bool save_result,
                     bool b_tcp_send) {
  std::vector<std::vector<float>> track_res;
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

    // NOTE: tcp client need: x0, y0, x1, y1, conf, track_id at 1920x1080 frame.
    int offset_y = -1 * OFFSET_H;
    // std::vector<std::vector<float>> track_res(
    //     {{tr._tlwh[0], tr._tlwh[1] + offset_y, tr._tlwh[0] + tr._tlwh[2],
    //       tr._tlwh[0] + tr._tlwh[3] + offset_y, 0.f, trackerId}});
    // for 2K visualization
    std::vector<float> track_res_one(
        {tr._tlwh[0] / 2, (tr._tlwh[1] + offset_y) / 2,
         (tr._tlwh[0] + tr._tlwh[2]) / 2,
         (tr._tlwh[0] + tr._tlwh[3] + offset_y) / 2, 0.f, trackerId});
    track_res.push_back(track_res_one);
    // std::vector<std::vector<float>> track_res(
    //     {{tr._tlwh[0] / 2, (tr._tlwh[1] + offset_y) / 2,
    //       (tr._tlwh[0] + tr._tlwh[2]) / 2,
    //       (tr._tlwh[0] + tr._tlwh[3] + offset_y) / 2, 0.f, trackerId}});

    // 检查目标是否在图像边缘或尺寸是否超过640x640，如果是则移除该追踪器
    if (isAtImageEdge(tr._tlwh, edge_thres_x, edge_thres_y, IMAGE_HEIGHT,
                      IMAGE_WIDTH) ||
        tr._tlwh[2] > 640 || tr._tlwh[3] > 640) {
      it = trackers.erase(it);
    } else {
      ++it;
    }
  }

  if (save_result && real_result_f.is_open()) {
    save_one_track_result_csv(real_result_f, track_res, cameraId, timestamp);
  }

  if (b_tcp_send) {
    if (!tcp_obj.mb_sock_connected && tcp_obj.mb_tcpIp_setted) {
      tcp_obj.connect_to_tcp();
    }
    if (tcp_obj.mb_sock_connected) {
      send_track_result(tcp_obj.m_sock, track_res, cameraId, timestamp);
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
    if (iou > 0.1) {
      logger.log(DEBUG, "skip det in tracker-pool");
      continue;
    } else {
      logger.log(DEBUG, "to be add det: ", xyxy[0], ", ", xyxy[1], ", ",
                 xyxy[2], ", ", xyxy[3]);

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
      "om_path",    "yolov8_om_path", "tcp_ip",          "tcp_port",
      "output_dir", "save_result",    "decode_step_mode"};
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
  int yolov8_time_interval = config_data["yolov8_time_interval"]; // s
  yolov8_time_interval *= 1000000;                                // us
  int selected_det_id = config_data["selected_det_id"];
  int max_tracker_num = config_data["max_tracker_num"];

  // VDEC source
  const int imageH = IMAGE_HEIGHT;
  const int imageW = IMAGE_WIDTH;
  const int IMAGE_SIZE = imageH * imageW * 1.5;

  // tcp
  std::string tcp_ip = config_data["tcp_ip"];
  std::string tcp_port = config_data["tcp_port"];
  tcp_obj.set_ip_port(tcp_ip, std::stoi(tcp_port));

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

  std::vector<uint8_t> v_cameraIds;
  getCameraId_pair(v_cameraIds);

  logger.log(INFO, "cameraId number: ", v_cameraIds.size());

  // Initialize OST model
  NNN_Ostrack_Callback ostModel(omPath, template_factor, search_area_factor,
                                template_size, search_size);

  // Initialize tracking
  bool using_kal_filter = false;
  std::vector<std::unordered_map<int, STrack>> v_trackers(v_cameraIds.size());
  std::vector<uint64_t> v_last_yolo_ts(v_cameraIds.size(), 0);

  // pre-allocate buffers
  std::vector<unsigned char> img(IMAGE_SIZE);
  std::vector<ot_video_frame_info> v_frame_chs(v_cameraIds.size());

  signal(SIGINT, signal_handler); // Capture Ctrl+C

  // Save results
  std::ofstream real_result_f(output_dir + "results.csv");
  if (!real_result_f) {
    logger.log(ERROR, "opening file for writing: ", output_dir + "results.csv");
  } else {
    real_result_f << "cameraId,timestamp,trackerId,l,t,w,h" << std::endl;
  }

  while (running) {
    for (int current_ch = 0; current_ch < v_cameraIds.size(); ++current_ch) {
      Timer timer("process one frame of chn-" + std::to_string(current_ch));
      // process channel current_ch
      // get one frame
      uint8_t current_cameraId = v_cameraIds[current_ch];
      if (!process_frames(v_frame_chs[current_ch], current_ch)) {
        break;
      }

      copy_yuv420_from_frame(reinterpret_cast<char *>(img.data()),
                             &v_frame_chs[current_ch], IMAGE_HEIGHT,
                             IMAGE_WIDTH, OFFSET_H, OFFSET_W);

      auto &trackers = v_trackers[current_ch];
      auto &last_yolo_ts = v_last_yolo_ts[current_ch];

      if (trackers.size() < max_tracker_num &&
          (v_frame_chs[current_ch].video_frame.pts - last_yolo_ts) >
              yolov8_time_interval) {
        Timer timer("yolov8 processing ...");
        yolov8.process_one_image(img, det_bbox, det_conf, det_cls);
        logger.log(DEBUG, "add tracks ...");
        add_tracks_from_dets(trackers, det_bbox, det_cls, using_kal_filter,
                             max_tracker_num, selected_det_id);
        last_yolo_ts = v_frame_chs[current_ch].video_frame.pts;
      }

      // Use Kalman filter if enabled
      if (using_kal_filter) {
        STrack::multi_predict(trackers);
      }

      processTrackers(trackers, ostModel, img, imageW, imageH, current_cameraId,
                      v_frame_chs[current_ch].video_frame.pts / 1000,
                      real_result_f, save_result, true);

      process_frames(v_frame_chs[current_ch], current_ch, true);
    }
  }

  if (save_result && real_result_f.is_open()) {
    real_result_f.close();
  }
}
