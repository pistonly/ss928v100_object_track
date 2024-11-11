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
#include <sys/types.h>
#include <ctime>


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

bool isCurrentYear2024() {
  // 获取当前时间点
  auto now = std::chrono::system_clock::now();
  // 将时间点转换为 time_t 类型
  std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  // 转换为本地时间结构
  std::tm *localTime = std::localtime(&now_c);
  // 提取年份，tm_year 是从 1900 年开始计数的年份
  int year = localTime->tm_year + 1900;
  // 检查年份是否为 2024
  return (year == 2024);
}

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
    // send fake data for visualization
    track_res.push_back({0.f, 0.f, 0.f, 0.f, -1.f, 0.f});
    if (!tcp_obj.mb_sock_connected && tcp_obj.mb_tcpIp_setted) {
      tcp_obj.connect_to_tcp();
    }
    if (tcp_obj.mb_sock_connected) {
      send_track_result(tcp_obj, track_res, cameraId, timestamp);
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
  int needed_track_num = track_max_num - tracks.size();
  int added_num = 0;
  for (const auto &tlwh_pair : to_be_added_dets_with_score) {
    tracks.emplace(track_id++, STrack(tlwh_pair.second, using_kal_filter));
    added_num++;
    if (added_num == needed_track_num)
      return;
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

  // check time
  logger.log(INFO, "checking is current year 2024");
  while (true) {
    if (isCurrentYear2024()) {
      break;
    } else {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      logger.log(INFO, "checking is current year 2024");
    }
  }

  sync_to_system_time();
  // sync mpi time to system time, every 5s
  TimeSynchronizer sync_time(5000000);
  sync_time.sync();


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
  std::vector<int> selected_det_ids =
      config_data["selected_det_ids"].get<std::vector<int>>();
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
  std::ofstream real_result_f =
      create_file_from_pts(output_dir, "results.csv", output_dir);
  if (!real_result_f) {
    logger.log(ERROR, "opening file for writing: ", output_dir + "results.csv");
  } else {
    real_result_f << "cameraId,timestamp,x0,y0,x1,y1,conf,trackerId" << std::endl;
  }


  // sleep for next yolov8_time_interval
  int64_t _now = getCurrentTimestampInMicroseconds();
  int64_t start_pts = (_now / yolov8_time_interval + 1) * yolov8_time_interval;
  std::this_thread::sleep_for(std::chrono::microseconds(start_pts - _now));
  std::vector<int> v_last_multiple(v_frame_chs.size(), -1);

  // output start time
  logger.log(INFO, getCurrentTimeWithMilliseconds());

  while (running) {
    sync_time.sync();

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

      // run yolo at time_point which multiple of yolov8_time_interval
      _now = v_frame_chs[current_ch].video_frame.pts;
      int current_multiple = (_now - start_pts) / yolov8_time_interval;
      if (current_multiple > v_last_multiple[current_ch]) {
        v_last_multiple[current_ch] = current_multiple;
        Timer timer("yolov8 processing ...");
        yolov8.process_one_image(img, det_bbox, det_conf, det_cls);
        logger.log(DEBUG, "add tracks ...");
        add_tracks_from_dets(trackers, det_bbox, det_conf, det_cls,
                             using_kal_filter, selected_det_ids,
                             max_tracker_num);
        last_yolo_ts = v_frame_chs[current_ch].video_frame.pts;
        // save yolov8 results: timestamp_cameraId_detectNum.csv
        std::string det_file_name =
            from_pts_to_strWithMilliseconds(
                v_frame_chs[current_ch].video_frame.pts / 1000) +
            "_" + std::to_string(static_cast<int>(current_cameraId)) + "_" +
            std::to_string(det_bbox[0].size()) + ".csv";
        save_detect_results_csv(det_bbox, det_conf, det_cls, output_dir,
                                det_file_name);
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
