#include "Strack.hpp"
#include "ffmpeg_vdec_vpss.hpp"
#include "nnn_ostrack_callback.hpp"
#include "utils.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <fstream>
#include <string>
#include <vector>

std::atomic<bool> running(true);
void signal_handler(int signum) { running = false; }

void processTrackers(std::unordered_map<int, STrack> &trackers,
                     NNN_Ostrack_Callback &ostModel,
                     const std::vector<unsigned char> &img, int imageW,
                     int imageH, int imageId, bool &template_initialized,
                     std::ofstream &real_result_f, bool save_result) {
  for (auto &tracker_pair : trackers) {
    int trackerId = tracker_pair.first;
    auto &tlwh = tracker_pair.second._tlwh;
    int t_x0 = static_cast<int>(tlwh[0]);
    int t_y0 = static_cast<int>(tlwh[1]);
    int t_w = static_cast<int>(tlwh[2]);
    int t_h = static_cast<int>(tlwh[3]);

    float search_resize_factor;
    int search_crop_x0, search_crop_y0;

    auto model_start_time = std::chrono::high_resolution_clock::now();

    if (ostModel.preprocess(img.data(), imageW, imageH, t_x0, t_y0, t_w, t_h,
                            search_resize_factor, search_crop_x0,
                            search_crop_y0, !template_initialized) != SUCCESS) {
      return;
    }

    if (ostModel.ExecuteRPN_Async() != SUCCESS ||
        ostModel.SynchronizeStream() != SUCCESS) {
      return;
    }

    std::vector<float> tlwh_new;
    if (ostModel.postprocess(search_crop_x0, search_crop_y0,
                             search_resize_factor, tlwh_new) != SUCCESS) {
      return;
    }

    tracker_pair.second.update(tlwh_new);

    if (save_result && real_result_f.is_open()) {
      real_result_f << "predicted results: " << imageId << ", " << trackerId
                    << ", " << tlwh_new[0] << ", " << tlwh_new[1] << ", "
                    << tlwh_new[2] << ", " << tlwh_new[3] << std::endl;
    }

    auto model_end_time = std::chrono::high_resolution_clock::now();
    std::cout << "--model duration: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     model_end_time - model_start_time)
                     .count()
              << "ms" << std::endl;
  }
}

int main(int argc, char *argv[]) {
  // OST model params
  std::string omPath = "../models/MyViTrack_ep0100_192_L3.om";
  std::string init_bbox_csv = "../data/init_bbox.csv";
  float template_factor = 1.6;
  float search_area_factor = 4;
  int template_size = 96;
  int search_size = 192;
  bool save_result = false;

  // VDEC source
  std::string rtsp_url = "rtsp://172.23.24.52:8554/test";
  const int imageH = 2160;
  const int imageW = 3840;
  const int IMAGE_SIZE = imageH * imageW * 1.5;

  // Initialize decoder
  HardwareDecoder decoder(rtsp_url, true);
  decoder.start_decode();

  // Initialize OST model
  NNN_Ostrack_Callback ostModel(omPath, template_factor, search_area_factor,
                                template_size, search_size);

  // Initialize tracking
  std::vector<std::vector<float>> tlwhs = readCSV(init_bbox_csv);
  bool using_kal_filter = false;
  int track_id = 0;
  std::unordered_map<int, STrack> trackers;
  for (const auto &tlwh: tlwhs){
    trackers.emplace(track_id++, STrack(tlwh, using_kal_filter));
  }

  std::vector<unsigned char> img(IMAGE_SIZE);
  bool template_initialized = false;
  signal(SIGINT, signal_handler); // Capture Ctrl+C

  // Save results
  std::ofstream real_result_f("results.csv");

  int imageId = 0;
  while (running && !decoder.is_ffmpeg_exit()) {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (decoder.get_frame_without_release()) {
      std::cout << "Got one frame" << std::endl;
      copy_yuv420_from_frame(reinterpret_cast<char *>(img.data()),
                             &decoder.frame_H);

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
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "--duration: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     end_time - start_time)
                     .count()
              << "ms" << std::endl;

    template_initialized = true;
  }

  if (save_result && real_result_f.is_open()) {
    real_result_f.close();
  }
}
