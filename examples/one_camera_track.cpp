#include "Strack.hpp"
#include "ffmpeg_vdec_vpss.hpp"
#include "nnn_ostrack_callback.hpp"
#include "utils.hpp"
#include <atomic>
#include <string>
#include <vector>

std::atomic<bool> running(true);
void signal_handler(int signum) { running = false; }

int main(int argc, char *argv[]) {
  // ost model params
  std::string omPath = "../models/MyViTrack_ep0100_192_L3.om";
  float template_factor = 1.6;
  float search_area_factor = 4;
  int template_size = 96;
  int search_size = 192;

  // vdec source
  std::string rtsp_url = "rtsp://172.23.24.52:8554/test";
  const int imageH = 2160;
  const int imageW = 3840;
  const int IMAGE_SIZE = imageH * imageW * 1.5;

  // initialize ffmpeg_vdec_vpss
  HardwareDecoder decoder(rtsp_url);
  decoder.start_decode();

  // NNN_Ostrack_Callback ostModel(omPath, template_factor, search_area_factor,
  //                               template_size, search_size);

  // init bbox
  int x0 = 1575;
  int y0 = 1453;
  int w = 31;
  int h = 21;

  // tracks
  std::vector<float> tlwh = {(float)x0, (float)y0, (float)w, (float)h};
  STrack one_tracker(tlwh);
  one_tracker.activate();
  std::unordered_map<int, STrack> trackers = {{0, one_tracker}};

  std::vector<unsigned char> img(IMAGE_SIZE);
  td_s32 decoder_flag;

  while (running && !decoder.is_ffmpeg_exit()) {

    decoder_flag = decoder.get_frame_without_release();
    if (decoder_flag) {
      std::cout << "get one frame" << std::endl;
      copy_yuv420_from_frame(reinterpret_cast<char *>(img.data()),
                             &decoder.frame_H);
      // track
      STrack::multi_predict(trackers);
      for (auto it = trackers.begin(); it != trackers.end(); ++it) {
        auto &tlwh = (it->second).tlwh;
        int t_x0 = (int)tlwh[0];
        int t_y0 = (int)tlwh[1];
        int t_w = (int)tlwh[2];
        int t_h = (int)tlwh[3];
        float search_resize_factor;
        int search_crop_x0, search_crop_y0;
        std::cout << "tlwh: " << t_x0 << ", " << t_y0 << ", " << t_w << ", "
                  << t_h << std::endl;
        // ostModel.preprocess(img.data(), imageW, imageH, t_x0, t_y0, t_w, t_h,
        //                     search_resize_factor, search_crop_x0,
        //                     search_crop_y0, true);
        // ostModel.ExecuteRPN_Async();
        // ostModel.SynchronizeStream();

        // // measure results
        // float measure_x =
        //     ostModel.m_outputs_f[0][0] * search_size / search_resize_factor;
        // float measure_y =
        //     ostModel.m_outputs_f[0][1] * search_size / search_resize_factor;
        // float measure_w = ostModel.m_outputs_f[0][2] * search_size / search_resize_factor;
        // float measure_h = ostModel.m_outputs_f[0][3] * search_size / search_resize_factor;
        // // std::cout << "model result: " << x << ", " << y << ", " << w << ", "
        // //           << h << std::endl;

        // float measure_x0 = measure_x - measure_w / 2;
        // float measure_y0 = measure_y - measure_h / 2;
        // float measure_x0_real = search_crop_x0 + measure_x0;
        // float measure_y0_real = search_crop_y0 + measure_y0;
        // std::vector<float> tlwh_new = {measure_x0_real, measure_y0_real, measure_w, measure_w};
        // (it->second).update(tlwh_new);
      }
    } else {
      continue;
    }
    decoder.release_frames();
  }
}
