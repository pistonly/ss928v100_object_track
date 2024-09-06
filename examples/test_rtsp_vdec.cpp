#include "ffmpeg_vdec_vpss.hpp"
#include "utils.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

std::atomic<bool> running(true);
void signal_handler(int signum) { running = false; }
#define IMAGE_HEIGHT 2160
#define IMAGE_WIDTH 3840

int main(int argc, char *argv[]) {
  std::string rtsp_url = "rtsp://172.23.24.52:8554/test";
  // initialize ffmpeg_vdec_vpss
  HardwareDecoder decoder(rtsp_url, true);
  decoder.start_decode();

  const int IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * 1.5;
  td_s32 decoder_flag;

  signal(SIGINT, signal_handler); // capture Ctrl+C
  int frame_id = 0;
  while (running && !decoder.is_ffmpeg_exit()) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<unsigned char> img(IMAGE_SIZE);

    decoder_flag = decoder.get_frame_without_release();
    if (decoder_flag) {
      copy_yuv420_from_frame(reinterpret_cast<char *>(img.data()),
                             &decoder.frame_H);
      // save image
      std::stringstream ss;
      ss << "frame_" << frame_id++ << ".bin";
      std::string frame_name = ss.str();
      saveBinaryFile(img, frame_name);
      std::cout << frame_name << std::endl;

      // simulate executing
      // std::this_thread::sleep_for(std::chrono::milliseconds(4));
    } else {
      break;
    }
    decoder.release_frames();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    std::cout << "--duration: " << duration.count() << "ms" << std::endl;
  }
  return 0;
}
