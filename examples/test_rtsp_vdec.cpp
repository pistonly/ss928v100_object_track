#include "ffmpeg_vdec_vpss.hpp"
#include "utils.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <string>
#include <vector>

std::atomic<bool> running(true);
void signal_handler(int signum) { running = false; }
#define IMAGE_HEIGHT 2160
#define IMAGE_WIDTH 3840

int main(int argc, char *argv[]) {
  std::string rtsp_url = "rtsp://172.23.24.52:8554/test";
  // initialize ffmpeg_vdec_vpss
  HardwareDecoder decoder(rtsp_url);
  decoder.start_decode();

  const int IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * 1.5;
  td_s32 decoder_flag;

  signal(SIGINT, signal_handler); // capture Ctrl+C
  while (running && !decoder.is_ffmpeg_exit()) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<unsigned char> img(IMAGE_SIZE);

    decoder_flag = decoder.get_frame_without_release();
    if (decoder_flag) {
      std::cout << "here" << std::endl;
      // copy_yuv420_from_frame(reinterpret_cast<char *>(img.data()),
      //                        &decoder.frame_H);
    } else {
      continue;
    }
    decoder.release_frames();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    std::cout << "--duration: " << duration.count() << "ms" << std::endl;
  }
  return 0;
}
