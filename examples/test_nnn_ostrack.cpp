#include "nnn_ostrack_callback.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


int main(int argc, char *argv[]) {
  // std::string omPath =
  //     "../models/"
  //     "MyViTrack_ep0100_192_L3_feature_new_3.om";
  // std::string templatePath = "../data/template.bin";
  // std::string targetPath = "../data/target.bin";
  std::string omPath = "../models/MyViTrack_ep0100_192_L3.om";
  std::string imgPath = "../data/frame00897_1575-1453-31-21.bin";

  // 初始化模型
  float template_factor = 1.6;
  float search_area_factor = 4;
  int template_size = 96;
  int search_size = 192;
  NNN_Ostrack_Callback ostModel(omPath, template_factor, search_area_factor,
                                template_size, search_size);

  // 读取二进制文件内容的函数
  auto readBinaryFile =
      [](const std::string &filePath) -> std::vector<uint8_t> {
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file) {
      std::cerr << "无法打开文件: " << filePath << std::endl;
      return {};
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
      std::cerr << "读取文件失败: " << filePath << std::endl;
      return {};
    }
    return buffer;
  };

  // 读取 image
  std::vector<uint8_t> image = readBinaryFile(imgPath);

  auto t0 = std::chrono::high_resolution_clock::now();
  // preprocess
  int imageH = 2160;
  int imageW = 3840;
  int x0 = 1575;
  int y0 = 1453;
  int w = 31;
  int h = 21;
  float search_resize_factor;
  int search_crop_x0, search_crop_y0;
  ostModel.preprocess(image.data(), imageW, imageH, x0, y0, w, h,
                      search_resize_factor, search_crop_x0, search_crop_y0, true);

  // execute
  int run_times = 10;
  // ostModel.Execute();
  ostModel.ExecuteRPN_Async();
  ostModel.SynchronizeStream();

  float x = ostModel.m_outputs_f[0][0] * search_size / search_resize_factor;
  float y = ostModel.m_outputs_f[0][1] * search_size / search_resize_factor;
  w = ostModel.m_outputs_f[0][2] * search_size / search_resize_factor;
  h = ostModel.m_outputs_f[0][3] * search_size / search_resize_factor;
  std::cout << "model result: " << x << ", " << y << ", " << w << ", " << h
            << std::endl;

  x0 = x - w / 2;
  y0 = y - h / 2;
  float x0_real = search_crop_x0 + x0;
  float y0_real = search_crop_y0 + y0;
  std::cout << "target_crop_x0: " << search_crop_x0
            << " target_crop_y0: " << search_crop_y0 << std::endl;
  std::cout << "result: " << x0_real << ", " << y0_real << ", " << w << ", "
            << h << std::endl;

  auto t1 = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
  std::cout << "execute cost: " << duration.count() << " milliseconds"
            << std::endl;

  return 0;
}
