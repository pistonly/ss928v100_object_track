#include "nnn_ostrack_callback.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

void saveBinaryFile(const std::vector<unsigned char> data,
                    const std::string filePath) {
  std::ofstream file(filePath, std::ios::binary);

  if (file.is_open()) {
    file.write(reinterpret_cast<const char *>(data.data()), data.size());
  } else {
    std::cerr << "unable to open file" << filePath << std::endl;
  }
}

int main(int argc, char *argv[]) {
  // std::string omPath =
  //     "../models/"
  //     "MyViTrack_ep0100_192_L3_feature_new_3.om";
  // std::string templatePath = "../data/template.bin";
  // std::string targetPath = "../data/target.bin";
  std::string omPath = "../models/MyViTrack_ep0100_192_L3.om";
  std::string imgPath = "../data/frame00897_1575-1453-31-21.bin";

  // 初始化模型
  NNN_Ostrack_Callback ostModel(omPath);

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
  int imageH = 2160;
  int imageW = 3840;
  int x0 = 1575 / 2 * 2;
  int y0 = 1453 / 2 * 2;
  int w = 31 / 2 * 2;
  int h = 21 / 2 * 2;
  float template_factor = 1.6;
  float search_area_factor = 4;
  int template_size = 96;
  int search_size = 192;

  auto t0 = std::chrono::high_resolution_clock::now();
  // get template
  float template_resize_factor;
  int template_crop_x0, template_crop_y0, template_crop_x1, template_crop_y1,
      template_pad_t, template_pad_b, template_pad_l, template_pad_r,
      template_crop_sz;

  sample_target(imageW, imageH, x0, y0, h, w, template_factor, template_size,
                template_resize_factor, template_crop_x0, template_crop_y0,
                template_crop_x1, template_crop_y1, template_pad_t,
                template_pad_b, template_pad_l, template_pad_r,
                template_crop_sz);
  int template_input_size = (template_crop_sz / 16 + 1) * 16;
  std::vector<unsigned char> templateData(
      template_input_size * template_input_size * 1.5, 0);
  yuv_crop(image.data(), imageW, imageH, template_crop_x0, template_crop_y0,
           template_crop_x1, template_crop_y1, template_input_size,
           template_input_size, templateData);
  // debug
  std::cout << "template_crop: " << template_crop_x0 << ", " << template_crop_y0
            << ", " << template_crop_x1 << ", " << template_crop_y1
            << std::endl;
  std::cout << "template size: " << template_input_size << std::endl;
  saveBinaryFile(templateData, "template.bin");
  // set aipp for template
  int crop_w = template_crop_x1 - template_crop_x0,
      crop_h = template_crop_y1 - template_crop_y0;
  ostModel.SetAIPPPSrcSize(template_input_size, template_input_size);
  ostModel.SetAIPPCrop(0, 0, crop_w, crop_h);
  ostModel.SetAIPPResize(crop_w, crop_h, template_size, template_size);
  // TODO:
  ostModel.SetAIPPPadding(0, 0, 0, 0);
  ostModel.SetAIPP(0);

  // get search image
  float target_resize_factor;
  int target_crop_x0, target_crop_y0, target_crop_x1, target_crop_y1,
      target_pad_t, target_pad_b, target_pad_l, target_pad_r, target_crop_sz;

  sample_target(imageW, imageH, x0, y0, h, w, search_area_factor, template_size,
                target_resize_factor, target_crop_x0, target_crop_y0,
                target_crop_x1, target_crop_y1, target_pad_t, target_pad_b,
                target_pad_l, target_pad_r, target_crop_sz);
  int search_input_size = (target_crop_sz / 16 + 1) * 16;
  std::vector<unsigned char> targetData(
      search_input_size * search_input_size * 1.5, 0);
  yuv_crop(image.data(), imageW, imageH, target_crop_x0, target_crop_y0,
           target_crop_x1, target_crop_y1, search_input_size, search_input_size,
           targetData);
  // debug
  std::cout << "target_crop: " << target_crop_x0 << ", " << target_crop_y0
            << ", " << target_crop_x1 << ", " << target_crop_y1 << std::endl;
  std::cout << "target input size: " << search_input_size << std::endl;
  saveBinaryFile(targetData, "target.bin");
  // set aipp for template
  crop_w = target_crop_x1 - target_crop_x0;
  crop_h = target_crop_y1 - target_crop_y0;
  ostModel.SetAIPPPSrcSize(search_input_size, search_input_size);
  ostModel.SetAIPPCrop(0, 0, crop_w, crop_h);
  ostModel.SetAIPPResize(crop_w, crop_h, search_size, search_size);
  // TODO:
  ostModel.SetAIPPPadding(0, 0, 0, 0);
  ostModel.SetAIPP(1);

  // copy to device
  ostModel.Host2Device(0, templateData.data(), templateData.size());
  ostModel.Host2Device(1, targetData.data(), targetData.size());

  // execute
  int run_times = 10;
  // ostModel.Execute();
  ostModel.ExecuteRPN_Async();
  ostModel.SynchronizeStream();

  float x = ostModel.m_outputs_f[0][0] * search_size / template_resize_factor;
  float y = ostModel.m_outputs_f[0][1] * search_size / template_resize_factor;
  w = ostModel.m_outputs_f[0][2] * search_size / template_resize_factor;
  h = ostModel.m_outputs_f[0][3] * search_size / template_resize_factor;
  std::cout << "model result: " << x << ", " << y << ", " << w << ", " << h
            << std::endl;

  x0 = x - w / 2;
  y0 = y - h / 2;
  float x0_real = target_crop_x0 + x0;
  float y0_real = target_crop_y0 + y0;
  std::cout << "target_crop_x0: " << target_crop_x0
            << " target_crop_y0: " << target_crop_y0 << std::endl;
  std::cout << "result: " << x0_real << ", " << y0_real << ", " << w << ", "
            << h << std::endl;

  auto t1 = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
  std::cout << "execute cost: " << duration.count() << " milliseconds"
            << std::endl;

  return 0;
}
