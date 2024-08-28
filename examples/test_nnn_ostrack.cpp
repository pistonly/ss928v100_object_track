#include "nnn_ostrack_callback.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>


int main(int argc, char *argv[]) {
  // std::string omPath =
  //     "../models/"
  //     "MyViTrack_ep0100_192_L3_feature_new_3.om";
  // std::string templatePath = "../data/template.bin";
  // std::string targetPath = "../data/target.bin";
  std::string omPath = "../models/MyViTrack_ep0100_192_L3.om";
  std::string templatePath = "../data/template_YUV.bin";
  std::string targetPath = "../data/target_YUV.bin";

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

  // 读取 template 和 target 数据
  std::vector<uint8_t> templateData = readBinaryFile(templatePath);
  std::vector<uint8_t> targetData = readBinaryFile(targetPath);

  // 检查数据是否成功读取
  if (templateData.empty() || targetData.empty()) {
    std::cerr << "模板或目标数据读取失败。" << std::endl;
    return -1;
    }

    // copy to device
    ostModel.Host2Device(0, templateData.data(), templateData.size());
    ostModel.Host2Device(1, targetData.data(), targetData.size());

    // execute
    auto t0 = std::chrono::high_resolution_clock::now();
    int run_times = 10;
    for (int _t=0; _t<run_times; ++_t){
      // ostModel.Execute();
      ostModel.ExecuteRPN_Async();
      // std::cout << "here0" << std::endl;
      ostModel.SynchronizeStream();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    std::cout << "execute cost: " << duration.count() << " milliseconds"
              << std::endl;

    return 0;
}

