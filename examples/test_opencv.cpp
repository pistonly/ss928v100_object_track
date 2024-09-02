#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
  // 读取图像
  cv::Mat img = cv::imread("../data/dog1_1024_683.jpg");

  // 检查图像是否加载成功
  if (img.empty()) {
    std::cerr << "图像加载失败！" << std::endl;
    return -1;
  }

  // 获取图像的高度和宽度
  int h = img.rows;
  int w = img.cols;

  // 调整图像大小为原来的2倍
  cv::Mat img_large;
  cv::resize(img, img_large, cv::Size(2 * w, 2 * h));

  // 将调整大小后的图像写入文件
  cv::imwrite("../out/test_opencv.jpg", img_large);

  return 0;
}
