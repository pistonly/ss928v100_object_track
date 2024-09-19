/**
 * @file utils.hpp
 *
 */
#ifndef UTILS_H
#define UTILS_H

#include "ot_common_video.h"
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <sys/stat.h>
#include <vector>

// 定义日志级别
enum LogLevel { DEBUG, INFO, WARNING, ERROR };

// 简单的日志类
class Logger {
public:
  Logger(LogLevel level) : log_level(level) {}

  void setLogLevel(LogLevel level) { log_level = level; }

  template <typename... Args> void log(LogLevel level, Args &&...args) {
    if (level >= log_level) {
      print(level, std::forward<Args>(args)...);
    }
  }

private:
  LogLevel log_level;

  template <typename... Args> void print(LogLevel level, Args &&...args) {
    std::string level_str;
    switch (level) {
    case DEBUG:
      level_str = "[DEBUG]";
      break;
    case INFO:
      level_str = "[INFO]";
      break;
    case WARNING:
      level_str = "[WARNING]";
      break;
    case ERROR:
      level_str = "[ERROR]";
      break;
    }
    std::lock_guard<std::mutex> guard(mtx);
    std::cout << level_str << " ";
    (std::cout << ... << args) << std::endl;
  }

  std::mutex mtx; // 保证多线程环境下日志输出的原子性
};

#define INFO_LOG(fmt, ...) fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__)
#define WARN_LOG(fmt, ...) fprintf(stdout, "[WARN]  " fmt "\n", ##__VA_ARGS__)
#define ERROR_LOG(fmt, ...) fprintf(stdout, "[ERROR] " fmt "\n", ##__VA_ARGS__)

#ifdef _WIN32
#define S_ISREG(m) (((m)&0170000) == (0100000))
#endif

static const int BYTE_BIT_NUM = 8; // 1 byte = 8 bit

typedef enum Result { SUCCESS = 0, FAILED = 1 } Result;

class Utils {
public:
  static void InitData(int8_t *data, size_t dataSize) {
    for (size_t i = 0; i < dataSize; i++) {
      data[i] = 0;
    }
  };
};

template <typename T> class ThreadSafeQueue {
private:
  std::queue<T> queue;
  mutable std::mutex mutex;

public:
  ThreadSafeQueue() {}

  void push(T value) {
    std::lock_guard<std::mutex> lock(mutex);
    queue.push(std::move(value));
  }

  bool pop(T &value) {
    std::lock_guard<std::mutex> lock(mutex);
    if (queue.empty()) {
      return false;
    }
    value = std::move(queue.front());
    queue.pop();
    return true;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex);
    return queue.empty();
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mutex);
    return queue.size();
  }
};

struct YoloModelInfo {
  std::size_t input_size;
  int batch;
  int h;
  int w;
  int cl_num;
  int reg_max;
  std::vector<int> strides;
  std::vector<std::size_t> v_output_sizes;
  std::vector<std::string> v_output_names;
  std::vector<std::vector<size_t>> v_output_dims;

  void save_info(const std::string &filename) {
    std::ofstream file(filename);
    file << "input_size: " << input_size << std::endl;
    file << "batch: " << batch << std::endl;
    file << "h: " << h << std::endl;
    file << "w: " << w << std::endl;
    file << "cl_num: " << cl_num << std::endl;
    file << "reg_max: " << reg_max << std::endl;

    //
    file << "strides: [";
    for (auto i : strides)
      file << i << ",";
    file << "]" << std::endl;
    //
    file << "v_output_sizes: [";
    for (auto i : v_output_sizes)
      file << i << ", ";
    file << "]" << std::endl;

    //
    file << "v_output_names: [";
    for (auto i : v_output_names)
      file << i << ", ";
    file << "]" << std::endl;

    //
    file << "v_output_dims:" << std::endl;
    for (auto v_i : v_output_dims) {
      file << "  "
           << "- [";
      for (auto i : v_i) {
        file << i << ", ";
      }
      file << "]" << std::endl;
    }
    file.close();
    chmod(filename.c_str(), S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  }
};

void get_file_stem(const std::string &filepath, std::string &file_stem);

void save_output(const std::string &output_dir, const std::string output_stem,
                 const std::string &feature,
                 const std::vector<size_t> &output_sizes,
                 const std::vector<std::string> &outputnode_names);

/**
 * @brief copy yuv420 from frame
 * @details Description
 * @param[out] yuv420 Description
 * @param[in] frame Description
 */
void copy_yuv420_from_frame(char *yuv420, ot_video_frame_info *frame);
void saveBinaryFile(const std::vector<unsigned char> &data,
                    const std::string &filePath);
void saveBinaryFile(const std::vector<char> &data, const std::string &filePath);

std::vector<std::vector<float>> readCSV(const std::string &filename);
#endif
