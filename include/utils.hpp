/**
 * @file utils.hpp
 *
 */
#ifndef UTILS_H
#define UTILS_H

#include <fstream>
#include <stdint.h>
#include <stddef.h>
#include <mutex>
#include <queue>
#include <string>
#include <sys/stat.h>

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

template <typename T>
class ThreadSafeQueue{
private:
  std::queue<T> queue;
  mutable std::mutex mutex;

  public:
  ThreadSafeQueue() {}

  void push(T value) {
    std::lock_guard<std::mutex> lock(mutex);
    queue.push(std::move(value));
  }

  bool pop(T& value) {
    std::lock_guard<std::mutex> lock(mutex);
    if (queue.empty()){
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


struct YoloModelInfo{
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

  void save_info(const std::string& filename) {
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
    for (auto i: v_output_names)
      file << i << ", ";
    file << "]" << std::endl;

    //
    file << "v_output_dims:" << std::endl;
    for (auto v_i : v_output_dims) {
      file << "  " << "- [";
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
                 const std::vector<std::string>& outputnode_names);

#endif
