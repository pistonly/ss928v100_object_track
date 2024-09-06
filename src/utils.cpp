#include "utils.hpp"
#include "sample_comm.h"
#include "ss_mpi_sys.h"
#include <iostream>

void copy_yuv420_from_frame(char *yuv420, ot_video_frame_info *frame) {
  td_u32 height = frame->video_frame.height;
  td_u32 width = frame->video_frame.width;
  td_u32 size = height * width * 3 / 2; // 对于YUV420格式，大小为宽*高*1.5

  td_void *frame_data =
      ss_mpi_sys_mmap_cached(frame->video_frame.phys_addr[0], size);
  if (frame_data == NULL) {
    sample_print("mmap failed!\n");
    /* free(tmp); */
    return;
  }

  memcpy(yuv420, frame_data, size);
}

void saveBinaryFile(const std::vector<unsigned char> &data,
                    const std::string &filePath) {
  std::ofstream file(filePath, std::ios::binary);
  if (file.is_open()) {
    file.write(reinterpret_cast<const char *>(data.data()), data.size());
  } else {
    std::cerr << "Unable to open file " << filePath << std::endl;
  }
}

void saveBinaryFile(const std::vector<char> &data,
                    const std::string &filePath) {
  std::ofstream file(filePath, std::ios::binary);
  if (file.is_open()) {
    file.write(data.data(), data.size());
  } else {
    std::cerr << "Unable to open file " << filePath << std::endl;
  }
}

std::vector<std::vector<float>> readCSV(const std::string &filename) {
  std::vector<std::vector<float>> data;
  std::ifstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return data;
  }

  std::string line;
  while (std::getline(file, line)) {
    std::vector<float> row;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ',')) {
      try {
        row.push_back(std::stof(cell));
      } catch (const std::invalid_argument &e) {
        std::cerr << "Error: Invalid float conversion in cell: " << cell
                  << std::endl;
        row.push_back(0.0f); // 或者根据需要处理错误
      }
    }
    data.push_back(row);
  }

  file.close();
  return data;
}
