#include "utils.hpp"
#include "sample_comm.h"
#include "ss_mpi_sys.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

// 全局日志器实例，初始日志级别为 INFO
Logger logger(INFO);

static std::map<int, std::vector<int>> camera_id_map = {
    {11, {0x11, 0x12}},
    {12, {0x13, 0x14}},
    {13, {0x15, 0x16}},
    {21, {0x21, 0x22}},
    {22, {0x23, 0x24}},
    {23, {0x25, 0x26}},
    {31, {0x31, 0x32}},
    {32, {0x33, 0x34}},
    {33, {0x35, 0x36}},
    {100, {0x64, 0x64}}    };

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

void copy_yuv420_from_frame(char *yuv420, ot_video_frame_info *frame, int yuv_H, int yuv_W, int offsize_H, int offset_W) {
  td_u32 height = frame->video_frame.height;
  td_u32 width = frame->video_frame.width;
  td_u32 offset_size = offsize_H * yuv_W + offset_W;

  if (height > yuv_H){
    std::cerr << "yuv_H should not less than frame_H" << std::endl;
  }
  if (width > yuv_W) {
    std::cerr << "yuv_W should not less than frame_W" << std::endl;
  }

  td_u32 Y_size = height * width;
  td_u32 uv_size = height * width / 2;
  td_u32 frame_data_size = Y_size + uv_size;

  td_void *frame_data =
      ss_mpi_sys_mmap_cached(frame->video_frame.phys_addr[0], frame_data_size);
  if (frame_data == NULL) {
    sample_print("mmap failed!\n");
    /* free(tmp); */
    return;
  }

  // copy Y channel
  memcpy(yuv420 + offset_size, frame_data, Y_size);

  // copy uv channel
  td_u32 yuv_canvas_Ysize = yuv_H * yuv_W;
  memcpy(yuv420 + yuv_canvas_Ysize + offset_size / 2, frame_data + Y_size, uv_size);
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

std::string getIPAddressUsingIfconfig() {
  FILE *pipe = popen("ifconfig", "r");
  if (!pipe) {
    std::cerr << "popen failed" << std::endl;
    return "";
  }

  std::stringstream buffer;
  char ch;
  while (fread(&ch, 1, 1, pipe) > 0) {
    buffer.put(ch);
  }

  std::string output = buffer.str();
  pclose(pipe);

  std::size_t inetPos = output.find("inet ");
  if (inetPos == std::string::npos) {
    return "";
  }

  std::size_t addrStart = output.find_first_not_of(" \t", inetPos + 5);
  std::size_t addrEnd = output.find_first_of(" \t\n", addrStart);
  return output.substr(addrStart, addrEnd - addrStart);
}


uint8_t getCameraId() {
  std::string ipAddress = getIPAddressUsingIfconfig();
  std::size_t lastDotPos = ipAddress.find_last_of('.');
  if (lastDotPos == std::string::npos) {
    return -1;
  }
  std::string lastOctetStr = ipAddress.substr(lastDotPos + 1);
  return static_cast<uint8_t>(std::stoi(lastOctetStr));
}

void getCameraId_pair(std::vector<uint8_t> &cameraIds) {
  cameraIds.clear();
  cameraIds.assign(2, 0);

  std::string ipAddress = getIPAddressUsingIfconfig();
  std::size_t lastDotPos = ipAddress.find_last_of('.');
  if (lastDotPos == std::string::npos) {
    return;
  }
  std::string lastOctetStr = ipAddress.substr(lastDotPos + 1);
  int key = std::stoi(lastOctetStr);
  if (camera_id_map.find(key) != camera_id_map.end()) {
    const auto &val = camera_id_map[key];
    cameraIds[0] = static_cast<uint8_t>(val[0]);
    cameraIds[1] = static_cast<uint8_t>(val[1]);
  } else {
    std::cout << "get cameraId error" << std::endl;
  }
}

// tracker_res: left, top, w, h, confidence=0, tracker_id
// or: x0, y0, x1, y1, confidence=0, tracker_id
std::vector<char>
serialize_track_results(const std::vector<std::vector<float>> &tracker_res,
                        const uint8_t cameraId, const uint64_t timestamp) {
  std::vector<char> buffer;
  // total size
  unsigned int len = 13 + 24 * tracker_res.size();
  buffer.insert(buffer.end(), reinterpret_cast<const char *>(&len),
                reinterpret_cast<const char *>(&len) + sizeof(len));
  // cameraId
  buffer.insert(buffer.end(), reinterpret_cast<const char *>(&cameraId),
                reinterpret_cast<const char *>(&cameraId) + sizeof(cameraId));

  // time stamp
  buffer.insert(buffer.end(), reinterpret_cast<const char *>(&timestamp),
                reinterpret_cast<const char *>(&timestamp) + sizeof(timestamp));

  // bboxes
  for (auto &dec : tracker_res) {
    assert(dec.size() == 6);
    for (float value : dec) {
      buffer.insert(buffer.end(), reinterpret_cast<const char *>(&value),
                    reinterpret_cast<const char *>(&value) + sizeof(value));
    }
  }

  assert(sizeof(len) + sizeof(cameraId) + sizeof(timestamp) == 13);
  return buffer;
}

void send_track_result(int sock,
                       const std::vector<std::vector<float>> &tracker_res,
                       uint8_t cameraId, uint64_t ts) {
  std::vector<char> serialized_data =
      serialize_track_results(tracker_res, cameraId, ts);
  send(sock, serialized_data.data(), serialized_data.size(), 0);
}

void save_one_track_result_csv(
    std::ofstream &outFile, const std::vector<std::vector<float>> &tracker_res,
    uint8_t cameraId, uint64_t ts) {
  for (auto &dec : tracker_res) {
    int i = 0;
    outFile << static_cast<int>(cameraId) << "," << ts << ",";
    for (; i < dec.size() - 1; ++i) {
      outFile << dec[i] << ",";
    }
    outFile << dec[i] << std::endl;
  }
}

bool isAtImageEdge(std::vector<float> tlwh, int threshold, int image_height, int image_width) {
  const float x0 = tlwh[0];
  const float y0 = tlwh[1];
  const float x1 = x0 + tlwh[2];
  const float y1 = y0 + tlwh[3];
  if (x0 < threshold || y0 < threshold || x1 >= (image_width - threshold) ||
      y1 >= (image_height - threshold))
    return true;
  return false;
}
