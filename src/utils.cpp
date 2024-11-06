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

#include <sys/stat.h>

#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <time.h>

// 全局日志器实例，初始日志级别为 INFO
Logger logger(INFO);

// static std::map<int, std::vector<int>> camera_id_map = {
//     {11, {0x11, 0x12}}, {12, {0x13, 0x14}}, {13, {0x15, 0x16}},
//     {21, {0x21, 0x22}}, {22, {0x23, 0x24}}, {23, {0x25, 0x26}},
//     {31, {0x31, 0x32}}, {32, {0x33, 0x34}}, {33, {0x35, 0x36}},
//     {100, {0x37}}};

static std::map<int, std::vector<int>> camera_id_map = {
    {11, {1, 2}}, {12, {3, 4}}, {13, {5, 6}},
    {21, {7, 8}}, {22, {9, 10}}, {23, {11, 12}},
    {31, {13, 14}}, {32, {15, 16}}, {33, {17, 18}},
    {100, {19}}};

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

void copy_yuv420_from_frame(char *yuv420, ot_video_frame_info *frame, int yuv_H,
                            int yuv_W, int offsize_H, int offset_W) {
  td_u32 height = frame->video_frame.height;
  td_u32 width = frame->video_frame.width;
  td_u32 offset_size = offsize_H * yuv_W + offset_W;

  if (height > yuv_H) {
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
  memcpy(yuv420 + yuv_canvas_Ysize + offset_size / 2, frame_data + Y_size,
         uv_size);
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
  cameraIds.reserve(2);

  std::string ipAddress = getIPAddressUsingIfconfig();
  std::size_t lastDotPos = ipAddress.find_last_of('.');
  if (lastDotPos == std::string::npos) {
    return;
  }
  std::string lastOctetStr = ipAddress.substr(lastDotPos + 1);
  int key = std::stoi(lastOctetStr);
  if (camera_id_map.find(key) != camera_id_map.end()) {
    const auto &v_ids = camera_id_map[key];
    for (auto id_ : v_ids) {
      cameraIds.push_back(static_cast<uint8_t>(id_));
    }
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

void save_detect_results_csv(
    const std::vector<std::vector<std::vector<half>>> det_bbox,
    const std::vector<std::vector<half>> det_conf,
    const std::vector<std::vector<half>> det_cls, const std::string &out_dir,
    const std::string &filename) {
  std::ofstream outFile(out_dir + filename, std::ios::binary);
  if (!outFile) {
    std::cerr << "Error opening file " << filename << " for writing."
              << std::endl;
    return;
  }

  // only works when batch_num == 0
  const std::vector<std::vector<half>> &det_bbox_0 = det_bbox[0];
  const std::vector<half> &det_conf_0 = det_conf[0];
  const std::vector<half> &det_cls_0 = det_cls[0];

  for (int i=0; i<det_bbox_0.size(); ++i) {
    // xyxy
    for (const auto v: det_bbox_0[i]){
      outFile << v << ",";
    }
    // conf
    outFile << det_conf_0[i];
    // cls
    outFile << det_cls_0[i] << std::endl;
  }
  outFile.close();
  return;
}

bool isAtImageEdge(std::vector<float> tlwh, int threshold, int image_height,
                   int image_width) {
  const float x0 = tlwh[0];
  const float y0 = tlwh[1];
  const float x1 = x0 + tlwh[2];
  const float y1 = y0 + tlwh[3];
  if (x0 < threshold || y0 < threshold || x1 >= (image_width - threshold) ||
      y1 >= (image_height - threshold))
    return true;
  return false;
}

bool isAtImageEdge(std::vector<float> tlwh, int threshold_x, int threshold_y,
                   int image_height, int image_width) {
  const float x0 = tlwh[0];
  const float y0 = tlwh[1];
  const float x1 = x0 + tlwh[2];
  const float y1 = y0 + tlwh[3];
  if (x0 < threshold_x || y0 < threshold_y ||
      x1 >= (image_width - threshold_x) || y1 >= (image_height - threshold_y))
    return true;
  return false;
}

std::string from_pts_to_dirName(unsigned long long framePts) {
  // 将毫秒转换为秒
  time_t timeSec = framePts / 1000;
  tm *ptm = localtime(&timeSec);
  if (ptm == nullptr) {
    std::cerr << "Failed to convert framePts to local time." << std::endl;
    return std::string("_");
  }

  std::ostringstream oss;
  oss << std::setw(4) << (ptm->tm_year + 1900) // 年份，四位
      << std::setw(2) << std::setfill('0') << (ptm->tm_mon + 1) // 月份，两位
      << std::setw(2) << std::setfill('0') << ptm->tm_mday // 日期，两位
      << "_" << std::setw(2) << std::setfill('0') << ptm->tm_hour // 小时，两位
      << std::setw(2) << std::setfill('0') << ptm->tm_min  // 分钟，两位
      << std::setw(2) << std::setfill('0') << ptm->tm_sec; // 秒数，两位

  return oss.str();
}

std::string from_pts_to_strWithMilliseconds(unsigned long long framePts) {
  // 将毫秒转换为秒
  time_t timeSec = framePts / 1000;
  tm *ptm = localtime(&timeSec);
  if (ptm == nullptr) {
    std::cerr << "Failed to convert framePts to local time." << std::endl;
    return std::string("_");
  }

  std::ostringstream oss;
  oss << std::setw(4) << (ptm->tm_year + 1900) // 年份，四位
      << std::setw(2) << std::setfill('0') << (ptm->tm_mon + 1) // 月份，两位
      << std::setw(2) << std::setfill('0') << ptm->tm_mday // 日期，两位
      << "_" << std::setw(2) << std::setfill('0') << ptm->tm_hour // 小时，两位
      << std::setw(2) << std::setfill('0') << ptm->tm_min  // 分钟，两位
      << std::setw(2) << std::setfill('0') << ptm->tm_sec // 秒数，两位
      << std::setw(3) << std::setfill('0') << framePts % 1000; // ms, 3
  return oss.str();
}

bool file_exists(const std::string &name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

bool directory_exists(const std::string &path) {
  struct stat buffer;
  // Check if path exists and is a directory
  return (stat(path.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

bool create_directory(const std::string &path) {
  // 模式0755是目录的典型权限
  return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

std::ofstream create_file_from_pts(const std::string &parent_dir,
                                   const std::string &fileName,
                                   unsigned long long pts) {
  std::string base_dir = parent_dir + "/" + from_pts_to_dirName(pts);
  std::string base_path = base_dir + "/" + fileName;

  try {
    if (directory_exists(base_dir)) {
      // results.csv exists
      int N = 1;
      while (true) {
        std::string new_dir = base_dir + "_" + std::to_string(N);
        if (!directory_exists(new_dir)) {
          // rename current_dir to current_dir_N
          if (std::rename(base_dir.c_str(), new_dir.c_str()) != 0) {
            std::cerr << "Error: Could not rename " << base_dir << " to "
                      << new_dir << std::endl;
            return std::ofstream("create_wrong_file");
          }
          break;
        }
        N++;
      }
    }

    // Create a base_dir and file .
    create_directory(base_dir);

    std::ofstream ofs(base_path.c_str());
    return ofs;
  } catch (const std::exception &e) {
    std::cerr << "Error: Exception occurred: " << e.what() << std::endl;
    return std::ofstream("create_wrong_file");
  }
}

std::ofstream create_file_from_pts(const std::string &parent_dir,
                                   const std::string &fileName) {

  // 获取当前时间
  std::time_t t = std::time(nullptr);
  char time_buffer[20];
  std::strftime(time_buffer, sizeof(time_buffer), "%Y%m%d_%H%M%S",
                std::localtime(&t));
  std::string time_str(time_buffer);

  std::string base_dir = parent_dir + "/" + time_str;
  std::string base_path = base_dir + "/" + fileName;

  try {
    if (directory_exists(base_dir)) {
      // results.csv exists
      int N = 1;
      while (true) {
        std::string new_dir = base_dir + "_" + std::to_string(N);
        if (!directory_exists(new_dir)) {
          // rename current_dir to current_dir_N
          if (std::rename(base_dir.c_str(), new_dir.c_str()) != 0) {
            std::cerr << "Error: Could not rename " << base_dir << " to "
                      << new_dir << std::endl;
            return std::ofstream("create_wrong_file");
          }
          break;
        }
        N++;
      }
    }

    // Create a base_dir and file .
    create_directory(base_dir);

    std::ofstream ofs(base_path.c_str());
    return ofs;
  } catch (const std::exception &e) {
    std::cerr << "Error: Exception occurred: " << e.what() << std::endl;
    return std::ofstream("create_wrong_file");
  }
}

std::ofstream create_file_from_pts(const std::string &parent_dir,
                                   const std::string &fileName,
                                   std::string &real_dir) {

  // 获取当前时间
  std::time_t t = std::time(nullptr);
  char time_buffer[20];
  std::strftime(time_buffer, sizeof(time_buffer), "%Y%m%d_%H%M%S",
                std::localtime(&t));
  std::string time_str(time_buffer);

  std::string base_dir = parent_dir + "/" + time_str;
  std::string base_path = base_dir + "/" + fileName;

  try {
    if (directory_exists(base_dir)) {
      // results.csv exists
      int N = 1;
      while (true) {
        std::string new_dir = base_dir + "_" + std::to_string(N);
        if (!directory_exists(new_dir)) {
          // rename current_dir to current_dir_N
          if (std::rename(base_dir.c_str(), new_dir.c_str()) != 0) {
            std::cerr << "Error: Could not rename " << base_dir << " to "
                      << new_dir << std::endl;
            return std::ofstream("create_wrong_file");
          }
          break;
        }
        N++;
      }
    }

    // Create a base_dir and file .
    create_directory(base_dir);
    real_dir = base_dir + "/";

    std::ofstream ofs(base_path.c_str());
    return ofs;
  } catch (const std::exception &e) {
    std::cerr << "Error: Exception occurred: " << e.what() << std::endl;
    return std::ofstream("create_wrong_file");
  }
}

std::string getCurrentTimeWithMilliseconds() {
    // 获取当前时间点
    auto now = std::chrono::system_clock::now();
    
    // 转换为 time_t 类型
    auto timeT = std::chrono::system_clock::to_time_t(now);
    
    // 转换为毫秒精度
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    
    // 格式化日期和时间
    std::stringstream ss;
    ss << std::put_time(std::localtime(&timeT), "%Y%m%d_%H%M%S") << '_' 
       << std::setw(3) << std::setfill('0') << ms.count();  // 输出毫秒，确保三位数字

    return ss.str();
}
