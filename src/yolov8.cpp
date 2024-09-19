#include "utils.hpp"
#include "yolov8.hpp"
#include "post_process_tools.hpp"
#include <vector>

extern Logger logger;

YOLOV8::YOLOV8(const std::string &modelPath,
               const std::string &output_dir,
               const std::string &aclJSON)
    : NNNYOLOV8(modelPath, aclJSON) {
  char c_output_dir[PATH_MAX];
  if (realpath(output_dir.c_str(), c_output_dir) == NULL) {
    logger.log(ERROR, "Output directory error: ", output_dir);
  }
  m_output_dir = std::string(c_output_dir);

  if (m_output_dir.back() != '/')
    m_output_dir += '/';
  logger.log(INFO, "Output directory is: ", m_output_dir);
  std::vector<size_t> outbuf_size;
  GetOutBufferSize(outbuf_size);
  logger.log(INFO, "out num: ", outbuf_size.size());
  for (size_t i = 0; i < outbuf_size.size(); ++i) {
    logger.log(INFO, "size of output_", i, ": ", outbuf_size[i]);
  }

  GetModelInfo(nullptr, &m_input_h, &m_input_w, nullptr, nullptr, &mv_outputs_dim);
  input_yuv.resize(m_input_h * m_input_w * 1.5);
  for (auto &dim_i : mv_outputs_dim) {
    std::stringstream ss;
    ss << "out dim: " << std::endl;
    for (auto dim_i_j : dim_i) {
      ss << dim_i_j << ", ";
    }
    logger.log(INFO, ss.str());
  }
}

void YOLOV8::set_postprocess_parameters(float conf_thres, float iou_thres,
                                        int max_det) {
  m_conf_thres = conf_thres;
  m_iou_thres = iou_thres;
  m_max_det = max_det;
}

void YOLOV8::set_roi_parameters(int left, int top, float scale) {
  m_topleft.first = left;
  m_topleft.second = top;
  m_scale = m_scale;
}

void YOLOV8::post_process(std::vector<std::vector<std::vector<half>>> &det_bbox,
                          std::vector<std::vector<half>> &det_conf,
                          std::vector<std::vector<half>> &det_cls) {

  std::vector<const char *> vp_outputs;
  Result ret = Device2Host(vp_outputs);
  if (ret != SUCCESS) {
    logger.log(ERROR, "Device2Host error");
    return;
  }

  split_bbox_conf_reduced(vp_outputs, mv_outputs_dim, mvp_bbox, mvp_conf,
                          mvp_cls);

  const int batch_num = mvp_bbox.size();

  const int roi_hw = 32;
  for (int i = 0; i < batch_num; ++i) {
    const int box_num = mv_outputs_dim[0][2];
    const std::vector<const half *> &bbox_batch_i = mvp_bbox[i];
    const std::vector<const half *> &conf_batch_i = mvp_conf[i];
    const std::vector<const half *> &cls_batch_i = mvp_cls[i];
    NMS_bboxTranspose(box_num, bbox_batch_i, conf_batch_i, cls_batch_i,
                      det_bbox[i], det_conf[i], det_cls[i], m_conf_thres,
                      m_iou_thres, m_max_det);

    // change to real location
    for (auto j = 0; i < det_bbox[i].size(); ++j) {
      std::vector<half> &box = det_bbox[i][j];
      box[0] = m_scale * box[0] + m_topleft.first;
      box[1] = m_scale * box[1] + m_topleft.second;
      box[2] = m_scale * box[2] + m_topleft.first;
      box[3] = m_scale * box[3] + m_topleft.second;
    }
  }
}

bool YOLOV8::process_one_image(
    const unsigned char *img, const int imgW, const int imgH,
    std::vector<std::vector<std::vector<half>>> &det_bbox,
    std::vector<std::vector<half>> &det_conf,
    std::vector<std::vector<half>> &det_cls) {
  // cut image roi
  int x0 = m_topleft.first;
  int y0 = m_topleft.second;
  int x1 = x0 + m_input_w;
  int y1 = y0 + m_input_h;

  if (x0 < 0 || y0 < 0 || x1 > imgW || y1 > imgH) {
    logger.log(ERROR, "cut roi out of range!");
    return false;
  }

  int roi_offset = 0;
  int img_offset = y0 * imgW;
  int roi_uv_offset = m_input_h * m_input_w;
  int img_uv_offset = (imgH + y0 * 0.5) * imgW;
  for (int h = y0; h < y1; ++h) {
    for (int w = x0, roi_x = 0; w < x1; ++w, ++roi_x) {
      input_yuv[roi_offset + roi_x] = img[img_offset + w];
      if (h % 2 == 0)
        input_yuv[roi_uv_offset + roi_x] = img[img_uv_offset + w];
    }
    roi_offset += m_input_w;
    img_offset += imgW;
    if (h % 2 == 0) {
      roi_uv_offset += m_input_w;
      img_uv_offset += imgW;
    }
  }

  // host to device
  Host2Device(input_yuv.data(), input_yuv.size());

  // inference
  Execute();

  // postprocess
  post_process(det_bbox, det_conf, det_cls);


}
