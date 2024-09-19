#include "nnn_yolov8.hpp"
#include "post_process_tools.hpp"
#include <half.hpp>
#include <string>
#include <vector>

static float default_conf_thres = 0.5;
static float default_iou_thres = 0.6;
static float default_max_det = 300;

class YOLOV8 : public NNNYOLOV8 {
public:
  YOLOV8(const std::string &modelPath, const std::string &output_dir = "./",
         const std::string &aclJSON = "");

  std::string m_output_dir;
  std::pair<int, int> m_topleft;
  float m_scale;

  std::vector<std::vector<size_t>> mv_outputs_dim;
  int m_input_h, m_input_w;
  bool mb_save_results = false;
  std::vector<char> input_yuv;

  // mvp_bbox shape: batch x branch_num x (anchors * 4)
  std::vector<std::vector<const half *>> mvp_bbox;
  // mvp_conf shape: batch x branch_num x anchors
  std::vector<std::vector<const half *>> mvp_conf;
  // mvp_cls shape: batch x branch_num x anchors
  std::vector<std::vector<const half *>> mvp_cls;

  void post_process(std::vector<std::vector<std::vector<half>>> &det_bbox,
                    std::vector<std::vector<half>> &det_conf,
                    std::vector<std::vector<half>> &det_cls);

  bool process_one_image(
      const unsigned char *img, const int imgW, const int imgH,
      std::vector<std::vector<std::vector<half>>> &det_bbox,
      std::vector<std::vector<half>> &det_conf,
      std::vector<std::vector<half>> &det_cls);

  void set_postprocess_parameters(float conf_thres, float iou_thres,
                                  int max_det);
  void set_roi_parameters(int left, int top, float scale);

private:
  float m_conf_thres = default_conf_thres;
  float m_iou_thres = default_iou_thres;
  int m_max_det = default_max_det;
};
