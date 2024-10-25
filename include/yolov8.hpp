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

  /**
   * @brief Performs post-processing on the model's output.
   *
   * This function extracts bounding boxes, confidence scores, and class
   * predictions from the model's output and applies Non-Maximum Suppression
   * (NMS) to filter the results.
   *
   * @param det_bbox Output parameter for detected bounding boxes.
   * @param det_conf Output parameter for detection confidence scores.
   * @param det_cls Output parameter for detected classes.
   */
  void post_process(std::vector<std::vector<std::vector<half>>> &det_bbox,
                    std::vector<std::vector<half>> &det_conf,
                    std::vector<std::vector<half>> &det_cls);

  /**
   * @brief Processes a single input image and performs inference.
   *
   * This function extracts the ROI from the input image, transfers data to the
   * device, runs inference, and performs post-processing.
   *
   * @param img Pointer to the input image data.
   * @param imgW Width of the input image.
   * @param imgH Height of the input image.
   * @param det_bbox Output parameter for detected bounding boxes.
   * @param det_conf Output parameter for detection confidence scores.
   * @param det_cls Output parameter for detected classes.
   * @return `true` if processing was successful, `false` otherwise.
   */
  bool process_one_image(const unsigned char *img, const int imgW,
                         const int imgH,
                         std::vector<std::vector<std::vector<half>>> &det_bbox,
                         std::vector<std::vector<half>> &det_conf,
                         std::vector<std::vector<half>> &det_cls);

  /**
   * @brief Processes a single input image and performs inference.
   *
   * This function transfers data to the
   * device, runs inference, and performs post-processing.
   *
   */
  bool process_one_image(const std::vector<unsigned char> img,
                         std::vector<std::vector<std::vector<half>>> &det_bbox,
                         std::vector<std::vector<half>> &det_conf,
                         std::vector<std::vector<half>> &det_cls);

  /**
   * @brief Sets parameters for the post-processing step.
   *
   * @param conf_thres Confidence threshold for predictions.
   * @param iou_thres Intersection over Union (IoU) threshold for Non-Maximum
   * Suppression (NMS).
   * @param max_det Maximum number of detections to keep after NMS.
   */
  void set_postprocess_parameters(float conf_thres, float iou_thres,
                                  int max_det);
  /**
   * @brief Sets the Region of Interest (ROI) parameters.
   *
   * @param left Left coordinate of the ROI.
   * @param top Top coordinate of the ROI.
   * @param scale Scale factor for ROI transformation.
   */
  void set_roi_parameters(int left, int top, float scale);

private:
  float m_conf_thres = default_conf_thres;
  float m_iou_thres = default_iou_thres;
  int m_max_det = default_max_det;
};
