/**
 * @file      nnn_ostrack_callback.hpp
 * @brief     Header of
 * @date      Tue Aug 27 10:14:40 2024
 * @author    liuyang
 * @copyright BSD-3-Clause
 *
 * This module
 */

#ifndef NNN_OSTRACK_CALLBACK_HPP
#define NNN_OSTRACK_CALLBACK_HPP

#include "acl/acl_mdl.h"
#include "utils.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

/**
 * @brief Summary
 * @details Description
 * @param[in] x0 Description
 * @param[in] y0 Description
 * @param[in] h Description
 * @param[in] w Description
 * @param[in] search_area_factor Description
 * @param[in] output_sz The size of output
 * @param[in] resize_factor The factor of resize
 * @param[out] crop_x0 Description
 * @param[out] crop_y0 Description
 * @param[out] crop_x1 Description
 * @param[out] crop_y1 Description
 * @param[out] pad_t Description
 * @param[out] pad_b Description
 * @param[out] pad_l Description
 * @param[out] pad_r Description
 */
void sample_target(int image_w, int image_h, int x0, int y0, int h, int w,
                   float search_area_factor, int output_sz,
                   float &resize_factor, int &crop_x0, int &crop_y0,
                   int &crop_x1, int &crop_y1, int &pad_t, int &pad_b,
                   int &pad_l, int &pad_r, int &crop_sz);

void yuv_crop(const unsigned char *img, const int imgW, const int imgH,
              const int crop_x0, const int crop_y0, const int crop_x1,
              const int crop_y1, const int roiW, const int roiH,
              std::vector<unsigned char> &roi);

class NNN_Ostrack_Callback {
public:
  NNN_Ostrack_Callback(const std::string &modelPath, float template_factor,
                       float search_area_factor, int template_size,
                       int search_size, const std::string &aclJSON = "");
  ~NNN_Ostrack_Callback();

  Result preprocess(const unsigned char *img, const int imgW, const int imgH,
                    int x0, int y0, int w, int h, float &target_resize_factor,
                    int &target_crop_x0, int &target_crop_y0,
                    bool updateTemplate = false);

  Result Host2Device(const int inputIdex, const void *inputdata,
                     const size_t input_size);
  Result ExecuteRPN_Async();
  Result Execute();

  /**
   * @brief synchronize stream
   */
  Result SynchronizeStream();

  std::vector<std::vector<char>> m_outputs;
  std::vector<std::vector<float>> m_outputs_f;

  /**
   * @brief: cp data from Device to Host
   */
  void CallbackFunc(void *data);
  Result Device2Host(std::vector<std::vector<char>> &outputs);

  /**
   * @name AIPP
   * Description
   * @{
   */
  Result SetAIPPCrop(int32_t start_x, int32_t start_y, int32_t crop_w,
                     int32_t crop_h, int8_t crop = 1);

  Result SetAIPPResize(int32_t input_w, int32_t input_h, int32_t output_w,
                       int32_t output_h, int8_t resize = 1);

  Result SetAIPPPadding(int32_t top, int32_t bottom, int32_t left,
                        int32_t right, int8_t padding = 1);

  Result SetAIPPPSrcSize(int32_t w, int32_t h);

  Result SetAIPPMean(int16_t m_ch0, int16_t m_ch1, int16_t m_ch2,
                     int16_t m_ch3);
  Result SetAIPPVar(float var_ch0, float var_ch1, float var_ch2, float var_ch3);

  Result GetAIPPInfo();
  Result SetAIPP(size_t inputIdx);
  /**
   * @}
   */

  int GetCurrentImageId() {return m_imageId;};

private:
  int m_imageId{0};
  float m_template_factor, m_search_area_factor;
  int m_template_size, m_search_size;
  volatile static size_t
      mg_ostrack_callbackInterval;                 // launch callback interval
  volatile static size_t mg_ostrack_startCallback; // start callback

  static bool mg_ostrack_isExit;
  std::string m_aclJSON;

  int m_batch;
  int m_input_num;  /**< input number of model */
  int m_output_num; /**< output number from model */

  std::vector<unsigned char> m_templateData;

  std::vector<size_t> mv_output_sizes;
  std::vector<std::string> mv_output_names;
  std::vector<std::vector<size_t>> mvv_output_dims;
  std::vector<size_t> mv_outputBuffer_sizes;

  int m_deviceId{0};
  aclrtContext m_context{nullptr};
  aclrtStream m_stream;
  bool mb_loadFlag{false}; /**< whether model is loaded */

  uint64_t m_tid{0};
  std::thread *mpt_td{nullptr};

  uint32_t m_modelId{0};
  void *m_modelMemPtr{nullptr};
  aclmdlDesc *mp_modelDesc{nullptr};
  aclmdlDataset *mp_input{nullptr};
  aclmdlDataset *mp_output{nullptr};

  Result LoadModelFromFile(const std::string &modelPath);
  Result CreateModelDesc();
  /**
   * @brief initialize resources
   * @return result
   */
  Result InitResource();

  /**
   * @brief get model parameters: m_batch, m_h, m_w, m_cl_num, m_reg_max,
   * m_output_num, m_v_strides
   */
  Result GetModelParams();

  /**
   * @brief: create buffers for input.
   */
  std::vector<size_t> m_input_buffersizes;
  Result CreateInputBuf(int index, aclmdlDataset *p_inputds);
  Result CreateInputBuf();

  /**
   * @brief: create buffers for output.
   */
  Result CreateOutputBuf();
  Result CreateOutputBuf(aclmdlDataset *&p_outputds, bool isFirst);
  void GetRealOutputSize();

  /**
   * @name tools
   * Description
   * @{
   */
  Result GetInputParam(int index, size_t &bufSie, aclmdlIODims &dims) const;
  size_t GetInputDataSize(int index) const;
  size_t GetOutputDataSize(int index) const;

  /**
   * @}
   */

  /**
   * @brief unload model, release m_modelMemPtr
   */
  void UnloadModel();
  void DestroyInput();
  void DestroyInput(aclmdlDataset *p_inputds);
  void DestroyOutput();
  void DestroyOutput(aclmdlDataset *p_outputds);
  void DestroyModelDesc();

  /**
   * @brief release stream, context, device
   */
  void DestroyResource();

  Result Host2Device(void *dev_data, const void *host_data,
                     const size_t input_size);

  /**
   * @name callback group
   * callback after executeasync finished.
   * @{
   */
  Result RegistCallBackThread();
  Result UnRegistCallBackThread();
  Result SubscribeReport(std::thread &td, uint64_t &tid);
  Result UnsubscribeReport(std::thread &td, uint64_t tid);
  static void ProcessCallback(aclrtContext context, void *arg);
  Result ExecuteCallback();

  static void StaticCallbackFunc(void *data);
  /**
   * @}
   */

  /**
   * @name AIPP
   * Description
   * @{
   */

  aclmdlAIPP *mp_aippParam;
  Result SetAIPP_Glob();

  /**< aapp yuv420sp->RGB */
  Result SetAIPPCsc();
  /**
   * @}
   */
};

#endif
