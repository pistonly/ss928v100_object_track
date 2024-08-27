/**
 * @file      nnn_ostrack_callback.hpp
 * @brief     Header of
 * @date      Tue Aug 27 10:14:40 2024
 * @author    liuyang
 * @copyright BSD-3-Clause
 * 
 * This module
 */

#ifndef NNN_OSTRACK_CALLBACK
#define NNN_OSTRACK_CALLBACK
#include "acl/acl_mdl.h"
#include <cstddef>
#include <string>
#include <thread>
#include <vector>

class NNN_OSTRACK_CALLBACK{
public:
  NNN_OSTRACK_CALLBACK(const std::string &modelPath,
                       const std::string &aclJSON="");
  ~NNN_OSTRACK_CALLBACK();

  Result Host2Device(const void *inputdata, const size_t input_size);
  Result ExecuteRPN_Async(std::vector<std::vector<char>> &outputs);

  /**
   * @brief synchronize stream
   */
  Result SynchronizeStream();

  /**
   * @brief: cp data from Device to Host
   */
  virtual void CallbackFunc(void *data);
  Result Device2Host(std::vector<std::vector<char>> &outputs);

private:
  volatile static size_t mg_ostrack_callbackInterval; // launch callback interval
  valatile static size_t mg_ostrack_startCallback;  // start callback

  static bool mg_ostrack_isExist;
  std::string m_aclJSON;

  int m_batch;

  std::vector<size_t> mv_output_sizes;
  std::vector<std::string> mv_output_names;
  std::vector<std::vector<size_t>> mvv_output_dims;
  std::vector<size_t> mv_outputBuffer_sizes;

  int m_deviceId{0};
  aclrtContext m_context{nullptr};
  aclrtStream m_stream;
  bool mb_loadFlag{false}; /**< whether model is loaded */

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
  std::size_t m_image_buffersize{0};
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

}
#endif
