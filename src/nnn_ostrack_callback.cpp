#include "nnn_ostrack_callback.hpp"
#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include "utils.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

volatile size_t NNN_Ostrack_Callback::mg_ostrack_callbackInterval = 0;
volatile size_t NNN_Ostrack_Callback::mg_ostrack_startCallback = 0;
bool NNN_Ostrack_Callback::mg_ostrack_isExit = false;

void static saveBinaryFile(const std::vector<unsigned char> data,
                           const std::string filePath) {
  std::ofstream file(filePath, std::ios::binary);

  if (file.is_open()) {
    file.write(reinterpret_cast<const char *>(data.data()), data.size());
  } else {
    std::cerr << "unable to open file" << filePath << std::endl;
  }
}

NNN_Ostrack_Callback::NNN_Ostrack_Callback(const std::string &modelPath,
                                           float template_factor,
                                           float search_area_factor,
                                           int template_size, int search_size,
                                           const std::string &aclJSON)
    : m_template_factor(template_factor),
      m_search_area_factor(search_area_factor), m_template_size(template_size),
      m_search_size(search_size), m_aclJSON(aclJSON) {
  // resource
  InitResource();

  // load model
  LoadModelFromFile(modelPath);

  // create bufs
  GetModelParams();
  CreateInputBuf();
  CreateOutputBuf();
  GetRealOutputSize();

  RegistCallBackThread();

  // aipp
  mp_aippParam = aclmdlCreateAIPP(1);
  SetAIPP_Glob();
}

NNN_Ostrack_Callback::~NNN_Ostrack_Callback() {
  if (mp_aippParam) {
    aclmdlDestroyAIPP(mp_aippParam);
    mp_aippParam = nullptr;
  }

  UnRegistCallBackThread();

  UnloadModel();
  DestroyModelDesc();
  DestroyInput();
  DestroyOutput();

  //
  DestroyResource();
  if (mpt_td != nullptr)
    delete mpt_td;
}

Result NNN_Ostrack_Callback::InitResource() {
  // ACL init
  aclError ret = aclInit(m_aclJSON.c_str());
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("nnn acl init failed");
    return FAILED;
  }
  INFO_LOG("nnn acl init success");

  // set device
  ret = aclrtSetDevice(m_deviceId);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("nnn acl open device %d failed", m_deviceId);
    return FAILED;
  }
  INFO_LOG("nnn acl open device %d success", m_deviceId);

  // create context (set current)
  ret = aclrtCreateContext(&m_context, m_deviceId);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("nnn create context failed");
    return FAILED;
  }
  INFO_LOG("nnn create context success");

  // create stream
  ret = aclrtCreateStream(&m_stream);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("nnn create stream failed");
    return FAILED;
  }
  INFO_LOG("nnn create stream success");

  // get run mode
  aclrtRunMode runMode;
  ret = aclrtGetRunMode(&runMode);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("acl get run mode failed");
    return FAILED;
  }
  if (runMode != ACL_DEVICE) {
    ERROR_LOG("acl run mode failed");
    return FAILED;
  }
  INFO_LOG("get run mode success");
  return SUCCESS;
}

Result NNN_Ostrack_Callback::LoadModelFromFile(const std::string &modelPath) {
  size_t weightSize, workSize;
  aclError ret = aclmdlQuerySize(modelPath.c_str(), &workSize, &weightSize);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("query model failed, model file is %s", modelPath.c_str());
    return FAILED;
  }

  ret = aclrtMalloc(&m_modelMemPtr, workSize, ACL_MEM_MALLOC_NORMAL_ONLY);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("mallac buffer for mem failed, require size is %lu", workSize);
    return FAILED;
  }

  void *weightPtr{nullptr};
  ret = aclmdlLoadFromFileWithMem(modelPath.c_str(), &m_modelId, m_modelMemPtr,
                                  workSize, weightPtr, weightSize);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("load model from file failed, model file is %s\n",
              modelPath.c_str());
    return FAILED;
  }

  // get model desc
  Result ret_m = CreateModelDesc();
  if (ret_m != SUCCESS) {
    return FAILED;
  }

  mb_loadFlag = true;
  INFO_LOG("load model %s success", modelPath.c_str());
  return SUCCESS;
}

Result NNN_Ostrack_Callback::GetModelParams() {
  m_input_num = aclmdlGetNumInputs(mp_modelDesc);
  aclError ret;

  aclmdlIODims inDims;
  for (int i = 0; i < m_input_num; ++i) {
    ret = aclmdlGetInputDims(mp_modelDesc, i, &inDims);
    if (ret != ACL_SUCCESS) {
      ERROR_LOG("aclmdlGetInputDims error");
      return FAILED;
    }
    std::cout << "input-" << i << ": " << std::endl;
    for (int j = 0; j < inDims.dimCount; ++j) {
      std::cout << inDims.dims[j] << ", ";
    }
    std::cout << std::endl;

    m_input_buffersizes.push_back(0);
  }

  // get m_ouitput_num
  m_output_num = aclmdlGetNumOutputs(mp_modelDesc);

  // initialize m_outputs;
  m_outputs = std::vector<std::vector<char>>(m_output_num);

  for (int i = 0; i < m_output_num; ++i) {
    aclmdlIODims outDims;
    ret = aclmdlGetOutputDims(mp_modelDesc, i, &outDims);
    if (ret != ACL_SUCCESS) {
      ERROR_LOG("aclmdlGetOutputDims error");
      return FAILED;
    }
    std::cout << "output-" << i << ": " << std::endl;
    for (int j = 0; j < outDims.dimCount; ++j) {
      std::cout << outDims.dims[j] << ", ";
    }
    std::cout << std::endl;
  }

  return SUCCESS;
}

Result NNN_Ostrack_Callback::CreateInputBuf(int index,
                                            aclmdlDataset *p_inputds) {
  void *bufPtr = nullptr;
  size_t bufSize = 0;
  size_t bufStride = 0;
  aclmdlIODims inDims;
  aclError ret = GetInputParam(index, bufSize, inDims);
  if (ret != SUCCESS) {
    ERROR_LOG("Error, GetInputParam failed");
    return FAILED;
  }

  if (index < m_input_buffersizes.size())
    m_input_buffersizes[index] = bufSize;

  ret = aclrtMalloc(&bufPtr, bufSize, ACL_MEM_MALLOC_NORMAL_ONLY);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("malloc device buffer failed. size is %zu", bufSize);
    return FAILED;
  }
  Utils::InitData(static_cast<int8_t *>(bufPtr), bufSize);

  aclDataBuffer *dataBuf = aclCreateDataBuffer(bufPtr, bufSize);
  if (dataBuf == nullptr) {
    ERROR_LOG("can't create data buffer, create buffer failed");
    aclrtFree(bufPtr);
    return FAILED;
  }

  ret = aclmdlAddDatasetBuffer(p_inputds, dataBuf);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("add input dataset buffer failed");
    aclDestroyDataBuffer(dataBuf);
    dataBuf = nullptr;
    return FAILED;
  }

  INFO_LOG("create input buffer SUCCESS");
  return SUCCESS;
}

Result NNN_Ostrack_Callback::CreateInputBuf() {
  mp_input = aclmdlCreateDataset();
  if (mp_input == nullptr) {
    ERROR_LOG("can't create dataset, create input failed");
    return FAILED;
  }

  for (int inputIndex = 0; inputIndex < m_input_num; ++inputIndex) {
    Result ret = CreateInputBuf(inputIndex, mp_input);
    if (ret != ACL_SUCCESS) {
      ERROR_LOG("CreateInputBuf failed");
      return FAILED;
    }
  }
  return SUCCESS;
}

Result NNN_Ostrack_Callback::CreateOutputBuf() {
  return CreateOutputBuf(mp_output, true);
}

Result NNN_Ostrack_Callback::CreateOutputBuf(aclmdlDataset *&p_outputds,
                                             bool isFirst) {
  p_outputds = aclmdlCreateDataset();
  if (p_outputds == nullptr) {
    ERROR_LOG("can't create dataset, create output failed!");
    return FAILED;
  }

  if (isFirst)
    mv_outputBuffer_sizes.clear();
  for (size_t i = 0; i < m_output_num; ++i) {
    size_t bufferSize = aclmdlGetOutputSizeByIndex(mp_modelDesc, i);
    if (bufferSize == 0) {
      ERROR_LOG("Error, output size is %lu.", bufferSize);
      return FAILED;
    }
    if (isFirst) {
      mv_outputBuffer_sizes.push_back(bufferSize);
      std::cout << "output-" << i << " size: " << bufferSize << std::endl;
    }

    void *outputBuffer = nullptr;
    aclError ret =
        aclrtMalloc(&outputBuffer, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) {
      ERROR_LOG("can't malloc buffer, size is %zu, create output failed",
                bufferSize);
      return FAILED;
    }
    Utils::InitData(static_cast<int8_t *>(outputBuffer), bufferSize);
    aclDataBuffer *outputData = aclCreateDataBuffer(outputBuffer, bufferSize);
    if (outputData == nullptr) {
      ERROR_LOG("can't create data buffer, create output failed");
      aclrtFree(outputBuffer);
      return FAILED;
    }

    ret = aclmdlAddDatasetBuffer(p_outputds, outputData);
    if (ret != ACL_SUCCESS) {
      ERROR_LOG("can't add data buffer, create output failed");
      aclrtFree(outputBuffer);
      aclDestroyDataBuffer(outputData);
      return FAILED;
    }
  }
  INFO_LOG("create model output success");
  return SUCCESS;
}

void NNN_Ostrack_Callback::GetRealOutputSize() {
  for (int i = 0; i < m_output_num; ++i) {
    aclmdlIODims outDims;
    aclmdlGetOutputDims(mp_modelDesc, i, &outDims);
    std::size_t output_size = 1;
    std::vector<std::size_t> output_size_i;
    for (int j = 0; j < outDims.dimCount; ++j) {
      output_size *= outDims.dims[j];
      output_size_i.push_back(outDims.dims[j]);
    }
    std::size_t dataSize = GetOutputDataSize(i);
    mv_output_sizes.push_back(output_size * dataSize);
    const char *output_name_i = aclmdlGetOutputNameByIndex(mp_modelDesc, i);
    mv_output_names.push_back(std::string(output_name_i));
    mvv_output_dims.push_back(output_size_i);
  }
}

Result NNN_Ostrack_Callback::RegistCallBackThread() {
  mg_ostrack_isExit = false;
  mg_ostrack_callbackInterval = 1;
  mpt_td = new std::thread(ProcessCallback, m_context, &mg_ostrack_isExit);
  Result ret = SubscribeReport(*mpt_td, m_tid);
  if (ret != SUCCESS) {
    ERROR_LOG("subscribe report failed");
    return FAILED;
  }
  return SUCCESS;
}

Result NNN_Ostrack_Callback::UnRegistCallBackThread() {
  if (mpt_td != nullptr) {
    aclError ret = UnsubscribeReport(*mpt_td, m_tid);
    delete mpt_td;
    mpt_td = nullptr;

    if (ret != ACL_SUCCESS) {
      ERROR_LOG("nnn acl unRegist failed");
      return FAILED;
    }
  }
  return SUCCESS;
}

void NNN_Ostrack_Callback::UnloadModel() {
  if (!mb_loadFlag) {
    WARN_LOG("no model had been loaded, unload failed!");
    return;
  }

  aclError ret = aclmdlUnload(m_modelId);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("unload model failed, modelId is %u", m_modelId);
  }

  if (mp_modelDesc != nullptr) {
    aclmdlDestroyDesc(mp_modelDesc);
    mp_modelDesc = nullptr;
  }

  if (m_modelMemPtr != nullptr) {
    aclrtFree(m_modelMemPtr);
    m_modelMemPtr = nullptr;
  }

  mb_loadFlag = false;
  INFO_LOG("unload model success, modelId is %u", m_modelId);
}

void NNN_Ostrack_Callback::DestroyModelDesc() {
  if (mp_modelDesc != nullptr) {
    aclmdlDestroyDesc(mp_modelDesc);
    mp_modelDesc = nullptr;
  }
}

void NNN_Ostrack_Callback::DestroyInput() { DestroyInput(mp_input); }

void NNN_Ostrack_Callback::DestroyInput(aclmdlDataset *p_inputds) {
  if (p_inputds == nullptr) {
    return;
  }

  for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(p_inputds); ++i) {
    aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(p_inputds, i);
    void *tmp = aclGetDataBufferAddr(dataBuffer);
    aclrtFree(tmp);
    aclDestroyDataBuffer(dataBuffer);
  }

  aclmdlDestroyDataset(p_inputds);
  p_inputds = nullptr;
}

void NNN_Ostrack_Callback::DestroyOutput() { DestroyOutput(mp_output); }

void NNN_Ostrack_Callback::DestroyOutput(aclmdlDataset *p_outputds) {
  if (p_outputds == nullptr) {
    return;
  }

  for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(p_outputds); ++i) {
    aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(p_outputds, i);
    void *tmp = aclGetDataBufferAddr(dataBuffer);
    aclrtFree(tmp);
    aclDestroyDataBuffer(dataBuffer);
  }
  aclmdlDestroyDataset(p_outputds);
  p_outputds = nullptr;
}

void NNN_Ostrack_Callback::DestroyResource() {
  aclError ret;
  if (m_stream != nullptr) {
    ret = aclrtDestroyStream(m_stream);
    if (ret != ACL_SUCCESS) {
      ERROR_LOG("destroy stream failed");
    }
    m_stream = nullptr;
  }
  INFO_LOG("end to destroy stream");

  if (m_context != nullptr) {
    ret = aclrtDestroyContext(m_context);
    if (ret != ACL_SUCCESS) {
      ERROR_LOG("destroy context failed");
    }
    m_context = nullptr;
  }
  INFO_LOG("end to destroy context");

  ret = aclrtResetDevice(m_deviceId);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("reset device failed");
  }
  INFO_LOG("end to reset device is %d", m_deviceId);

  ret = aclFinalize();
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("finalize nnn acl failed");
  }
  INFO_LOG("end to finalize nnn acl");
}

Result NNN_Ostrack_Callback::Host2Device(const int inputIndex,
                                         const void *inputdata,
                                         const size_t input_size) {
  // check input_size;
  if (input_size > m_input_buffersizes[inputIndex]) {
    std::cout << input_size << ", " << m_input_buffersizes[inputIndex]
              << std::endl;
    ERROR_LOG("input_size should <= buffersize: %d, %d", input_size,
              m_input_buffersizes[inputIndex]);
    return FAILED;
  }

  aclDataBuffer *buf = aclmdlGetDatasetBuffer(mp_input, inputIndex);
  if (buf == nullptr) {
    ERROR_LOG("get data buffer from input_dataset failed");
    return FAILED;
  }
  void *data = aclGetDataBufferAddr(buf);
  return Host2Device(data, inputdata, input_size);
}

Result NNN_Ostrack_Callback::Host2Device(void *dev_data, const void *host_data,
                                         const size_t input_size) {
  memcpy(dev_data, host_data, input_size);
  return SUCCESS;
}

Result NNN_Ostrack_Callback::Execute() {
  auto t0 = std::chrono::high_resolution_clock::now();
  aclrtSetCurrentContext(m_context);
  aclError ret = aclmdlExecute(m_modelId, mp_input, mp_output);
  auto t1 = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
  std::cout << "execute cost: " << duration.count() << " milliseconds"
            << std::endl;
  if (ret != ACL_SUCCESS) {
    return FAILED;
  } else {
    return SUCCESS;
  }
}

Result NNN_Ostrack_Callback::ExecuteRPN_Async() {
  std::cout << "execute async" << std::endl;
  aclrtSetCurrentContext(m_context);
  aclError ret = aclmdlExecuteAsync(m_modelId, mp_input, mp_output, m_stream);
  std::cout << "execute async success" << std::endl;

  if (ret != ACL_SUCCESS) {
    ERROR_LOG("nnn execute async start error");
    return FAILED;
  }

  if (mg_ostrack_callbackInterval != 0) {
    return ExecuteCallback();
  } else
    return SUCCESS;
}

Result NNN_Ostrack_Callback::SynchronizeStream() {
  aclError ret = aclrtSynchronizeStream(m_stream);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("aclrtSynchronizeStream failed, errorCode is %d",
              static_cast<int32_t>(ret));
    return FAILED;
  }

  return SUCCESS;
}

Result
NNN_Ostrack_Callback::Device2Host(std::vector<std::vector<char>> &outputs) {
  for (int i = 0; i < m_output_num; ++i) {
    aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(mp_output, i);
    if (dataBuffer == nullptr) {
      ERROR_LOG("output[%d] dataBuffer nullptr invalid", i);
      return FAILED;
    }
    const char *outData = (const char *)aclGetDataBufferAddr(dataBuffer);
    size_t outSize = mv_outputBuffer_sizes[i];

    // clear outputs[i];
    outputs[i].resize(outSize, 0);
    // std::fill(outputs[i].begin(), outputs[i].end(), 0);

    size_t index = 0;
    std::copy(outData, outData + outSize, outputs[i].begin());
  }
  return SUCCESS;
}

void NNN_Ostrack_Callback::StaticCallbackFunc(void *data) {
  std::cout << "static callback" << std::endl;
  NNN_Ostrack_Callback *instance = static_cast<NNN_Ostrack_Callback *>(data);
  instance->CallbackFunc(data);
}

Result NNN_Ostrack_Callback::ExecuteCallback() {
  std::cout << "execute callback" << std::endl;
  aclError ret = aclrtLaunchCallback(StaticCallbackFunc, (void *)this,
                                     ACL_CALLBACK_BLOCK, m_stream);
  std::cout << "execute callback success" << std::endl;
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("launch callback failed, error code: %d",
              static_cast<int32_t>(ret));
    return FAILED;
  }
  return SUCCESS;
}

void NNN_Ostrack_Callback::ProcessCallback(aclrtContext context, void *arg) {
  aclrtSetCurrentContext(context);
  while (mg_ostrack_callbackInterval != 0) {
    while (mg_ostrack_startCallback == 1) {
      // timeout value is 100ms
      (void)aclrtProcessReport(100);
      if (*(static_cast<bool *>(arg)) == true) {
        return;
      }
    }
  }
}

void NNN_Ostrack_Callback::CallbackFunc(void *data) {
  std::cout << "callback from ostrack" << std::endl;
  Result ret = Device2Host(m_outputs);
  if (ret != SUCCESS) {
    std::cerr << "Device2host error" << std::endl;
  }
}

Result NNN_Ostrack_Callback::CreateModelDesc() {
  mp_modelDesc = aclmdlCreateDesc();
  if (mp_modelDesc == nullptr) {
    ERROR_LOG("create model description failed!");
    return FAILED;
  }

  aclError ret = aclmdlGetDesc(mp_modelDesc, m_modelId);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("get model description failed");
    return FAILED;
  }

  INFO_LOG("create model description success");

  return SUCCESS;
}

Result NNN_Ostrack_Callback::GetInputParam(int index, size_t &bufSize,
                                           aclmdlIODims &dims) const {
  aclError ret = aclmdlGetInputDims(mp_modelDesc, index, &dims);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("acl_mdl_get_input_dims error!");
    return FAILED;
  }
  bufSize = aclmdlGetInputSizeByIndex(mp_modelDesc, index);
  if (bufSize == 0) {
    ERROR_LOG("acl_mdl_get_input_size_by_index error!");
    return FAILED;
  }

  // print info
  INFO_LOG("input index[%d] info: bufferSize[%zu]", index, bufSize);
  // print out input dims
  INFO_LOG("input dims:");
  std::stringstream ss;
  for (size_t loop = 0; loop < dims.dimCount; loop++)
    ss << dims.dims[loop] << ", ";
  ss << std::endl;
  std::cout << ss.str() << std::endl;
  return SUCCESS;
}

Result NNN_Ostrack_Callback::SubscribeReport(std::thread &td, uint64_t &tid) {
  // subscribe report
  std::ostringstream oss;
  oss << td.get_id();
  tid = std::stoull(oss.str());
  int aclRt = aclrtSubscribeReport(tid, m_stream);
  if (aclRt != ACL_SUCCESS) {
    mg_ostrack_isExit = true;
    td.join();
    ERROR_LOG("acl subscribe report failed");
    return FAILED;
  }
  mg_ostrack_startCallback = 1;
  INFO_LOG("subscribe report success");
  return SUCCESS;
}

Result NNN_Ostrack_Callback::UnsubscribeReport(std::thread &td, uint64_t tid) {
  mg_ostrack_isExit = true;
  td.join();
  int ret = aclrtUnSubscribeReport(tid, m_stream);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("nnn acl unsubscribe report failed");
    return FAILED;
  }
  return SUCCESS;
}

size_t NNN_Ostrack_Callback::GetOutputDataSize(int index) const {
  return aclmdlGetOutputSizeByIndex(mp_modelDesc, index);
}

Result NNN_Ostrack_Callback::SetAIPPCsc() {
  int8_t csc_switch = 1;
  int16_t matrix_r0c0 = 298;
  int16_t matrix_r0c1 = 0;
  int16_t matrix_r0c2 = 409;
  int16_t matrix_r1c0 = 298;
  int16_t matrix_r1c1 = -100;
  int16_t matrix_r1c2 = -208;
  int16_t matrix_r2c0 = 298;
  int16_t matrix_r2c1 = 516;
  int16_t matrix_r2c2 = 0;

  aclError ret = aclmdlSetAIPPCscParams(
      mp_aippParam, csc_switch, matrix_r0c0, matrix_r0c1, matrix_r0c2,
      matrix_r1c0, matrix_r1c1, matrix_r1c2, matrix_r2c0, matrix_r2c1,
      matrix_r2c2, 16, 128, 128, 16, 128, 128);
  if (ret != ACL_SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Result NNN_Ostrack_Callback::SetAIPPCrop(int32_t start_x, int32_t start_y,
                                         int32_t crop_w, int32_t crop_h,
                                         int8_t crop) {
  uint64_t batch_idx = 0;
  aclError ret = aclmdlSetAIPPCropParams(mp_aippParam, crop, start_x, start_y,
                                         crop_w, crop_h, batch_idx);
  if (ret != ACL_SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Result NNN_Ostrack_Callback::SetAIPPResize(int32_t input_w, int32_t input_h,
                                           int32_t output_w, int32_t output_h,
                                           int8_t resize) {
  uint64_t batch_idx = 0;
  aclError ret = aclmdlSetAIPPScfParams(mp_aippParam, resize, input_w, input_h,
                                        output_w, output_h, batch_idx);
  if (ret != ACL_SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Result NNN_Ostrack_Callback::SetAIPPPadding(int32_t top, int32_t bottom,
                                            int32_t left, int32_t right,
                                            int8_t padding) {
  uint64_t batch_idx = 0;
  aclError ret = aclmdlSetAIPPPaddingParams(mp_aippParam, padding, top, bottom,
                                            left, right, batch_idx);
  if (ret != ACL_SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Result NNN_Ostrack_Callback::SetAIPPPSrcSize(int32_t w, int32_t h) {
  aclError ret = aclmdlSetAIPPSrcImageSize(mp_aippParam, w, h);
  if (ret != ACL_SUCCESS) {
    std::cout << "aclmdlSetAIPPSrcImageSize error " << std::endl;
    return FAILED;
  }
  return SUCCESS;
}

Result NNN_Ostrack_Callback::SetAIPP_Glob() {
  aclError ret = aclmdlSetAIPPInputFormat(mp_aippParam, ACL_YUV420SP_U8);

  if (ret != ACL_SUCCESS) {
    return FAILED;
  }

  Result res = SetAIPPCsc();
  if (res != SUCCESS) {
    return FAILED;
  }

  res = SetAIPPMean(124, 116, 104, 0);
  if (res != SUCCESS) {
    return FAILED;
  }

  res = SetAIPPVar(0.01712475, 0.017507, 0.01742919, 1.0);
  if (res != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Result NNN_Ostrack_Callback::GetAIPPInfo() {
  aclAippInfo aippInfo;
  aclError ret = aclmdlGetFirstAippInfo(m_modelId, 0, &aippInfo);
  std::cout << "AIPP INFO:" << std::endl;
  std::cout << "srcImageSizeW: " << aippInfo.srcImageSizeW << std::endl;
  std::cout << "cropSwitch: " << aippInfo.cropSwitch << std::endl;
  std::cout << "resizeSwitch: " << aippInfo.resizeSwitch << std::endl;
}

Result NNN_Ostrack_Callback::SetAIPP(size_t inputIdx) {
  aclmdlInputAippType aippType;
  size_t aippIndex;
  aclError ret = aclmdlGetAippType(m_modelId, inputIdx, &aippType, &aippIndex);

  if (ret != ACL_SUCCESS) {
    std::cout << "aclmdlGetAippType failed" << std::endl;
    return FAILED;
  }

  ret = aclmdlSetInputAIPP(m_modelId, mp_input, aippIndex, mp_aippParam);
  if (ret != ACL_SUCCESS) {
    std::cout << "set aipp failed" << std::endl;
    return FAILED;
  }
  return SUCCESS;
}

Result NNN_Ostrack_Callback::SetAIPPMean(int16_t m_ch0, int16_t m_ch1,
                                         int16_t m_ch2, int16_t m_ch3) {
  uint64_t batch_idx = 0;
  aclError ret = aclmdlSetAIPPDtcPixelMean(mp_aippParam, m_ch0, m_ch1, m_ch2,
                                           m_ch3, batch_idx);

  if (ret != ACL_SUCCESS) {
    std::cout << "aclmdlSetAIPPDtcPixelMean failed" << std::endl;
    return FAILED;
  }
  return SUCCESS;
}

Result NNN_Ostrack_Callback::SetAIPPVar(float var_ch0, float var_ch1,
                                        float var_ch2, float var_ch3) {
  uint64_t batch_idx = 0;
  aclError ret = aclmdlSetAIPPPixelVarReci(mp_aippParam, var_ch0, var_ch1,
                                           var_ch2, var_ch3, batch_idx);

  if (ret != ACL_SUCCESS) {
    std::cout << "aclmdlSetAIPPDtcPixelMean failed" << std::endl;
    return FAILED;
  }
  return SUCCESS;
}

void sample_target(int image_w, int image_h, int x0, int y0, int h, int w,
                   float search_area_factor, int output_sz,
                   float &resize_factor, int &crop_x0, int &crop_y0,
                   int &crop_x1, int &crop_y1, int &pad_t, int &pad_b,
                   int &pad_l, int &pad_r, int &crop_sz) {
  crop_sz = (int)(std::ceil(std::sqrt(w * h) * search_area_factor)) / 2 * 2;
  float cx = x0 + 0.5 * w;
  float cy = y0 + 0.5 * h;
  crop_x0 = (int)(std::round(cx - crop_sz * 0.5)) / 2 * 2;
  crop_y0 = (int)(std::round(cy - crop_sz * 0.5)) / 2 * 2;
  crop_x1 = crop_x0 + crop_sz;
  crop_y1 = crop_y0 + crop_sz;

  pad_l = std::max(0, -crop_x0);
  pad_r = std::max(crop_x1 - image_w + 1, 0);
  pad_t = std::max(0, -crop_y0);
  pad_b = std::max(crop_y1 - image_h + 1, 0);

  // update crop
  crop_x0 = crop_x0 + pad_l;
  crop_y0 = crop_y0 + pad_t;
  crop_x1 = crop_x1 - pad_r;
  crop_y1 = crop_y1 - pad_b;

  resize_factor = (float)output_sz / crop_sz;
}

void yuv_crop(const unsigned char *img, const int imgW, const int imgH,
              const int crop_x0, const int crop_y0, const int crop_x1,
              const int crop_y1, const int roiW, const int roiH,
              std::vector<unsigned char> &roi) {
  if (roi.size() < (roiH * roiW * 1.5)) {
    std::cerr << "roi size error! " << std::endl;
    return;
  }

  if (crop_x0 > imgW || crop_y0 > imgH) {
    std::cerr << "crop start error" << std::endl;
    return;
  }

  int crop_w = crop_x1 - crop_x0;
  int crop_h = crop_y1 - crop_y0;
  if (crop_w > roiW)
    crop_w = roiW;
  if (crop_h > roiH)
    crop_h = roiH;

  const int img_Y_size = imgH * imgW;
  const int roi_Y_size = roiH * roiW;
  int img_Y_row_head = crop_y0 * imgW, roi_Y_row_head = 0;
  int img_uv_row_head = (imgH + crop_y0 * 0.5) * imgW,
      roi_uv_row_head = roiH * roiW;
  for (auto h = 0; h < crop_h; ++h) {
    for (auto w = 0; w < crop_w; ++w) {
      // Y channel row
      roi[roi_Y_row_head + w] = img[img_Y_row_head + w + crop_x0];
      // uv channel. uv channel has the same w and Y channel
      if (h % 2 == 0) {
        roi[roi_uv_row_head + w] = img[img_uv_row_head + w + crop_x0];
      }
    }
    img_Y_row_head += imgW;
    roi_Y_row_head += roiW;
    if (h % 2 == 0) {
      img_uv_row_head += imgW;
      roi_uv_row_head += roiW;
    }
  }
}

Result NNN_Ostrack_Callback::preprocess(
    const unsigned char *img, const int imgW, const int imgH, int x0, int y0,
    int w, int h, float &target_resize_factor, int &target_crop_x0,
    int &target_crop_y0, bool updateTemplate) {
  m_imageId++;
  Result ret;
  // x y w h should be even;
  x0 = x0 / 2 * 2;
  y0 = y0 / 2 * 2;
  w = w / 2 * 2;
  h = h / 2 * 2;

  // get template
  if (updateTemplate) {
    float template_resize_factor;
    int template_crop_x0, template_crop_y0, template_crop_x1, template_crop_y1,
        template_pad_t, template_pad_b, template_pad_l, template_pad_r,
        template_crop_sz;

    sample_target(imgW, imgH, x0, y0, h, w, m_template_factor, m_template_size,
                  template_resize_factor, template_crop_x0, template_crop_y0,
                  template_crop_x1, template_crop_y1, template_pad_t,
                  template_pad_b, template_pad_l, template_pad_r,
                  template_crop_sz);
    int template_input_size = (template_crop_sz / 16 + 1) * 16;
    std::vector<unsigned char> templateData(
        template_input_size * template_input_size * 1.5, 0);
    yuv_crop(img, imgW, imgH, template_crop_x0, template_crop_y0,
             template_crop_x1, template_crop_y1, template_input_size,
             template_input_size, templateData);
    m_templateData.assign(templateData.begin(), templateData.end());

    // set aipp for template
    int crop_w = template_crop_x1 - template_crop_x0,
        crop_h = template_crop_y1 - template_crop_y0;
    SetAIPPPSrcSize(template_input_size, template_input_size);
    SetAIPPCrop(0, 0, crop_w, crop_h);
    SetAIPPResize(crop_w, crop_h, m_template_size, m_template_size);
    // TODO:
    SetAIPPPadding(0, 0, 0, 0);
    ret = SetAIPP(0);
    if (ret != SUCCESS)
      return ret;
    // copy to device
    ret = Host2Device(0, templateData.data(), templateData.size());
    if (ret != SUCCESS)
      return ret;
  }

  // get search image
  int target_crop_x1, target_crop_y1, target_pad_t, target_pad_b, target_pad_l,
      target_pad_r, target_crop_sz;

  sample_target(imgW, imgH, x0, y0, h, w, m_search_area_factor, m_search_size,
                target_resize_factor, target_crop_x0, target_crop_y0,
                target_crop_x1, target_crop_y1, target_pad_t, target_pad_b,
                target_pad_l, target_pad_r, target_crop_sz);
  int search_input_size = (target_crop_sz / 16 + 1) * 16;
  std::vector<unsigned char> targetData(
      search_input_size * search_input_size * 1.5, 0);
  yuv_crop(img, imgW, imgH, target_crop_x0, target_crop_y0, target_crop_x1,
           target_crop_y1, search_input_size, search_input_size, targetData);

  // set aipp for search area
  int crop_w = target_crop_x1 - target_crop_x0;
  int crop_h = target_crop_y1 - target_crop_y0;
  SetAIPPPSrcSize(search_input_size, search_input_size);
  SetAIPPCrop(0, 0, crop_w, crop_h);
  SetAIPPResize(crop_w, crop_h, m_search_size, m_search_size);
  // TODO:
  SetAIPPPadding(0, 0, 0, 0);
  ret = SetAIPP(1);
  if (ret != SUCCESS)
    return ret;
  // copy to device
  ret = Host2Device(1, targetData.data(), targetData.size());
  if (ret != SUCCESS)
    return ret;
  return SUCCESS;
}

Result NNN_Ostrack_Callback::postprocess(const int search_crop_x0,
                                         const int search_crop_y0,
                                         const float search_resize_factor,
                                         std::vector<float> &tlwh) {
  // Check if output size is 16 (4 * 4-byte float)
  if (mv_outputBuffer_sizes[0] != 16) {
    std::cerr << "This postprocess only works for float32" << std::endl;
    return FAILED;
  }

  // Read and convert the first 4 floats from m_outputs
  std::array<float, 4> outputs;
  for (int i = 0; i < 4; ++i) {
    outputs[i] = *reinterpret_cast<float *>(&m_outputs[0][4 * i]);
  }

  // Precompute resize factor
  const float resize_factor_inv = m_search_size / search_resize_factor;

  // Calculate coordinates and dimensions
  const float x = outputs[0] * resize_factor_inv;
  const float y = outputs[1] * resize_factor_inv;
  const float w = outputs[2] * resize_factor_inv;
  const float h = outputs[3] * resize_factor_inv;
  const float x0 = x - w / 2;
  const float y0 = y - h / 2;

  // Add search_crop offset
  const float real_x0 = x0 + search_crop_x0;
  const float real_y0 = y0 + search_crop_y0;

  // Assign results to tlwh
  tlwh.assign({real_x0, real_y0, w, h});

  return SUCCESS;
}

