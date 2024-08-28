#include "nnn_ostrack_callback.hpp"
#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include "utils.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>

volatile size_t NNN_Ostrack_Callback::mg_ostrack_callbackInterval = 0;
volatile size_t NNN_Ostrack_Callback::mg_ostrack_startCallback = 0;
bool NNN_Ostrack_Callback::mg_ostrack_isExit = false;

NNN_Ostrack_Callback::NNN_Ostrack_Callback(const std::string &modelPath,
                                           const std::string &aclJSON)
    : m_aclJSON(aclJSON) {
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
}

NNN_Ostrack_Callback::~NNN_Ostrack_Callback() {
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
  std::vector<float> outputs(4);
  for (int i=0; i<4; ++i){
    outputs[i] = *reinterpret_cast<float*>(&m_outputs[0][4 * i]);
  }

  std::cout << "outputs: ";
  for (auto o: outputs){
    std::cout << o << ", ";
  }
  std::cout << std::endl;

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
