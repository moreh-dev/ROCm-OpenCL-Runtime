/* Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <hip/hip_runtime.h>
#include <spdlog/spdlog.h>
#include "platform/object.hpp"
#include "platform/runtime.hpp"
#include "platform/sampler.hpp"
#include "device/device.hpp"
#include "cl_common.hpp"
#include "hip_internal.hpp"
#include "hip_platform.hpp"
#include "hip_event.hpp"

std::vector<hip::Device*> g_devices;

namespace hip {

thread_local Device* g_device = nullptr;
thread_local std::stack<Device*> g_ctxtStack;
thread_local hipError_t g_lastError = hipSuccess;
Device* host_device = nullptr;
volatile bool initialized_ = false;

/// MOREH_COPIED_UNTOUCHED
Device* getCurrentDevice() {
  return g_device;
}

/// MOREH_COPIED_UNTOUCHED
void setCurrentDevice(unsigned int index) {
  assert(index<g_devices.size());
  g_device = g_devices[index];
  uint32_t preferredNumaNode = g_device->devices()[0]->getPreferredNumaNode();
  amd::Os::setPreferredNumaNode(preferredNumaNode);
}

/// MOREH_COPIED_UNTOUCHED
amd::HostQueue* getNullStream(amd::Context& ctx) {
  for (auto& it : g_devices) {
    if (it->asContext() == &ctx) {
      return it->NullStream();
    }
  }
  // If it's a pure SVM allocation with system memory access, then it shouldn't matter which device
  // runtime selects by default
  if (hip::host_device->asContext() == &ctx) {
    // Return current...
    return getNullStream();
  }
  return nullptr;
}

/// MOREH_COPIED_TOUCHED
amd::HostQueue* getNullStream() {
  Device* device = getCurrentDevice();
  if (device == nullptr) {
    ShouldNotReachHere();
    return nullptr;
  }

  amd::Device* amdDevice = device->devices()[0];
  if (amdDevice->hostQueues().empty()) {
    ShouldNotReachHere();
    return nullptr;
  }

  return amdDevice->hostQueues()[0];
}

/// MOREH_COPIED_TOUCHED
amd::HostQueue* getQueue(hipStream_t stream) {
  if (stream == nullptr) {
    return getNullStream();
  }
  else {
    cl_command_queue command_queue = reinterpret_cast<cl_command_queue>(stream);
    amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
    return queue;
  }
}

/// MOREH_COPIED_TOUCHED
bool isValid(hipStream_t& stream) {
  // TODO
  return true;
}

};

using namespace hip;

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// Context Management
///////////////////////////////////////////////////////////////////////////////
hipError_t hipInit(unsigned int flags) {
  if (hip::initialized())
    return hipSuccess;

  spdlog::info("Initializing Moreh OpenCL/HIP Runtime (version: {})",
      MOREH_GIT_VERSION);

  if (!amd::Runtime::initialized()) {
    spdlog::error("hipInit should be called after initializing OpenCL runtime!");
    exit(0);
  }

  const std::vector<amd::Device*>& devices =
    amd::Device::getDevices(CL_DEVICE_TYPE_GPU, false);

  for (unsigned int i = 0; i < devices.size(); ++i) {
    amd::Context* context = &(devices[i]->context());
    if (!context)
      return hipErrorInvalidDevice;

    auto device = new Device(context, devices[i], i);
    if ((device == nullptr) || !device->Create()) {
      return hipErrorUnknown;
    }
    g_devices.push_back(device);
  }

  amd::Context* hContext = new amd::Context(devices, amd::Context::Info());
  if (!hContext)
    return hipErrorUnknown;

  if (CL_SUCCESS != hContext->create(nullptr)) {
    hContext->release();
    ShouldNotReachHere();
  }
  host_device = new Device(hContext, NULL, -1);

  PlatformState::instance().init();

  initialized_ = true;

  return hipSuccess;
}

/// MOREH_COPIED_UNTOUCHED
hipError_t hipCtxCreate(hipCtx_t *ctx, unsigned int flags, hipDevice_t device) {
  HIP_INIT_API(hipCtxCreate, ctx, flags, device);

  if (static_cast<size_t>(device) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *ctx = reinterpret_cast<hipCtx_t>(g_devices[device]);

  // Increment ref count for device primary context
  g_devices[device]->retain();
  g_ctxtStack.push(g_devices[device]);

  HIP_RETURN(hipSuccess);
}

/// MOREH_COPIED_UNTOUCHED
hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
  HIP_INIT_API(hipCtxSetCurrent, ctx);

  if (ctx == nullptr) {
    if(!g_ctxtStack.empty()) {
      g_ctxtStack.pop();
    }
  } else {
    hip::g_device = reinterpret_cast<hip::Device*>(ctx);
    if(!g_ctxtStack.empty()) {
      g_ctxtStack.pop();
    }
    g_ctxtStack.push(hip::getCurrentDevice());
  }

  HIP_RETURN(hipSuccess);
}

/// MOREH_COPIED_UNTOUCHED
hipError_t hipCtxGetCurrent(hipCtx_t* ctx) {
  HIP_INIT_API(hipCtxGetCurrent, ctx);

  *ctx = reinterpret_cast<hipCtx_t>(hip::getCurrentDevice());

  HIP_RETURN(hipSuccess);
}

/// MOREH_COPIED_UNTOUCHED
hipError_t hipCtxDestroy(hipCtx_t ctx) {
  HIP_INIT_API(hipCtxDestroy, ctx);

  hip::Device* dev = reinterpret_cast<hip::Device*>(ctx);
  if (dev == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Need to remove the ctx of calling thread if its the top one
  if (!g_ctxtStack.empty() && g_ctxtStack.top() == dev) {
    g_ctxtStack.pop();
  }

  // Remove context from global context list
  for (unsigned int i = 0; i < g_devices.size(); i++) {
    if (g_devices[i] == dev) {
      // Decrement ref count for device primary context
      dev->release();
    }
  }

  HIP_RETURN(hipSuccess);
}

/// MOREH_COPIED_UNTOUCHED
hipError_t hipCtxPopCurrent(hipCtx_t* ctx) {
  HIP_INIT_API(hipCtxPopCurrent, ctx);

  hip::Device** dev = reinterpret_cast<hip::Device**>(ctx);
  if (!g_ctxtStack.empty()) {
    if (dev != nullptr) {
      *dev = g_ctxtStack.top();
    }
    g_ctxtStack.pop();
  } else {
    DevLogError("Context Stack empty \n");
    HIP_RETURN(hipErrorInvalidContext);
  }

  HIP_RETURN(hipSuccess);
}

/// MOREH_COPIED_UNTOUCHED
hipError_t hipCtxPushCurrent(hipCtx_t ctx) {
  HIP_INIT_API(hipCtxPushCurrent, ctx);

  hip::Device* dev = reinterpret_cast<hip::Device*>(ctx);
  if (dev == nullptr) {
    HIP_RETURN(hipErrorInvalidContext);
  }

  hip::g_device = dev;
  g_ctxtStack.push(hip::getCurrentDevice());

  HIP_RETURN(hipSuccess);
}

/// MOREH_COPIED_UNTOUCHED
hipError_t hipDriverGetVersion(int* driverVersion) {
  HIP_INIT_API_NO_RETURN(hipDriverGetVersion, driverVersion);

  if (!driverVersion) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // HIP_VERSION = HIP_VERSION_MAJOR*100 + HIP_MINOR_VERSION
  *driverVersion = HIP_VERSION;

  HIP_RETURN(hipSuccess);
}
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// Stream Management
///////////////////////////////////////////////////////////////////////////////
/// MOREH_COPIED_TOUCHED
int Stream::DeviceId(ihipStream_t* hStream) {
  if (hStream == nullptr)
    return ihipGetDevice();
  
  cl_command_queue command_queue = reinterpret_cast<cl_command_queue>(hStream);
  amd::Device* device = &as_amd(command_queue)->device();

  return device->index();
}

/// MOREH_COPIED_UNTOUCHED
void WaitThenDecrementSignal(hipStream_t stream, hipError_t status, void* user_data) {
  CallbackData* data =  reinterpret_cast<CallbackData*>(user_data);
  int offset = data->previous_read_index % IPC_SIGNALS_PER_EVENT;
  while (data->shmem->read_index < data->previous_read_index + IPC_SIGNALS_PER_EVENT &&
         data->shmem->signal[offset] != 0) {
    amd::Os::sleep(1);
  }
  delete data;
}

/// MOREH_COPIED_TOUCHED
hipStream_t getPerThreadDefaultStream() {
  // Moreh OpenCL/HIP does not support per-thread default stream
  return nullptr;
}

/// MOREH_COPIED_UNTOUCHED
void CL_CALLBACK ihipStreamCallback(cl_event event, cl_int command_exec_status, void* user_data) {
  hipError_t status = hipSuccess;
  StreamCallback* cbo = reinterpret_cast<StreamCallback*>(user_data);
  cbo->callBack_(cbo->stream_, status, cbo->userData_);
  cbo->command_->release();
  delete cbo;
}

/// MOREH_COPIED_TOUCHED
hipError_t ihipStreamCreate(hipStream_t* stream,
                           unsigned int flags, hip::Stream::Priority priority,
                           const std::vector<uint32_t>& cuMask = {}) {
  if (flags != hipStreamDefault && flags != hipStreamNonBlocking) {
    return hipErrorInvalidValue;
  }

  amd::CommandQueue::Priority p;
  switch (priority) {
    case hip::Stream::Priority::High:
      p = amd::CommandQueue::Priority::High;
      break;
    case hip::Stream::Priority::Low:
      p = amd::CommandQueue::Priority::Low;
      break;
    case hip::Stream::Priority::Normal:
    default:
      p = amd::CommandQueue::Priority::Normal;
      break;
  }

  amd::Context& context = *hip::getCurrentDevice()->asContext();
  amd::Device& device = *context.devices()[0];

  amd::HostQueue* queue = new amd::HostQueue(context, device, 0,
      amd::CommandQueue::RealTimeDisabled, p, cuMask);

  bool result = (queue != nullptr) ? queue->create() : false;
  if (result) {
    amd::ScopedLock lock(device.queueLock());
    device.addHostQueue(queue);
  } else if (queue != nullptr) {
    queue->release();
    delete queue;
    return hipErrorOutOfMemory;
  }

  *stream = reinterpret_cast<hipStream_t>(as_cl(queue));

  return hipSuccess;
}

/// MOREH_COPIED_UNTOUCHED
hipError_t hipStreamCreate(hipStream_t *stream) {
  HIP_INIT_API(hipStreamCreate, stream);

  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(ihipStreamCreate(stream, hipStreamDefault, hip::Stream::Priority::Normal), *stream);
}

/// MOREH_COPIED_UNTOUCHED
hipError_t hipStreamCreateWithFlags(hipStream_t *stream, unsigned int flags) {
  HIP_INIT_API(hipStreamCreateWithFlags, stream, flags);

  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(ihipStreamCreate(stream, flags, hip::Stream::Priority::Normal), *stream);
}

/// MOREH_COPIED_UNTOUCHED
hipError_t hipExtStreamCreateWithCUMask(hipStream_t* stream, uint32_t cuMaskSize,
                                        const uint32_t* cuMask) {
  HIP_INIT_API(hipExtStreamCreateWithCUMask, stream, cuMaskSize, cuMask);

  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }
  if (cuMaskSize == 0 || cuMask == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  const std::vector<uint32_t> cuMaskv(cuMask, cuMask + cuMaskSize);

  HIP_RETURN(ihipStreamCreate(stream, hipStreamDefault, hip::Stream::Priority::Normal, cuMaskv), *stream);
}

/// MOREH_COPIED_TOUCHED
hipError_t hipStreamDestroy(hipStream_t stream) {
  HIP_INIT_API(hipStreamDestroy, stream);

  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }
  if (stream == hipStreamPerThread) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }
  if (!hip::isValid(stream)) {
    HIP_RETURN(hipErrorContextIsDestroyed);
  }

  cl_command_queue command_queue = reinterpret_cast<cl_command_queue>(stream);
  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  amd::Device& amdDevice = queue->device();

  hip::Device* hipDevice = g_devices[amdDevice.index()];
  hipDevice->RemoveStreamFromPools(queue);

  amd::ScopedLock lock(amdDevice.queueLock());
  amdDevice.removeHostQueue(queue);
  
  queue->release();

  HIP_RETURN(hipSuccess);
}

/// MOREH_COPIED_TOUCHED
hipError_t hipStreamQuery(hipStream_t stream) {
  HIP_INIT_API(hipStreamQuery, stream);

  if (!hip::isValid(stream)) {
    HIP_RETURN(hipErrorContextIsDestroyed);
  }

  amd::HostQueue* hostQueue = hip::getQueue(stream);

  amd::Command* command = new amd::Marker(*hostQueue, false);
  if (command == NULL) {
    HIP_RETURN(hipErrorContextIsDestroyed);
  }
  command->enqueue();

  amd::Event& event = command->event();
  bool ready = command->queue()->device().IsHwEventReady(event);
  if (!ready) {
    ready = (command->status() == CL_COMPLETE);
  }
  hipError_t status = ready ? hipSuccess : hipErrorNotReady;
  command->release();
  HIP_RETURN(status);
}

/// MOREH_COPIED_UNTOUCHED
hipError_t hipStreamSynchronize_common(hipStream_t stream) {
  if (!hip::isValid(stream)) {
    HIP_RETURN(hipErrorContextIsDestroyed);
  }
  // Wait for the current host queue
  hip::getQueue(stream)->finish();
  return hipSuccess;
}

/// MOREH_COPIED_UNTOUCHED
hipError_t hipStreamSynchronize(hipStream_t stream) {
  HIP_INIT_API(hipStreamSynchronize, stream);
  HIP_RETURN(hipStreamSynchronize_common(stream));
}

/// MOREH_COPIED_UNTOUCHED
hipError_t hipStreamSynchronize_spt(hipStream_t stream) {
  HIP_INIT_API(hipStreamSynchronize, stream);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN(hipStreamSynchronize_common(stream));
}

/// MOREH_COPIED_UNTOUCHED
hipError_t hipStreamWaitEvent_common(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  EVENT_CAPTURE(hipStreamWaitEvent, event, stream, flags);

  if (event == nullptr) {
    return hipErrorInvalidHandle;
  }

  if (flags != 0) {
    return hipErrorInvalidValue;
  }

  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }

  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  return e->streamWait(stream, flags);
}

/// MOREH_COPIED_UNTOUCHED
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  HIP_INIT_API(hipStreamWaitEvent, stream, event, flags);
  HIP_RETURN(hipStreamWaitEvent_common(stream, event, flags));
}

/// MOREH_COPIED_UNTOUCHED
hipError_t hipStreamWaitEvent_spt(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  HIP_INIT_API(hipStreamWaitEvent, stream, event, flags);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN(hipStreamWaitEvent_common(stream, event, flags));
}

/// MOREH_COPIED_TOUCHED
hipError_t hipStreamGetCaptureInfo(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
                                   unsigned long long* pId) {
  HIP_INIT_API(hipStreamGetCaptureInfo, stream, pCaptureStatus, pId);
  if (pCaptureStatus == nullptr) {
    return hipErrorInvalidValue;
  }
  *pCaptureStatus = hipStreamCaptureStatusNone;
  HIP_RETURN(hipSuccess);
}

/// MOREH_COPIED_TOUCHED
hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out,
                                      unsigned long long* id_out, hipGraph_t* graph_out,
                                      const hipGraphNode_t** dependencies_out,
                                      size_t* numDependencies_out) {
  HIP_INIT_API(hipStreamGetCaptureInfo_v2, stream, captureStatus_out, id_out, graph_out,
               dependencies_out, numDependencies_out);
  if (captureStatus_out == nullptr) {
    return hipErrorInvalidValue;
  }
  *captureStatus_out = hipStreamCaptureStatusNone;
  HIP_RETURN(hipSuccess);
}
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// Memory Management
///////////////////////////////////////////////////////////////////////////////
hipError_t ihipFree(void* ptr);
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// Texture Management
///////////////////////////////////////////////////////////////////////////////

/// MOREH_COPIED_UNTOUCHED
struct __hip_texture {
  uint32_t imageSRD[HIP_IMAGE_OBJECT_SIZE_DWORD];
  uint32_t samplerSRD[HIP_SAMPLER_OBJECT_SIZE_DWORD];
  amd::Image* image;
  amd::Sampler* sampler;
  hipResourceDesc resDesc;
  hipTextureDesc texDesc;
  hipResourceViewDesc resViewDesc;

  __hip_texture(amd::Image* image_,
                amd::Sampler* sampler_,
                const hipResourceDesc& resDesc_,
                const hipTextureDesc& texDesc_,
                const hipResourceViewDesc& resViewDesc_) :
    image(image_),
    sampler(sampler_),
    resDesc(resDesc_),
    texDesc(texDesc_),
    resViewDesc(resViewDesc_) {
    amd::Context& context = *hip::getCurrentDevice()->asContext();
    amd::Device& device = *context.devices()[0];

    device::Memory* imageMem = image->getDeviceMemory(device);
    std::memcpy(imageSRD, imageMem->cpuSrd(), sizeof(imageSRD));

    device::Sampler* samplerMem = sampler->getDeviceSampler(device);
    std::memcpy(samplerSRD, samplerMem->hwState(), sizeof(samplerSRD));
  }
};

/// MOREH_COPIED_UNTOUCHED
hipError_t ihipDestroyTextureObject(hipTextureObject_t texObject) {
  if (texObject == nullptr) {
    return hipSuccess;
  }
  amd::Device* device = hip::getCurrentDevice()->devices()[0];
  const device::Info& info = device->info();
  if (!info.imageSupport_) {
    LogPrintfError("Texture not supported on the device %s", info.name_);
    return hipErrorNotSupported;
  }

  const hipResourceType type = texObject->resDesc.resType;
  const bool isImageFromBuffer = (type == hipResourceTypeLinear) || (type == hipResourceTypePitch2D);
  const bool isImageView = ((type == hipResourceTypeArray) || (type == hipResourceTypeMipmappedArray)) &&
                           !texObject->image->isParent();
  // If the texture object was created from an array, then the array owns the image SRD.
  // Otherwise, if the texture object is a view, or was created from a buffer, then it owns the image SRD.
  if (isImageFromBuffer || isImageView) {
    texObject->image->release();
  }

  // The texture object always owns the sampler SRD.
  texObject->sampler->release();

  // TODO Should call ihipFree() to not polute the api trace.
  return ihipFree(texObject);
}

/// MOREH_COPIED_UNTOUCHED
hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w, hipChannelFormatKind f) {
  return {x, y, z, w, f};
}

/// MOREH_COPIED_UNTOUCHED
hipError_t ihipUnbindTexture(textureReference* texRef) {
  hipError_t hip_error = hipSuccess;

  do {
    if (texRef == nullptr) {
      hip_error = hipErrorInvalidValue;
      break;
    }

    amd::Device* device = hip::getCurrentDevice()->devices()[0];
    const device::Info& info = device->info();
    if (!info.imageSupport_) {
      LogPrintfError("Texture not supported on the device %s", info.name_);
      HIP_RETURN(hipErrorNotSupported);
    }

    hip_error = ihipDestroyTextureObject(texRef->textureObject);
    if (hip_error != hipSuccess) {
      break;
    }

    const_cast<textureReference*>(texRef)->textureObject = nullptr;

  } while (0);

  return hip_error;
}
///////////////////////////////////////////////////////////////////////////////
