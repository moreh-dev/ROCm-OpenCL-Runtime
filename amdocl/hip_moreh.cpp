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
#include "hip_moreh.hpp"

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
amd::HostQueue* getNullStream(amd::Context& ctx) {
  amd::Device* amdDevice = ctx.devices()[0];
  if (amdDevice->hostQueues().empty())
    ShouldNotReachHere();
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

/// MOREH_COPIED_TOUCHED
amd::HostQueue* Device::NullStream(bool skip_alloc) {
  amd::Device* amdDevice = this->devices()[0];
  return amdDevice->hostQueues()[0];
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
    spdlog::info("hipInit should be called after initializing OpenCL runtime!");
    exit(0);
  }

  const std::vector<amd::Device*>& devices =
    amd::Device::getDevices(CL_DEVICE_TYPE_GPU, false);

  for (unsigned int i = 0; i < devices.size(); ++i) {
    amd::Context* context = &(devices[i]->context());
    if (!context)
      return hipErrorInvalidDevice;
    g_devices.push_back(new Device(context, devices[i], i));
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

hipError_t hipCtxGetCurrent(hipCtx_t* ctx) {
  HIP_INIT_API(hipCtxGetCurrent, ctx);

  *ctx = reinterpret_cast<hipCtx_t>(hip::getCurrentDevice());

  HIP_RETURN(hipSuccess);
}

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

hipError_t hipDriverGetVersion(int* driverVersion) {
  HIP_INIT_API_NO_RETURN(hipDriverGetVersion, driverVersion);

  if (!driverVersion) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *driverVersion = 50120531;

  HIP_RETURN(hipSuccess);
}
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// Device Management
///////////////////////////////////////////////////////////////////////////////
hipError_t ihipDeviceGetCount(int* count) {
  if (count == nullptr) {
    return hipErrorInvalidValue;
  }

  // Get all available devices
  *count = g_devices.size();

  if (*count < 1) {
    return hipErrorNoDevice;
  }

  return hipSuccess;
}


hipError_t ihipDeviceGet(hipDevice_t* device, int deviceId) {
  if (deviceId < 0 || static_cast<size_t>(deviceId) >= g_devices.size() || device == nullptr) {
    return hipErrorInvalidDevice;
  }
  *device = deviceId;
  return hipSuccess;
}

hipError_t hipDeviceGet(hipDevice_t* device, int deviceId) {
  HIP_INIT_API(hipDeviceGet, device, deviceId);

  HIP_RETURN(ihipDeviceGet(device, deviceId));
}

hipError_t ihipGetDeviceProperties(hipDeviceProp_t* props, hipDevice_t device) {
  if (props == nullptr) {
    return hipErrorInvalidValue;
  }

  if (unsigned(device) >= g_devices.size()) {
    return hipErrorInvalidDevice;
  }
  auto* deviceHandle = g_devices[device]->devices()[0];

  hipDeviceProp_t deviceProps = {0};

  const auto& info = deviceHandle->info();
  const auto& isa = deviceHandle->isa();
  ::strncpy(deviceProps.name, info.boardName_, 128);
  deviceProps.totalGlobalMem = info.globalMemSize_;
  deviceProps.sharedMemPerBlock = info.localMemSizePerCU_;
  deviceProps.regsPerBlock = info.availableRegistersPerCU_;
  deviceProps.warpSize = info.wavefrontWidth_;
  deviceProps.maxThreadsPerBlock = info.maxWorkGroupSize_;
  deviceProps.maxThreadsDim[0] = info.maxWorkItemSizes_[0];
  deviceProps.maxThreadsDim[1] = info.maxWorkItemSizes_[1];
  deviceProps.maxThreadsDim[2] = info.maxWorkItemSizes_[2];
  deviceProps.maxGridSize[0] = INT32_MAX;
  deviceProps.maxGridSize[1] = INT32_MAX;
  deviceProps.maxGridSize[2] = INT32_MAX;
  deviceProps.clockRate = info.maxEngineClockFrequency_ * 1000;
  deviceProps.memoryClockRate = info.maxMemoryClockFrequency_ * 1000;
  deviceProps.memoryBusWidth = info.globalMemChannels_;
  deviceProps.totalConstMem = info.maxConstantBufferSize_;
  deviceProps.major = isa.versionMajor();
  deviceProps.minor = isa.versionMinor();
  deviceProps.multiProcessorCount = info.maxComputeUnits_;
  deviceProps.l2CacheSize = info.l2CacheSize_;
  deviceProps.maxThreadsPerMultiProcessor = info.maxThreadsPerCU_;
  deviceProps.computeMode = 0;
  deviceProps.clockInstructionRate = info.timeStampFrequency_;
  deviceProps.arch.hasGlobalInt32Atomics = 1;
  deviceProps.arch.hasGlobalFloatAtomicExch = 1;
  deviceProps.arch.hasSharedInt32Atomics = 1;
  deviceProps.arch.hasSharedFloatAtomicExch = 1;
  deviceProps.arch.hasFloatAtomicAdd = 1;
  deviceProps.arch.hasGlobalInt64Atomics = 1;
  deviceProps.arch.hasSharedInt64Atomics = 1;
  deviceProps.arch.hasDoubles = 1;
  deviceProps.arch.hasWarpVote = 1;
  deviceProps.arch.hasWarpBallot = 1;
  deviceProps.arch.hasWarpShuffle = 1;
  deviceProps.arch.hasFunnelShift = 0;
  deviceProps.arch.hasThreadFenceSystem = 1;
  deviceProps.arch.hasSyncThreadsExt = 0;
  deviceProps.arch.hasSurfaceFuncs = 0;
  deviceProps.arch.has3dGrid = 1;
  deviceProps.arch.hasDynamicParallelism = 0;
  deviceProps.concurrentKernels = 1;
  deviceProps.pciDomainID = info.pciDomainID;
  deviceProps.pciBusID = info.deviceTopology_.pcie.bus;
  deviceProps.pciDeviceID = info.deviceTopology_.pcie.device;
  deviceProps.maxSharedMemoryPerMultiProcessor = info.localMemSizePerCU_;
  deviceProps.canMapHostMemory = 1;
  // FIXME: This should be removed, targets can have character names as well.
  deviceProps.gcnArch = isa.versionMajor() * 100 + isa.versionMinor() * 10 + isa.versionStepping();
  sprintf(deviceProps.gcnArchName, "%s", isa.targetId());
  deviceProps.cooperativeLaunch = info.cooperativeGroups_;
  deviceProps.cooperativeMultiDeviceLaunch = info.cooperativeMultiDeviceGroups_;

  deviceProps.cooperativeMultiDeviceUnmatchedFunc = info.cooperativeMultiDeviceGroups_;
  deviceProps.cooperativeMultiDeviceUnmatchedGridDim = info.cooperativeMultiDeviceGroups_;
  deviceProps.cooperativeMultiDeviceUnmatchedBlockDim = info.cooperativeMultiDeviceGroups_;
  deviceProps.cooperativeMultiDeviceUnmatchedSharedMem = info.cooperativeMultiDeviceGroups_;

  deviceProps.maxTexture1DLinear = 16 * info.imageMaxBufferSize_;  // Max pixel size is 16 bytes
  deviceProps.maxTexture1D = info.image1DMaxWidth_;
  deviceProps.maxTexture2D[0] = info.image2DMaxWidth_;
  deviceProps.maxTexture2D[1] = info.image2DMaxHeight_;
  deviceProps.maxTexture3D[0] = info.image3DMaxWidth_;
  deviceProps.maxTexture3D[1] = info.image3DMaxHeight_;
  deviceProps.maxTexture3D[2] = info.image3DMaxDepth_;
  deviceProps.hdpMemFlushCntl = info.hdpMemFlushCntl;
  deviceProps.hdpRegFlushCntl = info.hdpRegFlushCntl;

  deviceProps.memPitch = info.maxMemAllocSize_;
  deviceProps.textureAlignment = info.imageBaseAddressAlignment_;
  deviceProps.texturePitchAlignment = info.imagePitchAlignment_;
  deviceProps.kernelExecTimeoutEnabled = 0;
  deviceProps.ECCEnabled = info.errorCorrectionSupport_ ? 1 : 0;
  deviceProps.isLargeBar = info.largeBar_ ? 1 : 0;
  deviceProps.asicRevision = info.asicRevision_;

  // HMM capabilities
  deviceProps.managedMemory = info.hmmSupported_;
  deviceProps.concurrentManagedAccess = info.hmmSupported_;
  deviceProps.directManagedMemAccessFromHost = info.hmmDirectHostAccess_;
  deviceProps.pageableMemoryAccess = info.hmmCpuMemoryAccessible_;
  deviceProps.pageableMemoryAccessUsesHostPageTables = info.hostUnifiedMemory_;

  *props = deviceProps;
  return hipSuccess;
}

hipError_t hipGetDeviceProperties(hipDeviceProp_t* props, hipDevice_t device) {
  HIP_INIT_API(hipGetDeviceProperties, props, device);

  HIP_RETURN(ihipGetDeviceProperties(props, device));
}

hipError_t hipDeviceTotalMem(size_t *bytes, hipDevice_t device) {
  HIP_INIT_API(hipDeviceTotalMem, bytes, device);

  if (device < 0 || static_cast<size_t>(device) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  if (bytes == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto* deviceHandle = g_devices[device]->devices()[0];
  const auto& info = deviceHandle->info();

  *bytes = info.globalMemSize_;

  HIP_RETURN(hipSuccess);
}
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// Stream Management
///////////////////////////////////////////////////////////////////////////////
/// MOREH_COPIED_TOUCHED
int Stream::DeviceId(const hipStream_t hStream) {
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

/// MOREH_COPIED_UNTOUCHED
void CL_CALLBACK ihipStreamCallback(cl_event event, cl_int command_exec_status, void* user_data) {
  hipError_t status = hipSuccess;
  StreamCallback* cbo = reinterpret_cast<StreamCallback*>(user_data);
  cbo->callBack_(cbo->stream_, status, cbo->userData_);
  cbo->command_->release();
  delete cbo;
}

hipError_t ihipStreamCreate(hipStream_t* stream, unsigned int flags,
    const std::vector<uint32_t>& cuMask = {}) {
  if (flags != hipStreamDefault && flags != hipStreamNonBlocking) {
    return hipErrorInvalidValue;
  }

  amd::Context& context = *hip::getCurrentDevice()->asContext();
  amd::Device& device = *context.devices()[0];

  cl_command_queue_properties properties =
    HIP_FORCE_QUEUE_PROFILING ? CL_QUEUE_PROFILING_ENABLE : 0;

  amd::HostQueue* queue = new amd::HostQueue(context, device, properties,
      amd::CommandQueue::RealTimeDisabled,
      amd::CommandQueue::Priority::Normal, cuMask);

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

/// MOREH_COPIED_TOUCHED
hipError_t hipStreamCreate(hipStream_t *stream) {
  HIP_INIT_API(hipStreamCreate, stream);

  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(ihipStreamCreate(stream, hipStreamDefault), *stream);
}

/// MOREH_COPIED_TOUCHED
hipError_t hipStreamCreateWithFlags(hipStream_t *stream, unsigned int flags) {
  HIP_INIT_API(hipStreamCreateWithFlags, stream, flags);

  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(ihipStreamCreate(stream, flags), *stream);
}

/// MOREH_COPIED_TOUCHED
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

  HIP_RETURN(ihipStreamCreate(stream, hipStreamDefault, cuMaskv), *stream);
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
hipError_t hipStreamSynchronize(hipStream_t stream) {
  HIP_INIT_API(hipStreamSynchronize, stream);

  if (!hip::isValid(stream)) {
    HIP_RETURN(hipErrorContextIsDestroyed);
  }

  // Wait for the current host queue
  hip::getQueue(stream)->finish();

  HIP_RETURN(hipSuccess);
}

/// MOREH_COPIED_UNTOUCHED
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  HIP_INIT_API(hipStreamWaitEvent, stream, event, flags);

  EVENT_CAPTURE(hipStreamWaitEvent, event, stream, flags);

  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  if (!hip::isValid(stream)) {
    HIP_RETURN(hipErrorContextIsDestroyed);
  }

  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  HIP_RETURN(e->streamWait(stream, flags));
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
// Event Management
///////////////////////////////////////////////////////////////////////////////
;
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
