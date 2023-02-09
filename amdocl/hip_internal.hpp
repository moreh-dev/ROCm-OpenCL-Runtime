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

#ifndef HIP_SRC_HIP_INTERNAL_H
#define HIP_SRC_HIP_INTERNAL_H

#include "vdi_common.hpp"
//#include "hip_prof_api.h"
#include "trace_helper.h"
//#include "utils/debug.hpp"
#include "hip_formatting.hpp"
//#include "hip_graph_capture.hpp"

#include <unordered_set>
#include <thread>
#include <stack>
#include <mutex>
#include <iterator>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

/*! IHIP IPC MEMORY Structure */
#define IHIP_IPC_MEM_HANDLE_SIZE   32
#define IHIP_IPC_MEM_RESERVED_SIZE LP64_SWITCH(24,16)

typedef struct ihipIpcMemHandle_st {
  char ipc_handle[IHIP_IPC_MEM_HANDLE_SIZE];  ///< ipc memory handle on ROCr
  size_t psize;
  size_t poffset;
  char reserved[IHIP_IPC_MEM_RESERVED_SIZE];
} ihipIpcMemHandle_t;

#define IHIP_IPC_EVENT_HANDLE_SIZE 32
#define IHIP_IPC_EVENT_RESERVED_SIZE LP64_SWITCH(28,24)
typedef struct ihipIpcEventHandle_st {
    //hsa_amd_ipc_signal_t ipc_handle;  ///< ipc signal handle on ROCr
    //char ipc_handle[IHIP_IPC_EVENT_HANDLE_SIZE];
    //char reserved[IHIP_IPC_EVENT_RESERVED_SIZE];
    char shmem_name[IHIP_IPC_EVENT_HANDLE_SIZE];
}ihipIpcEventHandle_t;

#ifdef _WIN32
  inline int getpid() { return _getpid(); }
#endif

const char* ihipGetErrorName(hipError_t hip_error);

static  amd::Monitor g_hipInitlock{"hipInit lock"};
#define HIP_INIT(noReturn) {\
    amd::ScopedLock lock(g_hipInitlock);                     \
    if (!hip::initialized()) {                               \
      if (hipInit(0) != hipSuccess && !noReturn) {           \
        HIP_RETURN(hipErrorInvalidDevice);                   \
      }                                                      \
    }                                                        \
    if (hip::g_device == nullptr && g_devices.size() > 0) {  \
      hip::g_device = g_devices[0];                          \
      amd::Os::setPreferredNumaNode(g_devices[0]->devices()[0]->getPreferredNumaNode());  \
    }                                                        \
  }

#define HIP_INIT_VOID() {\
    amd::ScopedLock lock(g_hipInitlock);                     \
    if (!hip::initialized()) {                               \
      if (hipInit(0)) {}                                     \
    }                                                        \
    if (hip::g_device == nullptr && g_devices.size() > 0) {  \
      hip::g_device = g_devices[0];                          \
      amd::Os::setPreferredNumaNode(g_devices[0]->devices()[0]->getPreferredNumaNode());  \
    }                                                        \
  }

// HIP API callbacks spawner object macro
#define HIP_CB_SPAWNER_OBJECT(CB_ID)

namespace hip {
extern volatile bool initialized_;
inline bool initialized() { return initialized_; }
};


#define HIP_API_PRINT(...)                                 \
  uint64_t startTimeUs=0 ; HIPPrintDuration(amd::LOG_INFO, amd::LOG_API, &startTimeUs, "%s%s ( %s )%s", KGRN,    \
          __func__, ToString( __VA_ARGS__ ).c_str(),KNRM);

#define HIP_ERROR_PRINT(err, ...)                                                  \
  ClPrint(amd::LOG_INFO, amd::LOG_API, "%s: Returned %s : %s",                     \
          __func__, ihipGetErrorName(err), ToString( __VA_ARGS__ ).c_str());

#define HIP_INIT_API_INTERNAL(noReturn, cid, ...)            \
  HIP_API_PRINT(__VA_ARGS__)                                 \
  amd::Thread* thread = amd::Thread::current();              \
  if (!VDI_CHECK_THREAD(thread) && !noReturn) {              \
    HIP_RETURN(hipErrorOutOfMemory);                         \
  }                                                          \
  HIP_INIT(noReturn)                                         \
  HIP_CB_SPAWNER_OBJECT(cid);

// This macro should be called at the beginning of every HIP API.
#define HIP_INIT_API(cid, ...)                                                                     \
  HIP_INIT_API_INTERNAL(0, cid, __VA_ARGS__)                                                       \
  if (g_devices.size() == 0) {                                                                     \
    HIP_RETURN(hipErrorNoDevice);                                                                  \
  }

#define HIP_INIT_API_NO_RETURN(cid, ...)                     \
  HIP_INIT_API_INTERNAL(1, cid, __VA_ARGS__)

#define HIP_RETURN_DURATION(ret, ...)                        \
  hip::g_lastError = ret;                                    \
  HIPPrintDuration(amd::LOG_INFO, amd::LOG_API, &startTimeUs,                      \
                   "%s: Returned %s : %s",                                         \
                   __func__, ihipGetErrorName(hip::g_lastError),                    \
                   ToString( __VA_ARGS__ ).c_str());                               \
  return hip::g_lastError;

#define HIP_RETURN(ret, ...)                      \
  hip::g_lastError = ret;                         \
  HIP_ERROR_PRINT(hip::g_lastError, __VA_ARGS__)  \
  return hip::g_lastError;

#define HIP_RETURN_ONFAIL(func)          \
  do {                                   \
    hipError_t herror = (func);          \
    if (herror != hipSuccess) {          \
      HIP_RETURN(herror);                \
    }                                    \
  } while (0);

// Cannot be use in place of HIP_RETURN.
// Refrain from using for external HIP APIs
#define IHIP_RETURN_ONFAIL(func)         \
  do {                                   \
    hipError_t herror = (func);          \
    if (herror != hipSuccess) {          \
      return herror;                     \
    }                                    \
  } while (0);

#define CHECK_STREAM_CAPTURE_SUPPORTED()

// Sync APIs cannot be called when stream capture is active
#define CHECK_STREAM_CAPTURING()

#define STREAM_CAPTURE(name, stream, ...)

#define EVENT_CAPTURE(name, event, ...)

// Moreh OpenCL/HIP does not support per-thread default stream
#define PER_THREAD_DEFAULT_STREAM(stream)

//namespace hc {
//class accelerator;
//class accelerator_view;
//};
namespace hip {
  class Device;
  class MemoryPool;
  using Stream = amd::HostQueue;

  /// HIP Device class
  class Device {
    amd::Monitor lock_{"Device lock"};
    /// ROCclr context
    amd::Context* context_;
    /// ROCclr device
    amd::Device* device_;
    /// Device's ID
    /// Store it here so we don't have to loop through the device list every time
    int deviceId_;
//    /// ROCclr host queue for default streams
//    Stream null_stream_;
    /// Store device flags
    unsigned int flags_;
    /// Maintain list of user enabled peers
    std::list<int> userEnabledPeers;

//    /// True if this device is active
//    bool isActive_;
//
//    std::vector<amd::HostQueue*> queues_;

    MemoryPool* default_mem_pool_;
    MemoryPool* current_mem_pool_;

    std::set<MemoryPool*> mem_pools_;

  public:
    Device(amd::Context* ctx, amd::Device* dev, int devId): context_(ctx),
         device_(dev), deviceId_(devId),
//         null_stream_(this, Stream::Priority::Normal, 0, true),
         flags_(hipDeviceScheduleSpin),
//        isActive_(false),
        default_mem_pool_(nullptr),
        current_mem_pool_(nullptr)
        { assert(ctx != nullptr); }
    ~Device();

    bool Create();
    amd::Context* asContext() const { return context_; }
    int deviceId() const { return deviceId_; }
    void retain() const { context_->retain(); }
    void release() const { context_->release(); }
    const std::vector<amd::Device*>& devices() const { return context_->devices(); }
    hipError_t EnablePeerAccess(int peerDeviceId){
      amd::ScopedLock lock(lock_);
      bool found = (std::find(userEnabledPeers.begin(), userEnabledPeers.end(), peerDeviceId) != userEnabledPeers.end());
      if (found) {
        return hipErrorPeerAccessAlreadyEnabled;
      }
      userEnabledPeers.push_back(peerDeviceId);
      return hipSuccess;
    }
    hipError_t DisablePeerAccess(int peerDeviceId) {
      amd::ScopedLock lock(lock_);
      bool found = (std::find(userEnabledPeers.begin(), userEnabledPeers.end(), peerDeviceId) != userEnabledPeers.end());
      if (found) {
        userEnabledPeers.remove(peerDeviceId);
        return hipSuccess;
      } else {
        return hipErrorPeerAccessNotEnabled;
      }
    }
    unsigned int getFlags() const { return flags_; }
    void setFlags(unsigned int flags) { flags_ = flags; }
    amd::HostQueue* NullStream(bool skip_alloc = false);
//    Stream* GetNullStream();
//
//    void SaveQueue(amd::HostQueue* queue) {
//      amd::ScopedLock lock(lock_);
//      queues_.push_back(queue);
//    }
//
//    bool GetActiveStatus() {
//      amd::ScopedLock lock(lock_);
//      if (isActive_) return true;
//      for (int i = 0; i < queues_.size(); i++) {
//        if (queues_[i]->GetQueueStatus()) {
//          isActive_ = true;
//          return true;
//        }
//      }
//      return false;
//    }
    
    /// Set the current memory pool on the device
    void SetCurrentMemoryPool(MemoryPool* pool = nullptr) {
      current_mem_pool_ = (pool == nullptr) ? default_mem_pool_ : pool;
    }

    /// Get the current memory pool on the device
    MemoryPool* GetCurrentMemoryPool() const { return current_mem_pool_; }

    /// Get the default memory pool on the device
    MemoryPool* GetDefaultMemoryPool() const { return default_mem_pool_; }

    /// Add memory pool to the device
    void AddMemoryPool(MemoryPool* pool);

    /// Remove memory pool from the device
    void RemoveMemoryPool(MemoryPool* pool);

    /// Free memory from the device
    bool FreeMemory(amd::Memory* memory, Stream* stream);

    /// Release freed memory from all pools on the current device
    void ReleaseFreedMemory(Stream* stream);

    /// Removes a destroyed stream from the safe list of memory pools
    void RemoveStreamFromPools(Stream* stream);
  };

  /// Current thread's device
  extern thread_local Device* g_device;
  extern thread_local hipError_t g_lastError;
  /// Device representing the host - for pinned memory
  extern Device* host_device;

  extern bool init();

  extern Device* getCurrentDevice();

  extern void setCurrentDevice(unsigned int index);

  /// Get ROCclr queue associated with hipStream
  /// Note: This follows the CUDA spec to sync with default streams
  ///       and Blocking streams
  extern amd::HostQueue* getQueue(hipStream_t stream);
  /// Get default stream associated with the ROCclr context
  extern amd::HostQueue* getNullStream(amd::Context&);
  /// Get default stream of the thread
  extern amd::HostQueue* getNullStream();
  /// Get device ID associated with the ROCclr context
  int getDeviceID(amd::Context& ctx);
  /// Check if stream is valid
  extern bool isValid(hipStream_t& stream);
  extern amd::Monitor hipArraySetLock;
  extern std::unordered_set<hipArray*> hipArraySet;
};

extern void WaitThenDecrementSignal(hipStream_t stream, hipError_t status, void* user_data);
struct ihipExec_t {
  dim3 gridDim_;
  dim3 blockDim_;
  size_t sharedMem_;
  hipStream_t hStream_;
  std::vector<char> arguments_;
};

///// Wait all active streams on the blocking queue. The method enqueues a wait command and
///// doesn't stall the current thread
//extern void iHipWaitActiveStreams(amd::HostQueue* blocking_queue, bool wait_null_stream = false);

extern std::vector<hip::Device*> g_devices;
extern hipError_t ihipDeviceGetCount(int* count);
extern int ihipGetDevice();

extern hipError_t ihipMalloc(void** ptr, size_t sizeBytes, unsigned int flags);
extern amd::Memory* getMemoryObject(const void* ptr, size_t& offset, size_t size = 0);
extern amd::Memory* getMemoryObjectWithOffset(const void* ptr, const size_t size);
//extern void getStreamPerThread(hipStream_t& stream);
extern hipStream_t getPerThreadDefaultStream();
extern hipError_t ihipUnbindTexture(textureReference* texRef);
extern hipError_t ihipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags);
extern hipError_t ihipHostUnregister(void* hostPtr);
extern hipError_t ihipGetDeviceProperties(hipDeviceProp_t* props, hipDevice_t device);

extern hipError_t ihipDeviceGet(hipDevice_t* device, int deviceId);
extern hipError_t ihipStreamOperation(hipStream_t stream, cl_command_type cmdType, void* ptr,
                                      uint64_t value, uint64_t mask, unsigned int flags, size_t sizeBytes);

constexpr bool kOptionChangeable = true;
constexpr bool kNewDevProg = false;

constexpr bool kMarkerDisableFlush = true;   //!< Avoids command batch flush in ROCclr

//extern std::vector<hip::Stream*> g_captureStreams;
//extern amd::Monitor g_captureStreamsLock;
//extern thread_local std::vector<hip::Stream*> l_captureStreams;
//extern thread_local hipStreamCaptureMode l_streamCaptureMode;
#endif // HIP_SRC_HIP_INTERNAL_H
