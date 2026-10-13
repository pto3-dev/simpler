/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <acl/acl.h>
#include <runtime/rt.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "tensormap_and_ringbuffer/tensor.h"

namespace {

thread_local std::string last_error;

void Check(int status, const char *operation) {
    if (status != 0) {
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
    }
}

struct HostTensorArg {
    uint64_t address;
    uint64_t bytes;
    uint32_t dtype;
    uint32_t rank;
    uint32_t shape[MAX_TENSOR_DIMS];
};
static_assert(std::is_standard_layout_v<HostTensorArg> && std::is_trivially_copyable_v<HostTensorArg>);
static_assert(sizeof(HostTensorArg) == 48);
static_assert(static_cast<uint32_t>(DataType::FLOAT32) == 0);
static_assert(static_cast<uint32_t>(DataType::FLOAT16) == 1);
using KernelTensor = simpler::tmr::Tensor;
static_assert(std::is_standard_layout_v<KernelTensor> && std::is_trivially_copyable_v<KernelTensor>);
static_assert(sizeof(KernelTensor) == 128);

class ContextGuard {
public:
    explicit ContextGuard(aclrtContext context) {
        aclrtGetCurrentContext(&previous_);
        Check(aclrtSetCurrentContext(context), "set context");
    }
    ~ContextGuard() {
        if (previous_) {
            aclrtSetCurrentContext(previous_);
        }
    }
    void Restore() {
        if (previous_) {
            Check(aclrtSetCurrentContext(previous_), "restore caller context");
            previous_ = nullptr;
        }
    }
    ContextGuard(const ContextGuard &) = delete;
    ContextGuard &operator=(const ContextGuard &) = delete;

private:
    aclrtContext previous_ = nullptr;
};

struct DeviceFree {
    void operator()(void *pointer) const {
        if (pointer) {
            aclrtFree(pointer);
        }
    }
};
using DeviceAllocation = std::unique_ptr<void, DeviceFree>;

void Validate(const HostTensorArg &arg) {
    if (!arg.address || arg.rank == 0 || arg.rank > MAX_TENSOR_DIMS) {
        throw std::invalid_argument("invalid host tensor address or rank");
    }
    if (arg.dtype != static_cast<uint32_t>(DataType::FLOAT16) &&
        arg.dtype != static_cast<uint32_t>(DataType::FLOAT32)) {
        throw std::invalid_argument("A1 single launch requires FP16 or FP32");
    }
    uint64_t elements = 1;
    for (uint32_t i = 0; i < arg.rank; ++i) {
        if (!arg.shape[i] || elements > std::numeric_limits<uint32_t>::max() / arg.shape[i]) {
            throw std::invalid_argument("host tensor shape exceeds the kernel ABI");
        }
        elements *= arg.shape[i];
    }
    const uint64_t element_bytes = arg.dtype == static_cast<uint32_t>(DataType::FLOAT16) ? 2 : 4;
    if (arg.bytes != elements * element_bytes || arg.address > UINT64_MAX - arg.bytes) {
        throw std::invalid_argument("host tensor byte extent does not match its shape");
    }
}

class SingleLaunch {
public:
    SingleLaunch(const void *binary, uint64_t bytes, int32_t device) :
        object_(static_cast<const char *>(binary), static_cast<const char *>(binary) + bytes) {
        const auto init_status = aclInit(nullptr);
        if (init_status != ACL_ERROR_REPEAT_INITIALIZE) {
            Check(init_status, "aclInit");
        }
        const char *soc = aclrtGetSocName();
        if (!soc || std::string(soc).find("Ascend310P") != 0) {
            throw std::invalid_argument("A1 single launch requires Ascend310P silicon");
        }
        aclrtContext previous = nullptr;
        aclrtGetCurrentContext(&previous);
        try {
            Check(aclrtCreateContext(&context_, device), "create context");
            Check(aclrtCreateStream(&stream_), "create stream");
            rtDevBinary_t image{};
            image.magic = RT_DEV_BINARY_MAGIC_ELF;
            image.data = object_.data();
            image.length = object_.size();
            Check(rtRegisterAllKernel(&image, &kernel_), "register kernel");
            if (previous) {
                Check(aclrtSetCurrentContext(previous), "restore context");
            }
        } catch (...) {
            if (kernel_) {
                rtDevBinaryUnRegister(kernel_);
            }
            if (stream_) {
                aclrtDestroyStream(stream_);
            }
            if (context_) {
                aclrtDestroyContext(context_);
            }
            if (previous) {
                aclrtSetCurrentContext(previous);
            }
            throw;
        }
    }

    void Run(const HostTensorArg *args, uint32_t count) {
        if (!args || count == 0 || count > 64) {
            throw std::invalid_argument("tensor argument count must be in 1..64");
        }
        for (uint32_t i = 0; i < count; ++i) {
            Validate(args[i]);
            for (uint32_t j = 0; j < i; ++j) {
                const auto &a = args[i];
                const auto &b = args[j];
                const bool overlaps = a.address < b.address + b.bytes && b.address < a.address + a.bytes;
                if (overlaps && (a.address != b.address || a.bytes != b.bytes)) {
                    throw std::invalid_argument("partially overlapping host tensors are unsupported");
                }
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (!stream_ || !kernel_) {
            throw std::runtime_error("single launch session is closed");
        }
        ContextGuard context(context_);
        Check(aclrtSynchronizeStreamWithTimeout(stream_, 5000), "drain stream");
        allocations_.clear();
        std::vector<void *> data(count);
        std::vector<KernelTensor> descriptors(count);
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t alias = 0;
            while (alias < i && args[alias].address != args[i].address) {
                ++alias;
            }
            data[i] = alias < i ? data[alias] : Allocate(args[i].bytes);
            if (alias == i) {
                Check(
                    aclrtMemcpy(
                        data[i], args[i].bytes, reinterpret_cast<const void *>(args[i].address), args[i].bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE
                    ),
                    "upload tensor"
                );
            }
            descriptors[i].init_external(
                data[i], args[i].bytes, args[i].shape, args[i].rank, static_cast<DataType>(args[i].dtype), 0, false,
                AddressSpace::DEVICE
            );
        }
        const uint64_t descriptor_bytes = descriptors.size() * sizeof(KernelTensor);
        void *device_descriptors = Allocate(descriptor_bytes);
        Check(
            aclrtMemcpy(
                device_descriptors, descriptor_bytes, descriptors.data(), descriptor_bytes, ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "upload descriptors"
        );
        std::vector<uint64_t> slots(count);
        for (uint32_t i = 0; i < count; ++i) {
            slots[i] = reinterpret_cast<uint64_t>(device_descriptors) + i * sizeof(KernelTensor);
        }
        const uint64_t slot_bytes = slots.size() * sizeof(uint64_t);
        void *device_slots = Allocate(slot_bytes);
        Check(
            aclrtMemcpy(device_slots, slot_bytes, slots.data(), slot_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
            "upload arguments"
        );
        uint64_t entry_argument = reinterpret_cast<uint64_t>(device_slots);
        rtArgsEx_t launch_args{};
        launch_args.args = &entry_argument;
        launch_args.argsSize = sizeof(entry_argument);
        rtTaskCfgInfo_t config{};
        config.schemMode = RT_SCHEM_MODE_BATCH;
        Check(rtKernelLaunchWithHandleV2(kernel_, 0, 1, &launch_args, nullptr, stream_, &config), "launch kernel");
        Check(aclrtSynchronizeStreamWithTimeout(stream_, 5000), "synchronize kernel");
        for (uint32_t i = 0; i < count; ++i) {
            Check(
                aclrtMemcpy(
                    reinterpret_cast<void *>(args[i].address), args[i].bytes, data[i], args[i].bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST
                ),
                "download tensor"
            );
        }
        allocations_.clear();
        context.Restore();
    }

    void Close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!context_) {
            return;
        }
        aclrtContext previous = nullptr;
        aclrtGetCurrentContext(&previous);
        Check(aclrtSetCurrentContext(context_), "set context for close");
        try {
            if (stream_) {
                Check(aclrtSynchronizeStreamWithTimeout(stream_, 5000), "drain stream before close");
            }
            allocations_.clear();
            if (kernel_) {
                Check(rtDevBinaryUnRegister(kernel_), "unregister kernel");
                kernel_ = nullptr;
            }
            if (stream_) {
                Check(aclrtDestroyStream(stream_), "destroy stream");
                stream_ = nullptr;
            }
            Check(aclrtDestroyContext(context_), "destroy context");
            const auto destroyed_context = context_;
            context_ = nullptr;
            if (previous && previous != destroyed_context) {
                Check(aclrtSetCurrentContext(previous), "restore context after close");
            }
        } catch (...) {
            if (previous && previous != context_) {
                aclrtSetCurrentContext(previous);
            }
            throw;
        }
    }

private:
    void *Allocate(uint64_t bytes) {
        void *pointer = nullptr;
        Check(aclrtMalloc(&pointer, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "allocate device memory");
        DeviceAllocation allocation(pointer);
        allocations_.push_back(std::move(allocation));
        return pointer;
    }
    std::vector<char> object_;
    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;
    void *kernel_ = nullptr;
    std::vector<DeviceAllocation> allocations_;
    std::mutex mutex_;
};

template <typename Action>
int Invoke(Action action) noexcept {
    try {
        action();
        last_error.clear();
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return 1;
    }
}

}  // namespace

extern "C" const char *simpler_a1_last_error() { return last_error.c_str(); }

extern "C" int simpler_a1_create(const void *binary, uint64_t bytes, int32_t device, void **result) {
    return Invoke([&] {
        if (!binary || bytes < 4 || bytes > UINT32_MAX || !result || device < 0 ||
            std::memcmp(binary, "\177ELF", 4) != 0) {
            throw std::invalid_argument("invalid A1 ELF object or device");
        }
        *result = new SingleLaunch(binary, bytes, device);
    });
}

extern "C" int simpler_a1_run(void *session, const HostTensorArg *args, uint32_t count) {
    return Invoke([&] {
        if (!session) {
            throw std::invalid_argument("single launch session is closed");
        }
        static_cast<SingleLaunch *>(session)->Run(args, count);
    });
}

extern "C" int simpler_a1_close(void *session) {
    return Invoke([&] {
        if (session) {
            auto *runner = static_cast<SingleLaunch *>(session);
            runner->Close();
            delete runner;
        }
    });
}
