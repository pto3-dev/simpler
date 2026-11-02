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
#include <runtime/rt_model.h>
#include <runtime/rt_preload_task.h>

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
static_assert(static_cast<uint32_t>(DataType::INT32) == 2);
static_assert(static_cast<uint32_t>(DataType::UINT32) == 10);
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

struct ResidentArg {
    HostTensorArg tensor;
    uint64_t offset;
};
static_assert(sizeof(ResidentArg) == 56);

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
        arg.dtype != static_cast<uint32_t>(DataType::FLOAT32) && arg.dtype != static_cast<uint32_t>(DataType::INT32) &&
        arg.dtype != static_cast<uint32_t>(DataType::UINT32)) {
        throw std::invalid_argument("A1 single launch requires FP16, FP32, INT32 or UINT32");
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

constexpr uint32_t kArgumentAbiVersion = 2;

struct L2CacheInfo {
    uint64_t alias_offset;
    int32_t query_status;
};

L2CacheInfo QueryL2CacheOffset(uint32_t device, decltype(&rtGetL2CacheOffset) query) {
    uint64_t offset = 0;
    const int32_t status = query(device, &offset);
    return {status == RT_ERROR_NONE ? offset : 0, status};
}

std::vector<uint64_t> ArgumentSlots(
    uint64_t descriptor_address, const uint32_t *counts, uint32_t call_count, uint32_t abi_version,
    uint64_t l2_cache_offset
) {
    if (abi_version != 1 && abi_version != kArgumentAbiVersion)
        throw std::invalid_argument("unsupported A1 launch argument ABI");
    const bool footer = abi_version == kArgumentAbiVersion;
    if (!counts || call_count == 0 || call_count > 4096)
        throw std::invalid_argument("argument slots require 1..4096 signatures");
    uint64_t total = 0;
    for (uint32_t call = 0; call < call_count; ++call) {
        if (counts[call] == 0 || counts[call] > 64)
            throw std::invalid_argument("argument signature requires 1..64 tensors");
        total += counts[call];
    }
    if (!descriptor_address || descriptor_address > UINT64_MAX - total * sizeof(KernelTensor))
        throw std::invalid_argument("argument descriptor range overflows");
    std::vector<uint64_t> slots;
    slots.reserve(total + (footer ? call_count : 0));
    uint64_t descriptor = 0;
    for (uint32_t call = 0; call < call_count; ++call) {
        for (uint32_t argument = 0; argument < counts[call]; ++argument)
            slots.push_back(descriptor_address + descriptor++ * sizeof(KernelTensor));
        // The footer is outside the tensor signature and never changes descriptors.
        if (footer) slots.push_back(l2_cache_offset);
    }
    return slots;
}

class SingleLaunch {
public:
    SingleLaunch(const void *binary, uint64_t bytes, int32_t device, uint32_t argument_abi = 1) :
        argument_abi_(argument_abi),
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
            if (argument_abi_ == kArgumentAbiVersion)
                l2_cache_info_ = QueryL2CacheOffset(static_cast<uint32_t>(device), rtGetL2CacheOffset);
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

    L2CacheInfo CacheInfo() {
        std::lock_guard<std::mutex> lock(mutex_);
        RequireOpen();
        if (argument_abi_ != kArgumentAbiVersion) throw std::invalid_argument("cache info requires A1 argument ABI 2");
        return l2_cache_info_;
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
        Launch(descriptors.data(), count, kernel_, 1);
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

    uint64_t CreateTensor(const HostTensorArg &arg, bool upload) {
        Validate(arg);
        std::lock_guard<std::mutex> lock(mutex_);
        RequireOpen();
        ContextGuard context(context_);
        void *pointer = nullptr;
        Check(aclrtMalloc(&pointer, arg.bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "allocate resident tensor");
        DeviceAllocation allocation(pointer);
        if (upload) {
            Check(
                aclrtMemcpy(
                    pointer, arg.bytes, reinterpret_cast<void *>(arg.address), arg.bytes, ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "upload resident tensor"
            );
        }
        resident_.push_back({std::move(allocation), arg.bytes, arg.dtype});
        context.Restore();
        return resident_.size();
    }

    void CopyTensor(const ResidentArg &device, const HostTensorArg &host, bool upload) {
        Validate(host);
        std::lock_guard<std::mutex> lock(mutex_);
        RequireOpen();
        void *pointer = ResidentPointer(device);
        if (device.tensor.bytes != host.bytes || device.tensor.dtype != host.dtype || device.tensor.rank != host.rank ||
            std::memcmp(device.tensor.shape, host.shape, sizeof(host.shape)) != 0) {
            throw std::invalid_argument("resident transfer geometry mismatch");
        }
        ContextGuard context(context_);
        if (upload) {
            Check(
                aclrtMemcpy(
                    pointer, host.bytes, reinterpret_cast<void *>(host.address), host.bytes, ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "upload resident tensor"
            );
        } else {
            Check(
                aclrtMemcpy(
                    reinterpret_cast<void *>(host.address), host.bytes, pointer, host.bytes, ACL_MEMCPY_DEVICE_TO_HOST
                ),
                "download resident tensor"
            );
        }
        context.Restore();
    }

    uint32_t RegisterKernel(const void *binary, uint64_t bytes) {
        if (!binary || bytes < 4 || bytes > UINT32_MAX || std::memcmp(binary, "\177ELF", 4) != 0) {
            throw std::invalid_argument("invalid A1 ELF object");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        RequireOpen();
        ContextGuard context(context_);
        ExtraKernel extra;
        extra.object.assign(static_cast<const char *>(binary), static_cast<const char *>(binary) + bytes);
        rtDevBinary_t image{};
        image.magic = RT_DEV_BINARY_MAGIC_ELF;
        image.data = extra.object.data();
        image.length = extra.object.size();
        Check(rtRegisterAllKernel(&image, &extra.handle), "register resident kernel");
        try {
            extra_kernels_.push_back(std::move(extra));
        } catch (...) {
            rtDevBinaryUnRegister(extra.handle);
            throw;
        }
        context.Restore();
        return extra_kernels_.size();
    }

    void RunResident(const ResidentArg *args, uint32_t count, uint32_t kernel_id, uint32_t blocks) {
        if (!args || count == 0 || count > 64 || blocks == 0 || blocks > 8) {
            throw std::invalid_argument("resident launch requires 1..64 tensors and 1..8 blocks");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        RequireOpen();
        if (kernel_id > extra_kernels_.size()) {
            throw std::invalid_argument("resident kernel id is not registered");
        }
        std::vector<KernelTensor> descriptors(count);
        for (uint32_t i = 0; i < count; ++i) {
            const auto &arg = args[i].tensor;
            void *pointer = ResidentPointer(args[i]);
            descriptors[i].init_external(
                pointer, arg.bytes, arg.shape, arg.rank, static_cast<DataType>(arg.dtype), 0, false,
                AddressSpace::DEVICE
            );
        }
        ContextGuard context(context_);
        Check(aclrtSynchronizeStreamWithTimeout(stream_, 5000), "drain resident stream");
        allocations_.clear();
        Launch(descriptors.data(), count, kernel_id ? extra_kernels_[kernel_id - 1].handle : kernel_, blocks);
        allocations_.clear();
        context.Restore();
    }

    uint32_t PrepareSequence(
        const ResidentArg *args, const uint32_t *counts, const uint32_t *ids, const uint32_t *blocks, uint32_t count,
        bool device_model = false
    ) {
        if (!args || !counts || !ids || !blocks || count == 0 || count > 4096)
            throw std::invalid_argument("resident sequence requires 1..4096 calls");
        std::lock_guard<std::mutex> lock(mutex_);
        RequireOpen();
        uint32_t total = 0;
        for (uint32_t i = 0; i < count; ++i) {
            if (counts[i] == 0 || counts[i] > 64 || blocks[i] == 0 || blocks[i] > 8 || ids[i] > extra_kernels_.size())
                throw std::invalid_argument("invalid resident sequence signature or kernel id");
            total += counts[i];
        }
        std::vector<KernelTensor> descriptors(total);
        for (uint32_t i = 0; i < total; ++i) {
            const auto &arg = args[i].tensor;
            descriptors[i].init_external(
                ResidentPointer(args[i]), arg.bytes, arg.shape, arg.rank, static_cast<DataType>(arg.dtype), 0, false,
                AddressSpace::DEVICE
            );
        }
        ContextGuard context(context_);
        PreparedSequence sequence;
        void *descriptor_pointer = nullptr, *slot_pointer = nullptr;
        const uint64_t descriptor_bytes = total * sizeof(KernelTensor);
        const uint64_t slot_bytes = (total + (argument_abi_ == kArgumentAbiVersion ? count : 0)) * sizeof(uint64_t);
        Check(
            aclrtMalloc(&descriptor_pointer, descriptor_bytes, ACL_MEM_MALLOC_NORMAL_ONLY),
            "allocate sequence descriptors"
        );
        sequence.descriptors.reset(descriptor_pointer);
        Check(aclrtMalloc(&slot_pointer, slot_bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "allocate sequence slots");
        sequence.slots.reset(slot_pointer);
        Check(
            aclrtMemcpy(
                descriptor_pointer, descriptor_bytes, descriptors.data(), descriptor_bytes, ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "upload sequence descriptors"
        );
        const auto slots = ArgumentSlots(
            reinterpret_cast<uint64_t>(descriptor_pointer), counts, count, argument_abi_, l2_cache_info_.alias_offset
        );
        Check(
            aclrtMemcpy(slot_pointer, slot_bytes, slots.data(), slot_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
            "upload sequence slots"
        );
        uint32_t offset = 0;
        for (uint32_t i = 0; i < count; ++i) {
            sequence.calls.push_back(
                {reinterpret_cast<uint64_t>(slot_pointer) + offset * sizeof(uint64_t),
                 ids[i] ? extra_kernels_[ids[i] - 1].handle : kernel_, blocks[i]}
            );
            offset += counts[i] + (argument_abi_ == kArgumentAbiVersion ? 1 : 0);
        }
        if (device_model) sequence.model = std::make_unique<DeviceModel>(sequence.calls);
        sequences_.push_back(std::move(sequence));
        context.Restore();
        return sequences_.size();
    }

    void RunSequence(uint32_t id) {
        std::lock_guard<std::mutex> lock(mutex_);
        RequireOpen();
        if (id == 0 || id > sequences_.size()) throw std::invalid_argument("resident sequence id is not prepared");
        ContextGuard context(context_);
        for (const auto &call : sequences_[id - 1].calls) {
            uint64_t entry = call.slots;
            rtArgsEx_t args{};
            args.args = &entry;
            args.argsSize = sizeof(entry);
            rtTaskCfgInfo_t config{};
            config.schemMode = RT_SCHEM_MODE_BATCH;
            Check(
                rtKernelLaunchWithHandleV2(call.kernel, 0, call.blocks, &args, nullptr, stream_, &config),
                "launch sequence kernel"
            );
        }
        Check(aclrtSynchronizeStreamWithTimeout(stream_, 5000), "synchronize sequence");
        context.Restore();
    }

    void RunGraph(uint32_t id) {
        std::lock_guard<std::mutex> lock(mutex_);
        RequireOpen();
        if (id == 0 || id > sequences_.size() || !sequences_[id - 1].model)
            throw std::invalid_argument("resident graph id is not prepared");
        ContextGuard context(context_);
        sequences_[id - 1].model->Run(stream_);
        Check(aclrtSynchronizeStreamWithTimeout(stream_, 5000), "synchronize graph");
        context.Restore();
    }

    void RebuildGraph(uint32_t id) {
        std::lock_guard<std::mutex> lock(mutex_);
        RequireOpen();
        if (id == 0 || id > sequences_.size() || !sequences_[id - 1].model)
            throw std::invalid_argument("resident graph id is not prepared");
        ContextGuard context(context_);
        Check(aclrtSynchronizeStreamWithTimeout(stream_, 5000), "drain graph before rebuild");
        DrainRetiredModels();
        retired_models_.reserve(1);
        auto &sequence = sequences_[id - 1];
        auto replacement = std::make_unique<DeviceModel>(sequence.calls);
        retired_models_.push_back(std::move(sequence.model));
        sequence.model = std::move(replacement);
        DrainRetiredModels();
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
            DrainRetiredModels();
            while (!sequences_.empty()) {
                if (sequences_.back().model) sequences_.back().model->Close();
                sequences_.pop_back();
            }
            resident_.clear();
            while (!extra_kernels_.empty()) {
                Check(rtDevBinaryUnRegister(extra_kernels_.back().handle), "unregister resident kernel");
                extra_kernels_.pop_back();
            }
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
    struct PreparedCall {
        uint64_t slots;
        void *kernel;
        uint32_t blocks;
    };
    class DeviceModel {
    public:
        explicit DeviceModel(const std::vector<PreparedCall> &calls) {
            try {
                Check(aclrtCreateStream(&stream_), "create model stream");
                Check(rtModelCreate(&model_, 0), "create device model");
                Check(rtModelBindStream(model_, stream_, RT_MODEL_HEAD_STREAM), "bind model stream");
                bound_ = true;
                for (const auto &call : calls) {
                    uint64_t entry = call.slots;
                    rtArgsEx_t args{};
                    args.args = &entry;
                    args.argsSize = sizeof(entry);
                    rtTaskCfgInfo_t config{};
                    config.schemMode = RT_SCHEM_MODE_BATCH;
                    Check(
                        rtKernelLaunchWithHandleV2(call.kernel, 0, call.blocks, &args, nullptr, stream_, &config),
                        "prepare graph kernel"
                    );
                }
                Check(rtEndGraph(model_, stream_), "end device graph");
                Check(rtModelLoadComplete(model_), "load device graph");
            } catch (...) {
                if (bound_) rtModelUnbindStream(model_, stream_);
                if (model_) rtModelDestroy(model_);
                if (stream_) aclrtDestroyStream(stream_);
                throw;
            }
        }
        ~DeviceModel() {
            try {
                Close();
            } catch (...) {}
        }
        DeviceModel(const DeviceModel &) = delete;
        DeviceModel &operator=(const DeviceModel &) = delete;

        void Run(aclrtStream stream) { Check(rtModelExecute(model_, stream, 0), "execute device graph"); }
        void Close() {
            if (bound_) {
                Check(rtModelUnbindStream(model_, stream_), "unbind model stream");
                bound_ = false;
            }
            if (model_) {
                Check(rtModelDestroy(model_), "destroy device model");
                model_ = nullptr;
            }
            if (stream_) {
                Check(aclrtDestroyStream(stream_), "destroy model stream");
                stream_ = nullptr;
            }
        }

    private:
        rtModel_t model_ = nullptr;
        aclrtStream stream_ = nullptr;
        bool bound_ = false;
    };

    std::vector<std::unique_ptr<DeviceModel>> retired_models_;

    void DrainRetiredModels() {
        while (!retired_models_.empty()) {
            retired_models_.back()->Close();
            retired_models_.pop_back();
        }
    }

    struct PreparedSequence {
        DeviceAllocation descriptors;
        DeviceAllocation slots;
        std::vector<PreparedCall> calls;
        std::unique_ptr<DeviceModel> model;
    };
    std::vector<PreparedSequence> sequences_;
    struct ResidentBuffer {
        DeviceAllocation allocation;
        uint64_t bytes;
        uint32_t dtype;
    };
    struct ExtraKernel {
        std::vector<char> object;
        void *handle = nullptr;
    };
    void RequireOpen() const {
        if (!stream_ || !kernel_) {
            throw std::runtime_error("single launch session is closed");
        }
    }
    void *ResidentPointer(const ResidentArg &arg) {
        Validate(arg.tensor);
        const auto id = arg.tensor.address;
        if (!id || id > resident_.size()) {
            throw std::invalid_argument("resident tensor is not owned by this session");
        }
        const auto &buffer = resident_[id - 1];
        const uint64_t element_bytes = arg.tensor.dtype == static_cast<uint32_t>(DataType::FLOAT16) ? 2 : 4;
        if (arg.tensor.dtype != buffer.dtype || arg.offset % element_bytes || arg.tensor.bytes > buffer.bytes ||
            arg.offset > buffer.bytes - arg.tensor.bytes) {
            throw std::invalid_argument("resident tensor view exceeds its allocation or changes dtype");
        }
        return static_cast<char *>(buffer.allocation.get()) + arg.offset;
    }
    void Launch(const KernelTensor *descriptors, uint32_t count, void *kernel, uint32_t blocks) {
        const uint64_t descriptor_bytes = count * sizeof(KernelTensor);
        void *device_descriptors = Allocate(descriptor_bytes);
        Check(
            aclrtMemcpy(device_descriptors, descriptor_bytes, descriptors, descriptor_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
            "upload descriptors"
        );
        const auto slots = ArgumentSlots(
            reinterpret_cast<uint64_t>(device_descriptors), &count, 1, argument_abi_, l2_cache_info_.alias_offset
        );
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
        Check(rtKernelLaunchWithHandleV2(kernel, 0, blocks, &launch_args, nullptr, stream_, &config), "launch kernel");
        Check(aclrtSynchronizeStreamWithTimeout(stream_, 5000), "synchronize kernel");
    }
    uint32_t argument_abi_;
    L2CacheInfo l2_cache_info_{};
    std::vector<ResidentBuffer> resident_;
    std::vector<ExtraKernel> extra_kernels_;
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

int CreateSession(const void *binary, uint64_t bytes, int32_t device, void **result, uint32_t abi_version) {
    return Invoke([&] {
        if (!binary || bytes < 4 || bytes > UINT32_MAX || !result || device < 0 ||
            std::memcmp(binary, "\177ELF", 4) != 0) {
            throw std::invalid_argument("invalid A1 ELF object or device");
        }
        *result = new SingleLaunch(binary, bytes, device, abi_version);
    });
}

}  // namespace

extern "C" const char *simpler_a1_last_error() { return last_error.c_str(); }

extern "C" uint32_t simpler_a1_argument_abi_version() { return kArgumentAbiVersion; }

extern "C" int simpler_a1_l2_cache_info(void *session, uint64_t *offset, int32_t *query_status) {
    return Invoke([&] {
        if (!session || !offset || !query_status) throw std::invalid_argument("invalid A1 cache info request");
        const auto info = static_cast<SingleLaunch *>(session)->CacheInfo();
        *offset = info.alias_offset;
        *query_status = info.query_status;
    });
}

extern "C" int simpler_a1_create(const void *binary, uint64_t bytes, int32_t device, void **result) {
    return CreateSession(binary, bytes, device, result, 1);
}

extern "C" int simpler_a1_create_v2(const void *binary, uint64_t bytes, int32_t device, void **result) {
    return CreateSession(binary, bytes, device, result, kArgumentAbiVersion);
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

extern "C" int simpler_a1_tensor_create(void *session, const HostTensorArg *arg, uint32_t upload, uint64_t *id) {
    return Invoke([&] {
        if (!session || !arg || !id || upload > 1) throw std::invalid_argument("invalid resident tensor request");
        *id = static_cast<SingleLaunch *>(session)->CreateTensor(*arg, upload != 0);
    });
}
extern "C" int
simpler_a1_tensor_copy(void *session, const ResidentArg *device, const HostTensorArg *host, uint32_t upload) {
    return Invoke([&] {
        if (!session || !device || !host || upload > 1) throw std::invalid_argument("invalid resident transfer");
        static_cast<SingleLaunch *>(session)->CopyTensor(*device, *host, upload != 0);
    });
}
extern "C" int simpler_a1_register(void *session, const void *image, uint64_t bytes, uint32_t *id) {
    return Invoke([&] {
        if (!session || !id) throw std::invalid_argument("invalid resident kernel request");
        *id = static_cast<SingleLaunch *>(session)->RegisterKernel(image, bytes);
    });
}
extern "C" int
simpler_a1_run_resident(void *session, const ResidentArg *args, uint32_t count, uint32_t kernel_id, uint32_t blocks) {
    return Invoke([&] {
        if (!session) throw std::invalid_argument("single launch session is closed");
        static_cast<SingleLaunch *>(session)->RunResident(args, count, kernel_id, blocks);
    });
}

extern "C" int simpler_a1_prepare_sequence(
    void *session, const ResidentArg *args, const uint32_t *counts, const uint32_t *ids, const uint32_t *blocks,
    uint32_t count, uint32_t *result
) {
    return Invoke([&] {
        if (!session || !result) throw std::invalid_argument("invalid resident sequence request");
        *result = static_cast<SingleLaunch *>(session)->PrepareSequence(args, counts, ids, blocks, count);
    });
}
extern "C" int simpler_a1_run_sequence(void *session, uint32_t id) {
    return Invoke([&] {
        if (!session) throw std::invalid_argument("single launch session is closed");
        static_cast<SingleLaunch *>(session)->RunSequence(id);
    });
}

extern "C" int simpler_a1_prepare_graph(
    void *session, const ResidentArg *args, const uint32_t *counts, const uint32_t *ids, const uint32_t *blocks,
    uint32_t count, uint32_t *result
) {
    return Invoke([&] {
        if (!session || !result) throw std::invalid_argument("invalid resident graph request");
        *result = static_cast<SingleLaunch *>(session)->PrepareSequence(args, counts, ids, blocks, count, true);
    });
}
extern "C" int simpler_a1_run_graph(void *session, uint32_t id) {
    return Invoke([&] {
        if (!session) throw std::invalid_argument("single launch session is closed");
        static_cast<SingleLaunch *>(session)->RunGraph(id);
    });
}

extern "C" int simpler_a1_rebuild_graph(void *session, uint32_t id) {
    return Invoke([&] {
        if (!session) throw std::invalid_argument("single launch session is closed");
        static_cast<SingleLaunch *>(session)->RebuildGraph(id);
    });
}
