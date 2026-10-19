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

#include <pto/pto-inst.hpp>
#include "tensor.h"

using Data = float;
using Tile = pto::Tile<pto::TileType::Vec, Data, 16, 128>;
using Global = pto::GlobalTensor<Data, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<1, 1, 1, 128, 1>>;

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t *args) {
    auto *lhs = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *rhs = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *out = reinterpret_cast<__gm__ Tensor *>(args[2]);
    Tile a, b, c;
    pto::TASSIGN<0>(a);
    pto::TASSIGN<8192>(b);
    pto::TASSIGN<16384>(c);
    Global ga(reinterpret_cast<__gm__ Data *>(lhs->buffer.addr));
    Global gb(reinterpret_cast<__gm__ Data *>(rhs->buffer.addr));
    Global gc(reinterpret_cast<__gm__ Data *>(out->buffer.addr));
    pto::TLOAD(a, ga);
    pto::Event<pto::Op::TLOAD, pto::Op::TADD> loaded = pto::TLOAD(b, gb);
    pto::Event<pto::Op::TADD, pto::Op::TSTORE_VEC> added = pto::TADD(c, a, b, loaded);
    pto::TSTORE(gc, c, added);
}
