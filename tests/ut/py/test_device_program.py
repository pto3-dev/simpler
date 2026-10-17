# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import struct
from dataclasses import replace

import pytest

from simpler_setup.device_program import Invocation, LeafSignature, compile_device_program

LEAVES = (LeafSignature(0, "aic", 2), LeafSignature(1, "aiv", 2))
A = Invocation("a", 0, "aic", 0, 0, 2, (4, 5))
B = Invocation("b", 1, "aiv0", 0, 0, 2, (5, 6), ("a",))
C = Invocation("c", 0, "aic", 1, 1, 2, (4, 5), ("b",))


def compile_plan(nodes, **kw):
    return compile_device_program(nodes, LEAVES, grid=2, available_clusters=2, root_arg_count=8, epoch_arg=7, **kw)


def test_real_tile_progress_and_relocatable_image():
    p = compile_plan((A, C, B))
    h = struct.unpack_from("<16I", p.image)
    assert h[0] == 0x44505247 and h[2] == len(p.image)
    assert h[4] == 3 and h[7] == 3 and h[6] == 2
    assert p.frame_bytes == 3 * 2 * 512 and p.signal_bytes == 3 * 128
    assert p.event_names == ("a", "c", "b")
    # AIC tile 1 awaits vector tile 0; no whole-operation join is introduced.
    assert compile_plan((A, C, B)).image == p.image


@pytest.mark.parametrize(
    "nodes",
    [
        (replace(A, after=("b",)), B),
        (replace(A, after=("c",)), replace(C, owner=0, after=("a",))),
    ],
)
def test_reject_dependency_or_physical_queue_cycles(nodes):
    with pytest.raises(ValueError, match="cycle"):
        compile_plan(nodes)


@pytest.mark.parametrize(
    "node",
    [
        replace(A, func_id=99),
        replace(A, lane="aiv0"),
        replace(A, owner=2),
        replace(A, logical_block=2),
        replace(A, arguments=(0, 5)),
        replace(A, arguments=(4, 7)),
        replace(A, arguments=(4,)),
        replace(A, after=("unknown",)),
        replace(B, after=("a", "a")),
    ],
)
def test_reject_invalid_executable_or_binding(node):
    with pytest.raises(ValueError):
        compile_plan((node,))


def test_no_resident_oversubscription():
    with pytest.raises(ValueError, match="resident"):
        compile_device_program((A,), LEAVES, grid=2, available_clusters=1, root_arg_count=8, epoch_arg=7)


def test_same_core_order_needs_no_poll():
    a2 = replace(A, name="next", after=("a",))
    h = struct.unpack_from("<16I", compile_plan((A, a2)).image)
    assert h[6] == 0


def test_async_work_is_rejected():
    with pytest.raises(ValueError, match="asynchronous"):
        compile_device_program(
            (A,), (replace(LEAVES[0], synchronous=False),), grid=2, available_clusters=2, root_arg_count=8, epoch_arg=7
        )


def test_tracing_is_opt_in():
    assert struct.unpack_from("<16I", compile_plan((A,)).image)[15] == 0
    assert struct.unpack_from("<16I", compile_plan((A,), record_trace=True).image)[15] == 1
