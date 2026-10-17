# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import subprocess

import pytest

from simpler_setup.static_program import (
    StaticNode,
    StaticStep,
    plan_static_region,
    render_static_region,
    render_static_sequence,
)


def leaf(tmp_path, name="leaf"):
    path = tmp_path / (name + ".cpp")
    path.write_text("""#include <cstdint>
#define __aicore__
#define __gm__
// --- Kernel entry point ---
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    *reinterpret_cast<int64_t*>(args[1]) = *reinterpret_cast<int64_t*>(args[0]) + 1;
}
""")
    return path


def test_generated_sequence_executes_mapped_arguments(tmp_path):
    src = leaf(tmp_path)
    steps = (StaticStep("a", src, (0, 1)), StaticStep("b", src, (1, 2)))
    generated = render_static_sequence(steps)
    (tmp_path / "aicore").mkdir()
    (tmp_path / "aicore/aicore.h").write_text("#define __aicore__\n#define __gm__\n")
    path = tmp_path / "combined.cpp"
    path.write_text(
        generated
        + """
int main() {
        int64_t x=7,y=0,z=0;
        int64_t args[]={reinterpret_cast<int64_t>(&x),reinterpret_cast<int64_t>(&y),reinterpret_cast<int64_t>(&z)};
        kernel_entry(args);
        return y==8 && z==9 ? 0 : 1;
    }"""
    )
    subprocess.run(
        ["g++", "-std=c++17", "-D__CPU_SIM", "-I" + str(tmp_path), str(path), "-o", str(tmp_path / "test")], check=True
    )
    subprocess.run([str(tmp_path / "test")], check=True)


def test_fork_join_order(tmp_path):
    p = leaf(tmp_path)
    nodes = (
        StaticNode(StaticStep("join", p, (0, 1)), "aic", ("left", "right")),
        StaticNode(StaticStep("left", p, (0, 1)), "aiv0"),
        StaticNode(StaticStep("right", p, (0, 1)), "aiv1"),
    )
    assert [n.step.name for n in plan_static_region(nodes)] == ["left", "right", "join"]
    source = render_static_region(nodes, "aic", state_arg=2, epoch_arg=3)
    assert source.count("for (int peer") == 2
    assert "MAKE_ACK" not in source and "MAKE_FIN" not in source
    assert "AICORE_EXIT_SIGNAL" in source and "defer_error" in source


@pytest.mark.parametrize("kind", ["empty", "duplicate", "unknown", "cycle", "lane", "duplicate_dep"])
def test_reject_invalid_regions(tmp_path, kind):
    p = leaf(tmp_path)
    a = StaticStep("a", p, (0, 1))
    b = StaticStep("b", p, (0, 1))
    cases = {
        "empty": (),
        "duplicate": (StaticNode(a, "aic"), StaticNode(a, "aic")),
        "unknown": (StaticNode(a, "aic", ("missing",)),),
        "cycle": (StaticNode(a, "aic", ("b",)), StaticNode(b, "aiv0", ("a",))),
        "lane": (StaticNode(a, "wrong"),),
        "duplicate_dep": (StaticNode(a, "aic"), StaticNode(b, "aic", ("a", "a"))),
    }
    with pytest.raises(ValueError):
        plan_static_region(cases[kind])


@pytest.mark.parametrize(
    "arguments,index,count", [((0,), None, None), ((-1, 1), None, None), ((0, 1), 2, 2), ((0, 1), None, 1)]
)
def test_reject_invalid_bindings(tmp_path, arguments, index, count):
    with pytest.raises(ValueError):
        render_static_sequence((StaticStep("a", leaf(tmp_path), arguments, index, count),))


def test_logical_block_specialization(tmp_path):
    p = leaf(tmp_path)
    p.write_text(p.read_text().replace(" + 1", " + get_block_idx(args) + get_block_num(args)"))
    src = render_static_sequence((StaticStep("a", p, (0, 1), 1, 4),))
    assert " + 1 + 4" in src and "get_block_idx" not in src


def test_same_block_local_dependency_needs_no_poll(tmp_path):
    p = leaf(tmp_path)
    nodes = (
        StaticNode(StaticStep("a", p, (0, 1)), "aic"),
        StaticNode(StaticStep("b", p, (0, 1)), "aic", ("a",), "same_block"),
    )
    source = render_static_region(nodes, "aic", state_arg=2, epoch_arg=3)
    assert "for (int peer" not in source


def test_same_block_cross_lane_dependency(tmp_path):
    p = leaf(tmp_path)
    nodes = (
        StaticNode(StaticStep("a", p, (0, 1)), "aic"),
        StaticNode(StaticStep("b", p, (0, 1)), "aiv0", ("a",), "same_block"),
    )
    source = render_static_region(nodes, "aiv0", state_arg=2, epoch_arg=3)
    assert "peer = block; peer < block + 1" in source


@pytest.mark.parametrize(
    "kwargs",
    [
        {"state_arg": 0, "epoch_arg": 3},
        {"state_arg": 2, "epoch_arg": 2},
        {"state_arg": 2, "epoch_arg": 3, "trace_arg": 2},
        {"state_arg": 2, "epoch_arg": 3, "max_wait_cycles": 0},
    ],
)
def test_reject_control_binding_aliases(tmp_path, kwargs):
    p = leaf(tmp_path)
    nodes = (StaticNode(StaticStep("a", p, (0, 1)), "aic"),)
    with pytest.raises(ValueError):
        render_static_region(nodes, "aic", **kwargs)
