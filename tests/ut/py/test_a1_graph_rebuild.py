# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepared-model rebuild ownership and failure contracts."""

import ctypes
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from simpler_setup import KernelCompiler
from simpler_setup.a1_single_launch import A1PreparedSequence, A1SingleLaunch


def _graph(*, device_model=True, closed=False):
    events = []
    library = SimpleNamespace(
        simpler_a1_rebuild_graph=lambda handle, index: events.append((handle.value, index)) or 0,
        simpler_a1_run_graph=lambda handle, index: events.append(("run", index)) or 0,
        simpler_a1_close=lambda handle: 0,
    )

    def check(status):
        if status:
            raise RuntimeError("native rebuild failed")

    owner = A1SingleLaunch.__new__(A1SingleLaunch)
    owner._lock = threading.RLock()
    owner._handle = ctypes.c_void_p(None if closed else 123)
    owner._library = library
    owner._check = check
    return A1PreparedSequence(owner, 7, (), device_model=device_model), owner, events


def test_rebuild_retains_graph_identity_and_does_not_execute():
    graph, _, events = _graph()
    graph.rebuild()
    assert events == [(123, 7)]
    graph()
    assert events == [(123, 7), ("run", 7)]


@pytest.mark.parametrize("device_model,closed", [(False, False), (True, True)])
def test_rebuild_rejects_host_sequences_and_closed_sessions(device_model, closed):
    graph, _, events = _graph(device_model=device_model, closed=closed)
    with pytest.raises((ValueError, RuntimeError), match="device model|closed"):
        graph.rebuild()
    assert events == []


def test_rebuild_reports_native_failure_without_invalidating_owner():
    graph, owner, events = _graph()
    owner._library.simpler_a1_rebuild_graph = lambda handle, index: 1
    with pytest.raises(RuntimeError, match="native rebuild failed"):
        graph.rebuild()
    assert owner._handle.value == 123
    assert events == []


@pytest.mark.requires_hardware("a1")
@pytest.mark.parametrize("dtype_name", ["float16", "float32"])
def test_native_rebuild_preserves_dependent_bindings(tmp_path, dtype_name):
    dtype = getattr(torch, dtype_name)
    root = Path(__file__).resolve().parents[3]
    source = (Path(__file__).parent / "kernels/a1_resident_chain.cpp").read_text()
    if dtype == torch.float16:
        source = source.replace("using Data = float;", "using Data = half;")
    add_file, mul_file = tmp_path / "add.cpp", tmp_path / "mul.cpp"
    add_file.write_text(source)
    mul_file.write_text("#define RESIDENT_MULTIPLY\n" + source)
    compiler = KernelCompiler("a1")
    isa = str(root / "build/pto-isa")
    with A1SingleLaunch(compiler.compile_single_launch(str(add_file), isa)) as runner:
        multiply = runner.register_kernel(compiler.compile_single_launch(str(mul_file), isa))
        a = runner.empty((1, 2048), dtype)
        b = runner.empty((1, 2048), dtype)
        guarded = runner.upload(torch.full((1, 2112), 7.0, dtype=dtype))
        out = guarded.view((1, 2048), 32)
        graph = runner.prepare_graph([(0, 8, (a, b, out)), (multiply, 8, (out, b, out))])
        identity = graph._sequence_id
        for seed in [313, 317]:
            generator = torch.Generator().manual_seed(seed)
            x = torch.randint(-8, 9, (1, 2048), generator=generator).to(dtype) / 16
            y = torch.randint(-8, 9, (1, 2048), generator=generator).to(dtype) / 16
            a.copy_from(x)
            b.copy_from(y)
            graph.rebuild()
            assert graph._sequence_id == identity
            graph()
            actual = guarded.to_cpu()
            assert torch.equal(actual[:, 32:2080], (x + y) * y)
            assert torch.equal(actual[:, :32], torch.full((1, 32), 7.0, dtype=dtype))
            assert torch.equal(actual[:, 2080:], torch.full((1, 32), 7.0, dtype=dtype))
