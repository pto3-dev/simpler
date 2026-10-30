# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Resident tensor geometry, ownership and native cross-image execution."""

import ctypes
import threading
from pathlib import Path

import pytest
import torch

from simpler_setup import A1DeviceTensor, A1SingleLaunch, KernelCompiler
from simpler_setup import a1_single_launch as module


@pytest.mark.parametrize("shape", [(), (0, 1), (-1,), (True,), (0x100000000,), (65536, 65536), (1, 1, 1, 1, 1, 1)])
def test_resident_geometry_rejects_invalid_extents(shape):
    with pytest.raises(ValueError, match="ABI"):
        module._geometry(shape, torch.float32)


@pytest.mark.parametrize("dtype", [torch.int16, torch.int64, torch.float64])
def test_resident_geometry_rejects_unsupported_storage_types(dtype):
    with pytest.raises(ValueError):
        module._geometry((2, 128), dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int32, torch.uint32])
def test_dense_view_geometry_and_parent_bounds(dtype):
    owner = object.__new__(A1SingleLaunch)
    tensor = A1DeviceTensor(owner, 7, (2, 128), dtype, 256 * (2 if dtype == torch.float16 else 4))
    view = tensor.view((1, 64), offset_elements=64)
    nested = view.view((1, 16), offset_elements=16)
    arg = nested._argument()
    assert ctypes.sizeof(arg) == 56
    assert arg.tensor.address == 7
    assert arg.tensor.bytes == 16 * (2 if dtype == torch.float16 else 4)
    assert arg.offset == 80 * (2 if dtype == torch.float16 else 4)
    with pytest.raises(ValueError, match="parent"):
        view.view((1, 80))
    with pytest.raises(ValueError, match="parent"):
        view.view((1, 16), 60)
    with pytest.raises(ValueError, match="offset"):
        view.view((1, 16), -1)


def test_foreign_tensors_and_invalid_launches_fail_before_dispatch():
    owner, foreign = object.__new__(A1SingleLaunch), object.__new__(A1SingleLaunch)
    owner._lock = threading.RLock()
    owner._handle = ctypes.c_void_p(1)
    tensor = A1DeviceTensor(owner, 1, (1, 16), torch.float32, 64)
    other = A1DeviceTensor(foreign, 1, (1, 16), torch.float32, 64)
    with pytest.raises(ValueError, match="owned"):
        owner.run_device(other)
    with pytest.raises(ValueError, match="owned"):
        owner.prepare_sequence([(0, 1, (other,))])
    with pytest.raises(ValueError, match="1..4096"):
        owner.prepare_sequence([])
    for blocks in [0, 9, True]:
        with pytest.raises(ValueError, match="blocks"):
            owner.run_device(tensor, blocks=blocks)
    for kernel_id in [-1, True, 0x100000000]:
        with pytest.raises(ValueError, match="kernel id"):
            owner.run_device(tensor, kernel_id=kernel_id)
    owner._handle.value = None
    with pytest.raises(RuntimeError, match="closed"):
        owner.run_device(tensor)
    with pytest.raises(RuntimeError, match="closed"):
        owner.empty((1, 16), torch.float32)
    with pytest.raises(RuntimeError, match="closed"):
        tensor.to_cpu()


@pytest.mark.requires_hardware("a1")
@pytest.mark.parametrize("prepared", [False, True, "graph"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_native_resident_multi_kernel_chain(tmp_path, dtype, prepared):
    try:
        acl = ctypes.CDLL("libascendcl.so")
    except OSError:
        pytest.skip("Ascend runtime unavailable")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    soc = acl.aclrtGetSocName()
    if not soc or not soc.startswith(b"Ascend310P"):
        pytest.skip("requires Ascend310P")
    root = Path(__file__).resolve().parents[3]
    isa = root / "build/pto-isa"
    source = (Path(__file__).parent / "kernels/a1_resident_chain.cpp").read_text()
    if dtype == torch.float16:
        source = source.replace("using Data = float;", "using Data = half;")
    add_file, mul_file = tmp_path / "add.cpp", tmp_path / "mul.cpp"
    add_file.write_text(source)
    mul_file.write_text("#define RESIDENT_MULTIPLY\n" + source)
    compiler = KernelCompiler("a1")
    add = compiler.compile_single_launch(str(add_file), str(isa))
    mul = compiler.compile_single_launch(str(mul_file), str(isa))
    with A1SingleLaunch(add, device_id=0) as runner:
        a = runner.upload(torch.zeros((1, 2048), dtype=dtype))
        b = runner.empty((1, 2048), dtype)
        storage = runner.upload(torch.full((1, 2112), 7.0, dtype=dtype))
        out = storage.view((1, 2048), offset_elements=32)
        multiply = runner.register_kernel(mul)
        assert multiply == 1
        sequence = None
        if prepared:
            prepare = runner.prepare_graph if prepared == "graph" else runner.prepare_sequence
            sequence = prepare([(0, 8, (a, b, out)), (multiply, 8, (out, b, out))])
            with pytest.raises(RuntimeError, match="signature or kernel"):
                runner.prepare_sequence([(99, 8, (a, b, out))])
        for seed in [223, 227]:
            generator = torch.Generator().manual_seed(seed)
            x = (torch.randint(-8, 9, (1, 2048), generator=generator).float() / 16).to(dtype)
            y = (torch.randint(-8, 9, (1, 2048), generator=generator).float() / 16).to(dtype)
            expected = ((x + y) * y).to(dtype)
            a.copy_from(x)
            b.copy_from(y)
            x.zero_()
            y.zero_()
            if sequence is None:
                runner.run_device(a, b, out, blocks=8)
                runner.run_device(out, b, out, kernel_id=multiply, blocks=8)
            else:
                sequence()
            actual = storage.to_cpu()
            assert torch.equal(actual[:, 32:2080], expected)
            assert torch.equal(actual[:, :32], torch.full((1, 32), 7.0, dtype=dtype))
            assert torch.equal(actual[:, 2080:], torch.full((1, 32), 7.0, dtype=dtype))
        if prepared == "graph":
            for repeats in [16, 108]:
                graph = runner.prepare_graph([(0, 8, (a, b, out)), (multiply, 8, (out, b, out))] * repeats)
                for _ in range(2):
                    graph()
                    assert torch.equal(out.to_cpu(), expected)
        with pytest.raises(RuntimeError, match="registered"):
            runner.run_device(a, b, out, kernel_id=99)
        invalid = out._argument()
        invalid.offset = storage._capacity
        args = (module._ResidentArg * 1)(invalid)
        with pytest.raises(RuntimeError, match="exceeds"):
            runner._check(runner._library.simpler_a1_run_resident(runner._handle, args, 1, 0, 1))
        # Legacy CPU-carrier calls must preserve resident allocations too.
        legacy = torch.zeros((16, 128), dtype=dtype)
        runner(torch.ones_like(legacy), torch.ones_like(legacy), legacy)
        assert torch.equal(legacy.reshape(-1)[:256], torch.full((256,), 2.0, dtype=dtype))
        assert torch.equal(legacy.reshape(-1)[256:], torch.zeros(1792, dtype=dtype))
        assert torch.equal(out.to_cpu(), expected)
    with pytest.raises(RuntimeError, match="closed"):
        out.to_cpu()
    if sequence is not None:
        with pytest.raises(RuntimeError, match="closed"):
            sequence()
