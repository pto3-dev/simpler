# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Argument and lifetime contracts of the synchronous A1 launcher."""

import ctypes
from pathlib import Path

import pytest
import torch

from simpler_setup import a1_single_launch


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_host_tensor_geometry_matches_native_descriptor(dtype):
    tensor = torch.empty((16, 128), dtype=dtype)
    arg = a1_single_launch._tensor_arg(tensor)
    assert ctypes.sizeof(arg) == 48
    assert arg.address == tensor.data_ptr()
    assert arg.bytes == tensor.numel() * tensor.element_size()
    assert arg.rank == 2
    assert list(arg.shape) == [16, 128, 0, 0, 0]
    assert arg.dtype == (1 if dtype == torch.float16 else 0)


@pytest.mark.parametrize(
    "tensor",
    [torch.empty((8, 8)).t(), torch.empty((0, 128)), torch.tensor(1.0), torch.empty((16, 128), dtype=torch.bfloat16)],
)
def test_unsupported_carriers_fail_before_native_dispatch(tensor):
    with pytest.raises(ValueError):
        a1_single_launch._tensor_arg(tensor)


def test_session_close_failure_preserves_retryable_handle(monkeypatch):
    class Native:
        calls = 0

        def simpler_a1_create(self, _image, _size, _device, handle):
            handle._obj.value = 123
            return 0

        def simpler_a1_close(self, _handle):
            self.calls += 1
            return 1 if self.calls == 1 else 0

        def simpler_a1_last_error(self):
            return b"drain stream before close: timeout"

    native = Native()
    monkeypatch.setattr(a1_single_launch, "_load_runtime", lambda: native)
    runner = a1_single_launch.A1SingleLaunch(b"\x7fELF")
    with pytest.raises(RuntimeError, match="timeout"):
        runner.close()
    assert runner._handle.value == 123
    runner.close()
    assert runner._handle.value is None
    runner.close()
    assert native.calls == 2
    with pytest.raises(RuntimeError, match="closed"):
        runner(torch.empty((16, 128)))


def test_invalid_object_does_not_initialize_acl(monkeypatch):
    def unexpected_load():
        pytest.fail("invalid ELF must not load the native runtime")

    monkeypatch.setattr(a1_single_launch, "_load_runtime", unexpected_load)
    with pytest.raises(ValueError, match="ELF"):
        a1_single_launch.A1SingleLaunch(b"not an ELF")


@pytest.mark.requires_hardware("a1")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_native_alias_context_and_reuse(pytestconfig, tmp_path, dtype):
    if pytestconfig.getoption("--platform") != "a1":
        pytest.skip("requires explicit A1 hardware selection")
    from simpler_setup import ensure_pto_isa_root  # noqa: PLC0415

    isa_root = ensure_pto_isa_root()
    if not (Path(isa_root) / "include" / "pto" / "npu" / "a1").is_dir():
        pytest.skip("the runtime ISA pin does not yet include the A1 profile")
    _run_native_profile(tmp_path, dtype, isa_root, int(pytestconfig.getoption("--device")))


def _run_native_profile(tmp_path, dtype, isa_root, device_id):
    from simpler_setup import KernelCompiler  # noqa: PLC0415

    source = (Path(__file__).parent / "kernels" / "a1_single_launch.cpp").read_text()
    if dtype == torch.float16:
        source = source.replace("using Data = float;", "using Data = half;")
    entry = tmp_path / "entry.cpp"
    entry.write_text(source)
    image = KernelCompiler("a1").compile_single_launch(str(entry), isa_root)
    lhs = (torch.arange(2048).reshape(16, 128) % 17 - 8).to(dtype)
    rhs = ((torch.arange(2048).reshape(16, 128) * 3) % 17 - 8).to(dtype)
    out = torch.full_like(lhs, float("nan"))
    with a1_single_launch.A1SingleLaunch(image, device_id) as first:
        first(lhs, rhs, out)
        torch.testing.assert_close(out, lhs + rhs, rtol=0, atol=0)
        context = ctypes.c_void_p()
        query = first._library.aclrtGetCurrentContext
        query.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        query.restype = ctypes.c_int
        assert query(ctypes.byref(context)) == 0
        caller_context = context.value
        expected = lhs + rhs
        with a1_single_launch.A1SingleLaunch(image, device_id) as second:
            second(lhs, rhs, lhs)
            torch.testing.assert_close(lhs, expected, rtol=0, atol=0)
            assert query(ctypes.byref(context)) == 0
            assert context.value == caller_context
        assert query(ctypes.byref(context)) == 0
        assert context.value == caller_context
        storage = torch.empty(4096, dtype=dtype)
        with pytest.raises(RuntimeError, match="partially overlapping"):
            first(storage[:2048].reshape(16, 128), storage[128:2176].reshape(16, 128), out)
        first(lhs, rhs, out)
        torch.testing.assert_close(out, lhs + rhs, rtol=0, atol=0)
