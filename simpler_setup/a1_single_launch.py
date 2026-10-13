# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Synchronous execution of one coupled-core entry on Ascend310P."""

import ctypes
import platform
import subprocess
import threading
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from simpler import env_manager

from .environment import PROJECT_ROOT
from .toolchain import GxxToolchain


class _HostTensorArg(ctypes.Structure):
    _fields_ = [
        ("address", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("dtype", ctypes.c_uint32),
        ("rank", ctypes.c_uint32),
        ("shape", ctypes.c_uint32 * 5),
    ]


@lru_cache(maxsize=1)
def _load_runtime() -> Any:
    env_manager.ensure("ASCEND_HOME_PATH")
    toolchain = GxxToolchain()
    if toolchain.ascend_home_path is None:
        raise RuntimeError("ASCEND_HOME_PATH is required for A1 single launch")
    sdk = Path(toolchain.ascend_home_path)
    common = PROJECT_ROOT / "src" / "common"
    source = PROJECT_ROOT / "src" / "a1" / "platform" / "onboard" / "host" / "single_launch.cpp"
    with TemporaryDirectory(prefix="simpler_a1_") as directory:
        output = str(Path(directory) / "single_launch.so")
        cmd = [
            toolchain.cxx_path,
            *toolchain.get_compile_flags(),
            f"-I{sdk / 'include'}",
            f"-I{sdk / 'pkg_inc'}",
            f"-I{sdk / 'pkg_inc' / 'runtime'}",
            f"-I{sdk / (platform.machine() + '-linux') / 'pkg_inc'}",
            f"-I{sdk / (platform.machine() + '-linux') / 'pkg_inc' / 'runtime'}",
            f"-I{common}",
            f"-I{common / 'task_interface'}",
            str(source),
            str(common / "task_interface" / "assert_compat.cpp"),
            f"-L{sdk / 'lib64'}",
            f"-Wl,-rpath,{sdk / 'lib64'}",
            "-lascendcl",
            "-lruntime",
            "-ldl",
            "-pthread",
            "-o",
            output,
        ]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=PROJECT_ROOT)
        if result.returncode:
            raise RuntimeError(f"A1 host runtime compilation failed:\n{result.stderr}")
        library = ctypes.CDLL(output)
    library.simpler_a1_create.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.simpler_a1_create.restype = ctypes.c_int
    library.simpler_a1_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(_HostTensorArg), ctypes.c_uint32]
    library.simpler_a1_run.restype = ctypes.c_int
    library.simpler_a1_close.argtypes = [ctypes.c_void_p]
    library.simpler_a1_close.restype = ctypes.c_int
    library.simpler_a1_last_error.argtypes = []
    library.simpler_a1_last_error.restype = ctypes.c_char_p
    return library


def _tensor_arg(tensor: Any) -> _HostTensorArg:
    import torch  # noqa: PLC0415

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("A1 single launch accepts torch tensors")
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("A1 single launch requires contiguous CPU tensors")
    if tensor.dtype not in (torch.float16, torch.float32):
        raise ValueError("A1 single launch requires FP16 or FP32 tensors")
    shape = list(tensor.shape)
    if not 1 <= len(shape) <= 5 or any(not 0 < size <= 0xFFFFFFFF for size in shape):
        raise ValueError("tensor shape exceeds the A1 kernel ABI")
    if tensor.numel() > 0xFFFFFFFF:
        raise ValueError("tensor element count exceeds the A1 kernel ABI")
    return _HostTensorArg(
        tensor.data_ptr(),
        tensor.numel() * tensor.element_size(),
        1 if tensor.dtype == torch.float16 else 0,
        len(shape),
        (ctypes.c_uint32 * 5)(*(shape + [0] * (5 - len(shape)))),
    )


class A1SingleLaunch:
    """Run a single ``__global__`` kernel with a tensor-pointer-array ABI.

    ``kernel`` is the whole ELF object returned by
    ``KernelCompiler('a1').compile_single_launch``. The entry accepts one GM
    pointer to an array of tensor descriptor addresses. Arguments must match
    the kernel's static signature. Calls are synchronous and copy every
    contiguous CPU tensor back. Exact aliases share device storage; partially
    overlapping views are rejected. Each session owns its context and stream,
    without resetting the device or finalizing the process-wide ACL runtime.
    """

    def __init__(self, kernel: bytes, device_id: int = 0):
        if not isinstance(kernel, bytes) or not kernel.startswith(b"\x7fELF"):
            raise ValueError("kernel must be a CANN-loadable ELF object")
        if isinstance(device_id, bool) or not isinstance(device_id, int) or not 0 <= device_id <= 0x7FFFFFFF:
            raise ValueError("device_id must be a nonnegative int32")
        self._lock = threading.RLock()
        self._handle = ctypes.c_void_p()
        self._library = _load_runtime()
        image = ctypes.create_string_buffer(kernel)
        self._check(self._library.simpler_a1_create(image, len(kernel), device_id, ctypes.byref(self._handle)))

    def _check(self, status: int) -> None:
        if status:
            error = self._library.simpler_a1_last_error()
            raise RuntimeError(error.decode() if error else "A1 runtime operation failed")

    def __call__(self, *tensors: Any) -> None:
        """Run the entry and synchronously update all argument tensors."""
        if not 1 <= len(tensors) <= 64:
            raise ValueError("tensor argument count must be in 1..64")
        args = (_HostTensorArg * len(tensors))(*(_tensor_arg(tensor) for tensor in tensors))
        with self._lock:
            if not self._handle.value:
                raise RuntimeError("single launch session is closed")
            self._check(self._library.simpler_a1_run(self._handle, args, len(args)))

    def close(self) -> None:
        """Drain and release session resources; a failed drain remains retryable."""
        with self._lock:
            if self._handle.value:
                self._check(self._library.simpler_a1_close(self._handle))
                self._handle.value = None

    def __enter__(self) -> "A1SingleLaunch":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self):
        if hasattr(self, "_library"):
            try:
                self.close()
            except RuntimeError:
                pass
