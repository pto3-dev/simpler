# A1 chip-swimlane clock and dispatch producer

**Date**: 2026-10-17
**Verdict**: direct A2/A3 port rejected; AICPU-observed A1 protocol verified

## Question

Can the A2/A3 chip-swimlane collector be attached directly to the A1 static
CANN device-model backend?

## What was tried

A minimal global entry compiled with `KernelCompiler("a1")` for `dav-m200`
reads `get_sys_cnt()` before and after work. The compiler rejects both reads:
`get_sys_cnt does not support the given target feature`.
An equivalent `asm volatile("MOV %0, SYS_CNT\n" : "=l"(value))` probe fails
assembly because its supported target list excludes DAV_M200. Neither
unsupported probe is launched on hardware.

The installed CANN 9.1 Ascend C implementation in
`basic_api/impl/dav_m200/kernel_operator_sys_var_impl.h` marks
`GetSystemCycleImpl` unsupported. The legacy 310P runtime's AICore timeout
implementation likewise substitutes zero for its system-counter read; this is
not a timestamp source that can be copied into the new collector.

## Result

The current A1 static model has neither an AICore system-counter producer nor
a runtime-owned AICPU early-dispatch loop. CANN HWTS provides genuine task
start/end timestamps; its task envelope is not a physical-core execution slice.
The installed 310P PMU parser consumes aggregate per-task counter records.
It does not expose the missing physical-core start/receive/end timestamps.

A separate exact Add-to-Mul hardware repro isolates zero pipe counters:
preparing the model before `aclprofStart` leaves all 10 measured task rows with
zero vector/scalar counters; preparing or rebuilding it after profiler start
makes all 10 nonzero, with unchanged resident bindings and exact results.
Total cycles can be nonzero even when the pipe-counter state was not enabled.
The explicit `A1PreparedSequence.rebuild()` operation addresses that state.

## Why not directly port the collector

Do not hardcode a 50 MHz A2/A3 clock for A1, issue an instruction for another
target, or fill per-core records from task envelopes. The converter and
critical-path analyzer interpret those records as actual physical-core slices.
Likewise, a host model submission or a dependency-derived ready time is not a
measurement of the runtime's `early_dispatch` scheduler phase.

## When to reconsider

An A1 native control backend must expose a verified AICPU clock source and
actual dispatch/finish transport for the coupled cores. Its clock frequency,
epoch and observation error must be validated before exporting per-core
scheduler envelopes. Envelopes remain distinct from unsupported AICore
receive/start/end timestamps. Port early-dispatch execution and phase
instrumentation together, initially for two dependent tasks. Retain real HWTS
and restored PMU comparisons in the meantime; native chip/early-dispatch
collection remains incomplete.

## Verified A1 observation transport

A standalone common-loader AICPU entry now verifies `cntvct_el0` and the
queried 38.4 MHz `cntfrq_el0` frequency on 310P3. ISB-ordered timestamp reads
resolve zero-length/reordered observations from an unordered MRS probe.
Separate 128-byte CPU-command and core-status lines with the native A1
two-argument DCCI handshake pass five exact dependent Add-to-Mul rounds.
The dynamic-layout producer also passes three complete YOLO26n 128x128
rounds: 216 tasks, eight physical cores, 31 byte-identical output arrays.

The supported diagnostic protocol is documented in
`docs/dfx/a1-chip-swimlane.md`. It provides actual queue-stage and
completion-observation records, with an explicit observation-envelope
schema. The direct instruction-timestamp port remains invalid; the
default static-model replay still has no AICPU dispatch producer.
