"""Regression test for pytorch/pytorch#180908.

`RemoveLayoutConversions`'s cleanup phase uses MLIR's greedy pattern rewriter
with a default `maxIterations=10`. Kernels with ~10+ unrolled `scf.if` regions
holding loop-carried tensor values (e.g. `tl.static_range` with a runtime
guard inside) need 11+ iterations to converge. Before the fix, hitting the
limit caused `signalPassFailure()` to abort the compile with the cryptic
`RuntimeError: PassManager::run failed`. This test compiles such a kernel
and asserts no exception.
"""

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def _static_range_with_runtime_guard(
    out_ptr,
    vals_ptr,
    pos_ptr,
    N: tl.constexpr,
    MAX_ITER: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, N)
    pos = tl.load(pos_ptr + pid)
    acc = tl.zeros((N, ), dtype=tl.int32)
    for i in tl.static_range(MAX_ITER):
        if i < pos:
            v = tl.load(vals_ptr + pid * MAX_ITER + i)
            acc += (offs == v).to(tl.int32)
    tl.store(out_ptr + pid * N + offs, acc)


def test_remove_layout_conversions_converges_with_unrolled_scf_if() -> None:
    """Compile a kernel with 10 unrolled scf.if blocks. Must not raise."""
    signature = {
        "out_ptr": "*i32",
        "vals_ptr": "*i32",
        "pos_ptr": "*i32",
        "N": "constexpr",
        "MAX_ITER": "constexpr",
    }
    constexprs = {"N": 8192, "MAX_ITER": 10}
    # Pointer divisibility hints drive the vectorization that produces the
    # cleanup-stressing IR; mirror what the JIT path passes.
    attrs = {
        (0, ): [["tt.divisibility", 16]],
        (1, ): [["tt.divisibility", 16]],
        (2, ): [["tt.divisibility", 16]],
        (3, ): [],
        (4, ): [],
    }
    src = ASTSource(
        fn=_static_range_with_runtime_guard,
        signature=signature,
        constexprs=constexprs,
        attrs=attrs,
    )
    # Compile for sm_89 explicitly — bug is arch-agnostic in our minimal
    # repro, but this matches the original report (vLLM on L4).
    triton.compile(
        src,
        target=GPUTarget("cuda", 89, 32),
        options={
            "num_warps": 4,
            "num_ctas": 1,
            "num_stages": 3,
            "warp_size": 32,
            "arch": "sm89",
        },
    )
