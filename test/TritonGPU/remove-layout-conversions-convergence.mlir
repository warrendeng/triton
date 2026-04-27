// Regression test for pytorch/pytorch#180908.
//
// Before the fix, RemoveLayoutConversions called signalPassFailure() (with no
// diagnostic) when MLIR's greedy rewriter exhausted its iteration cap on
// cleanup — surfaced to the user as the cryptic "RuntimeError: PassManager::
// run failed". The fix:
//   - Replaces signalPassFailure with a one-shot-per-process warning to
//     stderr (bypassing MLIR's diagnostic handler, which silently filters
//     warnings unless MLIR_ENABLE_DIAGNOSTICS=warnings is set), so partially-
//     cleaned IR continues downstream rather than crashing the compile.
//   - Exposes TRITON_LAYOUT_CLEANUP_MAX_ITERATIONS as an opt-in knob to raise
//     the iteration cap above MLIR's default of 10.
//
// Two runs cover both behaviors:
//   1. Forced bailout (cap=1): assert pass succeeds, IR is emitted, warning
//      naming the override knob appears on stderr.
//   2. Within-cap (cap=64): same trivial IR converges easily; assert pass
//      succeeds and NO warning is emitted on stderr.

// RUN: TRITON_LAYOUT_CLEANUP_MAX_ITERATIONS=1 triton-opt %s \
// RUN:   -tritongpu-remove-layout-conversions 2>%t.bailout-stderr | FileCheck %s
// RUN: FileCheck %s --check-prefix=BAILOUT --input-file=%t.bailout-stderr

// RUN: TRITON_LAYOUT_CLEANUP_MAX_ITERATIONS=64 triton-opt %s \
// RUN:   -tritongpu-remove-layout-conversions 2>%t.silent-stderr | FileCheck %s
// RUN: FileCheck %s --check-prefix=NOWARN --input-file=%t.silent-stderr \
// RUN:   --allow-empty

#layout0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#layout1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// CHECK-LABEL: tt.func @bailout_emits_warning
// CHECK: tt.return
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @bailout_emits_warning() -> tensor<1024xi32, #layout1> {
    %cst = arith.constant dense<0> : tensor<1024xi32, #layout0>
    %1 = ttg.convert_layout %cst : tensor<1024xi32, #layout0> -> tensor<1024xi32, #layout1>
    tt.return %1: tensor<1024xi32, #layout1>
  }
}

// BAILOUT: warning: RemoveLayoutConversions: cleanup did not converge in 1 iterations
// BAILOUT-SAME: TRITON_LAYOUT_CLEANUP_MAX_ITERATIONS

// NOWARN-NOT: did not converge
// NOWARN-NOT: warning
