// RUN: circt-opt %s -verify-diagnostics | FileCheck %s

// Includes HW dialect
module {
// CHECK-LABEL: hw.module @MuxExamples
hw.module @MuxExamples(in %select: i1, in %input0: i1, in %input1: i1,
                      out out_muxf7: i1, out out_muxf8: i1, out out_muxf9: i1) {
  // Basic MUXF7 test
  // CHECK: %{{.+}} = xlnx.muxf7(S : %select, I0 : %input0, I1 : %input1) : i1, i1, i1 -> i1
  %0 = xlnx.muxf7(S: %select, I0: %input0, I1: %input1) : i1, i1, i1 -> i1

  // Basic MUXF8 test
  // CHECK: %{{.+}} = xlnx.muxf8(S : %select, I0 : %input0, I1 : %input1) : i1, i1, i1 -> i1
  %1 = xlnx.muxf8(S: %select, I0: %input0, I1: %input1) : i1, i1, i1 -> i1

  // Basic MUXF9 test
  // CHECK: %{{.+}} = xlnx.muxf9(S : %select, I0 : %input0, I1 : %input1) : i1, i1, i1 -> i1
  %2 = xlnx.muxf9(S: %select, I0: %input0, I1: %input1) : i1, i1, i1 -> i1

  hw.output %0, %1, %2 : i1, i1, i1
}

// CHECK-LABEL: hw.module @CascadedMux
hw.module @CascadedMux(in %select1: i1, in %select2: i1, in %select3: i1,
                      in %a: i1, in %b: i1, in %c: i1, in %d: i1,
                      out out: i1) {
  // Cascades multiple MUX operations to create a more complex multiplexer circuit

  // First level MUXF7
  // CHECK: %{{.+}} = xlnx.muxf7(S : %select1, I0 : %a, I1 : %b) : i1, i1, i1 -> i1
  %mux1 = xlnx.muxf7(S: %select1, I0: %a, I1: %b) : i1, i1, i1 -> i1

  // CHECK: %{{.+}} = xlnx.muxf7(S : %select1, I0 : %c, I1 : %d) : i1, i1, i1 -> i1
  %mux2 = xlnx.muxf7(S: %select1, I0: %c, I1: %d) : i1, i1, i1 -> i1

  // Second level MUXF8
  // CHECK: %{{.+}} = xlnx.muxf8(S : %select2, I0 : %{{.+}}, I1 : %{{.+}}) : i1, i1, i1 -> i1
  %mux3 = xlnx.muxf8(S: %select2, I0: %mux1, I1: %mux2) : i1, i1, i1 -> i1

  hw.output %mux3 : i1
}

// CHECK-LABEL: hw.module @ComplexMuxTree
hw.module @ComplexMuxTree(in %select1: i1, in %select2: i1, in %select3: i1,
                         in %a: i1, in %b: i1, in %c: i1, in %d: i1,
                         in %e: i1, in %f: i1, in %g: i1, in %h: i1,
                         out out: i1) {
  // Creates a complex MUX tree, simulating an 8-1 multiplexer

  // First level - 4 MUXF7s
  // CHECK: %{{.+}} = xlnx.muxf7(S : %select1, I0 : %a, I1 : %b) : i1, i1, i1 -> i1
  %mux1 = xlnx.muxf7(S: %select1, I0: %a, I1: %b) : i1, i1, i1 -> i1

  // CHECK: %{{.+}} = xlnx.muxf7(S : %select1, I0 : %c, I1 : %d) : i1, i1, i1 -> i1
  %mux2 = xlnx.muxf7(S: %select1, I0: %c, I1: %d) : i1, i1, i1 -> i1

  // CHECK: %{{.+}} = xlnx.muxf7(S : %select1, I0 : %e, I1 : %f) : i1, i1, i1 -> i1
  %mux3 = xlnx.muxf7(S: %select1, I0: %e, I1: %f) : i1, i1, i1 -> i1

  // CHECK: %{{.+}} = xlnx.muxf7(S : %select1, I0 : %g, I1 : %h) : i1, i1, i1 -> i1
  %mux4 = xlnx.muxf7(S: %select1, I0: %g, I1: %h) : i1, i1, i1 -> i1

  // Second level - 2 MUXF8s
  // CHECK: %{{.+}} = xlnx.muxf8(S : %select2, I0 : %{{.+}}, I1 : %{{.+}}) : i1, i1, i1 -> i1
  %mux5 = xlnx.muxf8(S: %select2, I0: %mux1, I1: %mux2) : i1, i1, i1 -> i1

  // CHECK: %{{.+}} = xlnx.muxf8(S : %select2, I0 : %{{.+}}, I1 : %{{.+}}) : i1, i1, i1 -> i1
  %mux6 = xlnx.muxf8(S: %select2, I0: %mux3, I1: %mux4) : i1, i1, i1 -> i1

  // Third level - 1 MUXF9
  // CHECK: %{{.+}} = xlnx.muxf9(S : %select3, I0 : %{{.+}}, I1 : %{{.+}}) : i1, i1, i1 -> i1
  %mux7 = xlnx.muxf9(S: %select3, I0: %mux5, I1: %mux6) : i1, i1, i1 -> i1

  hw.output %mux7 : i1
}

// CHECK-LABEL: hw.module @MuxWithLuts
hw.module @MuxWithLuts(in %select: i1, in %a: i1, in %b: i1, in %c: i1, in %d: i1,
                      in %e: i1, in %f: i1, out out: i1) {
  // Combines LUT6 and MUXF7/F8/F9 to create complex logic

  // Create two LUT6s
  // CHECK: %{{.+}} = xlnx.lut6(I0 : %a, I1 : %b, I2 : %c, I3 : %d, I4 : %e, I5 : %f) {INIT = 8589934590 : ui64} : i1, i1, i1, i1, i1, i1 -> i1
  %lut1 = xlnx.lut6(I0: %a, I1: %b, I2: %c, I3: %d, I4: %e, I5: %f) {INIT = 8589934590 : ui64} : i1, i1, i1, i1, i1, i1 -> i1

  // CHECK: %{{.+}} = xlnx.lut6(I0 : %a, I1 : %b, I2 : %c, I3 : %d, I4 : %e, I5 : %f) {INIT = 4294967295 : ui64} : i1, i1, i1, i1, i1, i1 -> i1
  %lut2 = xlnx.lut6(I0: %a, I1: %b, I2: %c, I3: %d, I4: %e, I5: %f) {INIT = 4294967295 : ui64} : i1, i1, i1, i1, i1, i1 -> i1

  // Connect two LUT6s using MUXF7
  // CHECK: %{{.+}} = xlnx.muxf7(S : %select, I0 : %{{.+}}, I1 : %{{.+}}) : i1, i1, i1 -> i1
  %mux = xlnx.muxf7(S: %select, I0: %lut1, I1: %lut2) : i1, i1, i1 -> i1

  hw.output %mux : i1
}

// CHECK-LABEL: hw.module @MultiLevelMuxTree
hw.module @MultiLevelMuxTree(in %select1: i1, in %select2: i1, in %select3: i1,
                           in %a: i1, in %b: i1, in %c: i1, in %d: i1, in %e: i1, in %f: i1,
                           out out: i1) {
  // Complex example of creating a multi-level MUX tree

  // First level - using MUXF7
  // CHECK: %{{.+}} = xlnx.muxf7(S : %select1, I0 : %a, I1 : %b) : i1, i1, i1 -> i1
  %mux1 = xlnx.muxf7(S: %select1, I0: %a, I1: %b) : i1, i1, i1 -> i1

  // CHECK: %{{.+}} = xlnx.muxf7(S : %select1, I0 : %c, I1 : %d) : i1, i1, i1 -> i1
  %mux2 = xlnx.muxf7(S: %select1, I0: %c, I1: %d) : i1, i1, i1 -> i1

  // Second level - using MUXF8
  // CHECK: %{{.+}} = xlnx.muxf8(S : %select2, I0 : %{{.+}}, I1 : %{{.+}}) : i1, i1, i1 -> i1
  %mux3 = xlnx.muxf8(S: %select2, I0: %mux1, I1: %mux2) : i1, i1, i1 -> i1

  // Third level - using MUXF9
  // CHECK: %{{.+}} = xlnx.muxf9(S : %select3, I0 : %{{.+}}, I1 : %e) : i1, i1, i1 -> i1
  %mux4 = xlnx.muxf9(S: %select3, I0: %mux3, I1: %e) : i1, i1, i1 -> i1

  hw.output %mux4 : i1
}
} 