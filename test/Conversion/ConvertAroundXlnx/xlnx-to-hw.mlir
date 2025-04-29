// RUN: circt-opt %s --convert-core-to-xlnx --convert-xlnx-to-hw --split-input-file --verify-diagnostics | firtool -format=mlir - | FileCheck %s
// REQUIRES: iverilog
// RUN: bash %S/difftest.sh -f %s -m RegAndLut

module {
  // CHECK-LABEL: module RegAndLut
  hw.module @RegAndLut(in %clock : !seq.clock, in %clock_en : i1, in %reset0 : i1, in %reset1 : i1, in %in : i1, out out0 : i1, out out1 : i1, out out2 : i1) {
    %c0_i1 = hw.constant false
    %c1_i1 = hw.constant true
    // CHECK: FDRE _GEN{{.*}}
    // CHECK: .C  (clock)
    // CHECK: .CE (1'h1)
    // CHECK: .R  (reset0)
    // CHECK: .D  (in)
    // CHECK: .Q  (out0)
    %reg0 = seq.compreg %in, %clock reset %reset0, %c0_i1 : i1
    // CHECK: FDSE _GEN{{.*}}
    // CHECK: .C  (clock)
    // CHECK: .CE (1'h1)
    // CHECK: .S  (reset1)
    // CHECK: .D  (in)
    // CHECK: .Q  (__Q_0)
    %reg1 = seq.compreg %in, %clock reset %reset1, %c1_i1 : i1
    // CHECK: FDRE _GEN{{.*}}
    // CHECK: .C  (clock)
    // CHECK: .CE (clock_en)
    // CHECK: .R  (reset0)
    // CHECK: .D  (in)
    // CHECK: .Q  (__Q)
    %reg2 = seq.compreg.ce %in, %clock, %clock_en reset %reset0, %c0_i1 : i1
    // CHECK: LUT2 #(
    // CHECK: .INIT(14)
    // CHECK: ) _GEN{{.*}}
    // CHECK: .I0 (__Q_0)
    // CHECK: .I1 (__Q)
    // CHECK: .O  (out2)
    %comb_out = xlnx.lut2(I0: %reg1, I1: %reg2) {INIT = 14 : ui4} : i1, i1 -> i1
    hw.output %reg0, %reg1, %comb_out : i1, i1, i1
  }
  // CHECK-LABEL: endmodule
}
