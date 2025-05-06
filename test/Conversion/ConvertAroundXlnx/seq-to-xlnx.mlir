// RUN: circt-opt %s --convert-seq-to-xlnx --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: hw.module @DoubleReg
  hw.module @DoubleReg(in %clock : !seq.clock, in %clock_en : i1, in %reset0 : i1, in %reset1 : i1, in %in : i1, out out_reg0 : i1, out out_reg1 : i1, out out_reg2 : i1) {
    // CHECK: %false = hw.constant false
    %c0_i1 = hw.constant false
    // CHECK: %true = hw.constant true
    %c1_i1 = hw.constant true
    // CHECK: %{{.+}} = xlnx.fdre(C : %clock, CE : %true, R : %reset0, D : %in) : !seq.clock, i1, i1, i1 -> i1
    %reg0 = seq.compreg %in, %clock reset %reset0, %c0_i1 : i1
    // CHECK: %{{.+}} = xlnx.fdse(C : %clock, CE : %true, S : %reset1, D : %in) : !seq.clock, i1, i1, i1 -> i1
    %reg1 = seq.compreg %in, %clock reset %reset1, %c1_i1 : i1
    // CHECK: %{{.+}} = xlnx.fdre(C : %clock, CE : %clock_en, R : %reset0, D : %in) : !seq.clock, i1, i1, i1 -> i1
    %reg2 = seq.compreg.ce %in, %clock, %clock_en reset %reset0, %c0_i1 : i1
    hw.output %reg0, %reg1, %reg2 : i1, i1, i1
  }

  // CHECK-LABEL: hw.module @WideDoubleReg
  hw.module @WideDoubleReg(in %clock : !seq.clock, in %clock_en : i1, in %reset0 : i1, in %in : i4, out out_reg0 : i4) {
    %reset_values = hw.constant 14 : i4  // 14 = 0b1110
    // CHECK: %{{.+}} = comb.extract %in from 0 : (i4) -> i1
    // CHECK-NEXT: %{{.+}} = xlnx.fdre(C : %clock, CE : %true, R : %reset0, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
    // CHECK-NEXT: comb.extract %in from 1 : (i4) -> i1
    // CHECK-NEXT: %{{.+}} = xlnx.fdse(C : %clock, CE : %true, S : %reset0, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
    // CHECK-NEXT: comb.extract %in from 2 : (i4) -> i1
    // CHECK-NEXT: %{{.+}} = xlnx.fdse(C : %clock, CE : %true, S : %reset0, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
    // CHECK-NEXT: comb.extract %in from 3 : (i4) -> i1
    // CHECK-NEXT: %{{.+}} = xlnx.fdse(C : %clock, CE : %true, S : %reset0, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
    %reg0 = seq.compreg %in, %clock reset %reset0, %reset_values : i4
    // CHECK-NEXT: %{{.+}} = comb.concat %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : i1, i1, i1, i1

    %reset_values_2 = hw.constant 11 : i4  // 11 = 0b1011
    // CHECK: %{{.+}} = comb.extract %{{.+}} from 0 : (i4) -> i1
    // CHECK-NEXT: %{{.+}} = xlnx.fdse(C : %clock, CE : %clock_en, S : %reset0, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
    // CHECK-NEXT: comb.extract %{{.+}} from 1 : (i4) -> i1
    // CHECK-NEXT: %{{.+}} = xlnx.fdse(C : %clock, CE : %clock_en, S : %reset0, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
    // CHECK-NEXT: comb.extract %{{.+}} from 2 : (i4) -> i1
    // CHECK-NEXT: %{{.+}} = xlnx.fdre(C : %clock, CE : %clock_en, R : %reset0, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
    // CHECK-NEXT: comb.extract %{{.+}} from 3 : (i4) -> i1
    // CHECK-NEXT: %{{.+}} = xlnx.fdse(C : %clock, CE : %clock_en, S : %reset0, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
    %reg1 = seq.compreg.ce %reg0, %clock, %clock_en reset %reset0, %reset_values_2 : i4
    // CHECK-NEXT: %{{.+}} = comb.concat %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : i1, i1, i1, i1
    hw.output %reg1 : i4
  }
}

