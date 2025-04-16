// RUN: circt-opt %s -verify-diagnostics | FileCheck %s

// CHECK-LABEL: hw.module @FDREBasic
hw.module @FDREBasic(in %clock: !seq.clock, in %ce: i1, in %r: i1, in %d: i1, 
                     out out: i1) {
  // Basic FDRE test (synchronous Reset)
  // CHECK: %{{.+}} = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %d) : !seq.clock, i1, i1, i1 -> i1
  %q = xlnx.fdre(C: %clock, CE: %ce, R: %r, D: %d) : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @FDREWithAttributes
hw.module @FDREWithAttributes(in %clock: !seq.clock, in %ce: i1, in %r: i1, in %d: i1, out out: i1) {
  // FDRE test with attributes
  // CHECK: %{{.+}} = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %d) {INIT = 1 : ui1, IS_C_INVERTED = 1 : ui1, IS_D_INVERTED = 1 : ui1, IS_R_INVERTED = 1 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %q = xlnx.fdre(C: %clock, CE: %ce, R: %r, D: %d) {INIT = 1 : ui1, IS_R_INVERTED = 1 : ui1, IS_C_INVERTED = 1 : ui1, IS_D_INVERTED = 1 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @FDRECounter
hw.module @FDRECounter(in %clock: !seq.clock, in %ce: i1, in %r: i1, out out: i1) {
  // Corresponding Verilog code:
  // module FDRECounter(input wire clock, input wire ce, input wire r, output wire out);
  //   reg state;
  //   wire next_state;
  //   assign next_state = ~state;
  //   FDRE #(.INIT(1'b0)) counter_ff (.C(clock), .CE(ce), .R(r), .D(next_state), .Q(state));
  //   assign out = state;
  // endmodule
  
  %c1_i1 = hw.constant 1 : i1
  
  // CHECK: %{{.+}} = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %state = xlnx.fdre(C: %clock, CE: %ce, R: %r, D: %next_state) : !seq.clock, i1, i1, i1 -> i1
  
  %next_state = comb.xor %state, %c1_i1 : i1
  
  hw.output %state : i1
}

// CHECK-LABEL: hw.module @FDREToggle
hw.module @FDREToggle(in %clock: !seq.clock, in %ce: i1, in %r: i1, in %toggle: i1, 
                     out out: i1) {
  // Corresponding Verilog code:
  // module FDREToggle(input wire clock, input wire ce, input wire r, input wire toggle, output wire out);
  //   reg state;
  //   wire next_state;
  //   wire should_toggle;
  //   assign should_toggle = toggle & ce;
  //   assign next_state = should_toggle ? ~state : state;
  //   FDRE #(.INIT(1'b0)) toggle_ff (.C(clock), .CE(ce), .R(r), .D(next_state), .Q(state));
  //   assign out = state;
  // endmodule
  
  %c0_i1 = hw.constant 0 : i1
  %c1_i1 = hw.constant 1 : i1
  
  // CHECK: %{{.+}} = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %state = xlnx.fdre(C: %clock, CE: %ce, R: %r, D: %next_state) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  %should_toggle = comb.and %toggle, %ce : i1
  %toggled_state = comb.xor %state, %c1_i1 : i1
  %next_state = comb.mux %should_toggle, %toggled_state, %state : i1
  
  hw.output %state : i1
}

// CHECK-LABEL: hw.module @FDREWithPriorityLogic
hw.module @FDREWithPriorityLogic(in %clock: !seq.clock, in %ce: i1, in %r: i1, 
                               in %d1: i1, in %d2: i1, in %sel: i1, 
                               out out: i1) {
  %selected_data = comb.mux %sel, %d1, %d2 : i1
  
  // CHECK: %{{.+}} = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %q = xlnx.fdre(C: %clock, CE: %ce, R: %r, D: %selected_data) : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @FDREMultiBit
hw.module @FDREMultiBit(in %clock: !seq.clock, in %ce: i1, in %r: i1, 
                       in %d0: i1, in %d1: i1, 
                       out out0: i1, out out1: i1) {
  // CHECK: %{{.+}} = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %d0) : !seq.clock, i1, i1, i1 -> i1
  %q0 = xlnx.fdre(C: %clock, CE: %ce, R: %r, D: %d0) : !seq.clock, i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %d1) : !seq.clock, i1, i1, i1 -> i1
  %q1 = xlnx.fdre(C: %clock, CE: %ce, R: %r, D: %d1) : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q0, %q1 : i1, i1
}

// CHECK-LABEL: hw.module @FDREWithRegion
hw.module @FDREWithRegion(in %clock: !seq.clock, in %ce: i1, in %r: i1, in %d: i1, 
                          out out: i1) {
  // CHECK: %{{.+}} = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %d) : !seq.clock, i1, i1, i1 -> i1
  %q = xlnx.fdre(C: %clock, CE: %ce, R: %r, D: %d) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @ShiftRegisterFDRE
hw.module @ShiftRegisterFDRE(in %clock: !seq.clock, in %ce: i1, in %r: i1, in %d: i1, 
                         out out: i1) {
  // Shift register created using FDRE
  // CHECK: %{{.+}} = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %d) : !seq.clock, i1, i1, i1 -> i1
  %q1 = xlnx.fdre(C: %clock, CE: %ce, R: %r, D: %d) : !seq.clock, i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %q2 = xlnx.fdre(C: %clock, CE: %ce, R: %r, D: %q1) : !seq.clock, i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %q3 = xlnx.fdre(C: %clock, CE: %ce, R: %r, D: %q2) : !seq.clock, i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %q4 = xlnx.fdre(C: %clock, CE: %ce, R: %r, D: %q3) : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q4 : i1
}

// CHECK-LABEL: hw.module @FDREWithCompReg
hw.module @FDREWithCompReg(in %clock: !seq.clock, in %ce: i1, in %r: i1, in %d: i1, 
                          out out_fdre: i1, out out_compreg: i1) {
  // Compare FDRE and seq.compreg.ce
  %c0_i1 = hw.constant 0 : i1
  
  // Using FDRE
  // CHECK: %{{.+}} = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %d) : !seq.clock, i1, i1, i1 -> i1
  %fdre_out = xlnx.fdre(C: %clock, CE: %ce, R: %r, D: %d) : !seq.clock, i1, i1, i1 -> i1
  
  // Using seq.compreg.ce, functionally equivalent to FDRE (assuming R is active-high synchronous reset)
  // CHECK: %{{.+}} = seq.compreg.ce %d, %clock, %ce reset %r, %false : i1
  %compreg_out = seq.compreg.ce %d, %clock, %ce reset %r, %c0_i1 : i1
  
  hw.output %fdre_out, %compreg_out : i1, i1
}

// CHECK-LABEL: hw.module @FDRECounter4Bit
hw.module @FDRECounter4Bit(in %clock: !seq.clock, in %ce: i1, in %r: i1, 
                           out out: i4) {
  // 4-bit counter implementation
  // Corresponding Verilog code:
  // module FDRECounter4Bit(input wire clock, input wire ce, input wire r, output wire [3:0] out);
  //   reg [3:0] count;
  //   wire [3:0] next_count;
  //   assign next_count = count + 1'b1;
  //   FDRE #(.INIT(1'b0)) count_ff0 (.C(clock), .CE(ce), .R(r), .D(next_count[0]), .Q(count[0]));
  //   FDRE #(.INIT(1'b0)) count_ff1 (.C(clock), .CE(ce), .R(r), .D(next_count[1]), .Q(count[1]));
  //   FDRE #(.INIT(1'b0)) count_ff2 (.C(clock), .CE(ce), .R(r), .D(next_count[2]), .Q(count[2]));
  //   FDRE #(.INIT(1'b0)) count_ff3 (.C(clock), .CE(ce), .R(r), .D(next_count[3]), .Q(count[3]));
  //   assign out = count;
  // endmodule
  
  %c1_i4 = hw.constant 1 : i4
  
  // CHECK: %{{.+}} = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %count0 = xlnx.fdre(C: %clock, CE: %ce, R: %r, D: %next_count0) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %count1 = xlnx.fdre(C: %clock, CE: %ce, R: %r, D: %next_count1) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %count2 = xlnx.fdre(C: %clock, CE: %ce, R: %r, D: %next_count2) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %count3 = xlnx.fdre(C: %clock, CE: %ce, R: %r, D: %next_count3) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  %count = comb.concat %count3, %count2, %count1, %count0 : i1, i1, i1, i1
  %next_count = comb.add %count, %c1_i4 : i4
  %next_count0 = comb.extract %next_count from 0 : (i4) -> i1
  %next_count1 = comb.extract %next_count from 1 : (i4) -> i1
  %next_count2 = comb.extract %next_count from 2 : (i4) -> i1
  %next_count3 = comb.extract %next_count from 3 : (i4) -> i1
  
  hw.output %count : i4
}

// CHECK-LABEL: hw.module @FDREWithConditionalLogic
hw.module @FDREWithConditionalLogic(in %clock: !seq.clock, in %ce: i1, in %r: i1, 
                                   in %d: i1, in %mode: i1, 
                                   out out: i1) {
  // Corresponding Verilog code:
  // module FDREWithConditionalLogic(input wire clock, input wire ce, input wire r,
  //                                input wire d, input wire mode, output wire out);
  //   reg state;
  //   wire next_state;
  //   wire condition;
  //   assign condition = mode ^ state;
  //   assign next_state = mode ? d : condition;
  //   FDRE #(.INIT(1'b0)) state_ff (.C(clock), .CE(ce), .R(r), .D(next_state), .Q(state));
  //   assign out = state;
  // endmodule
  
  %c0_i1 = hw.constant 0 : i1
  %c1_i1 = hw.constant 1 : i1
  
  // CHECK: %{{.+}} = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %state = xlnx.fdre(C: %clock, CE: %ce, R: %r, D: %next_state) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  %condition = comb.xor %mode, %state : i1
  %next_state = comb.mux %mode, %d, %condition : i1
  
  hw.output %state : i1
} 