// RUN: circt-opt %s -verify-diagnostics | FileCheck %s

// CHECK-LABEL: hw.module @FDSEBasic
hw.module @FDSEBasic(in %clock: !seq.clock, in %ce: i1, in %s: i1, in %d: i1, 
                     out out: i1) {
  // Basic FDSE test (synchronous Set)
  // CHECK: %{{.+}} = xlnx.fdse(C : %clock, CE : %ce, S : %s, D : %d) : !seq.clock, i1, i1, i1 -> i1
  %q = xlnx.fdse(C: %clock, CE: %ce, S: %s, D: %d) : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @FDSEWithAttributes
hw.module @FDSEWithAttributes(in %clock: !seq.clock, in %ce: i1, in %s: i1, in %d: i1, out out: i1) {
  // FDSE test with attributes
  // CHECK: %{{.+}} = xlnx.fdse(C : %clock, CE : %ce, S : %s, D : %d) {IS_C_INVERTED = 1 : ui1, IS_D_INVERTED = 1 : ui1, IS_S_INVERTED = 1 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %q = xlnx.fdse(C: %clock, CE: %ce, S: %s, D: %d) {INIT = 1 : ui1, IS_C_INVERTED = 1 : ui1, IS_D_INVERTED = 1 : ui1, IS_S_INVERTED = 1 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @FDSECounter
hw.module @FDSECounter(in %clock: !seq.clock, in %ce: i1, in %s: i1, out out: i1) {
  // Corresponding Verilog code (note S behavior):
  // module FDSECounter(input wire clock, input wire ce, input wire s, output wire out);
  //   reg state;
  //   wire next_state;
  //   assign next_state = ~state;
  //   FDSE #(.INIT(1'b0)) counter_ff (.C(clock), .CE(ce), .S(s), .D(next_state), .Q(state));
  //   assign out = state;
  // endmodule
  
  %c1_i1 = hw.constant 1 : i1
  
  // CHECK: %{{.+}} = xlnx.fdse(C : %clock, CE : %ce, S : %s, D : %{{.+}}) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %state = xlnx.fdse(C: %clock, CE: %ce, S: %s, D: %next_state) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  %next_state = comb.xor %state, %c1_i1 : i1
  
  hw.output %state : i1
}

// CHECK-LABEL: hw.module @FDSEToggle
hw.module @FDSEToggle(in %clock: !seq.clock, in %ce: i1, in %s: i1, in %toggle: i1, 
                     out out: i1) {
  // Corresponding Verilog code (note S behavior):
  // module FDSEToggle(input wire clock, input wire ce, input wire s, input wire toggle, output wire out);
  //   reg state;
  //   wire next_state;
  //   wire should_toggle;
  //   assign should_toggle = toggle & ce;
  //   assign next_state = should_toggle ? ~state : state;
  //   FDSE #(.INIT(1'b0)) toggle_ff (.C(clock), .CE(ce), .S(s), .D(next_state), .Q(state));
  //   assign out = state;
  // endmodule
  
  %c0_i1 = hw.constant 0 : i1
  %c1_i1 = hw.constant 1 : i1
  
  // CHECK: %{{.+}} = xlnx.fdse(C : %clock, CE : %ce, S : %s, D : %{{.+}}) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %state = xlnx.fdse(C: %clock, CE: %ce, S: %s, D: %next_state) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  %should_toggle = comb.and %toggle, %ce : i1
  %toggled_state = comb.xor %state, %c1_i1 : i1
  %next_state = comb.mux %should_toggle, %toggled_state, %state : i1
  
  hw.output %state : i1
}

// CHECK-LABEL: hw.module @FDSEWithPriorityLogic
hw.module @FDSEWithPriorityLogic(in %clock: !seq.clock, in %ce: i1, in %s: i1, 
                               in %d1: i1, in %d2: i1, in %sel: i1, 
                               out out: i1) {
  %selected_data = comb.mux %sel, %d1, %d2 : i1
  
  // CHECK: %{{.+}} = xlnx.fdse(C : %clock, CE : %ce, S : %s, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %q = xlnx.fdse(C: %clock, CE: %ce, S: %s, D: %selected_data) : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @FDSEMultiBit
hw.module @FDSEMultiBit(in %clock: !seq.clock, in %ce: i1, in %s: i1, 
                       in %d0: i1, in %d1: i1, 
                       out out0: i1, out out1: i1) {
  // CHECK: %{{.+}} = xlnx.fdse(C : %clock, CE : %ce, S : %s, D : %d0) : !seq.clock, i1, i1, i1 -> i1
  %q0 = xlnx.fdse(C: %clock, CE: %ce, S: %s, D: %d0) : !seq.clock, i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.fdse(C : %clock, CE : %ce, S : %s, D : %d1) : !seq.clock, i1, i1, i1 -> i1
  %q1 = xlnx.fdse(C: %clock, CE: %ce, S: %s, D: %d1) : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q0, %q1 : i1, i1
}

// CHECK-LABEL: hw.module @FDSEWithRegion
hw.module @FDSEWithRegion(in %clock: !seq.clock, in %ce: i1, in %s: i1, in %d: i1, 
                          out out: i1) {
  // CHECK: %{{.+}} = xlnx.fdse(C : %clock, CE : %ce, S : %s, D : %d) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %q = xlnx.fdse(C: %clock, CE: %ce, S: %s, D: %d) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @ShiftRegisterFDSE
hw.module @ShiftRegisterFDSE(in %clock: !seq.clock, in %ce: i1, in %s: i1, in %d: i1, 
                         out out: i1) {
  // Shift register created using FDSE
  // CHECK: %{{.+}} = xlnx.fdse(C : %clock, CE : %ce, S : %s, D : %d) : !seq.clock, i1, i1, i1 -> i1
  %q1 = xlnx.fdse(C: %clock, CE: %ce, S: %s, D: %d) : !seq.clock, i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.fdse(C : %clock, CE : %ce, S : %s, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %q2 = xlnx.fdse(C: %clock, CE: %ce, S: %s, D: %q1) : !seq.clock, i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.fdse(C : %clock, CE : %ce, S : %s, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %q3 = xlnx.fdse(C: %clock, CE: %ce, S: %s, D: %q2) : !seq.clock, i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.fdse(C : %clock, CE : %ce, S : %s, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %q4 = xlnx.fdse(C: %clock, CE: %ce, S: %s, D: %q3) : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q4 : i1
}

// CHECK-LABEL: hw.module @FDSEWithCompReg
hw.module @FDSEWithCompReg(in %clock: !seq.clock, in %ce: i1, in %s: i1, in %d: i1, 
                          out out_fdse: i1, out out_compreg_set: i1) {
  // Compare FDSE and seq.compreg.ce (simulate set using reset)
  %c1_i1 = hw.constant 1 : i1
  
  // Using FDSE
  // CHECK: %{{.+}} = xlnx.fdse(C : %clock, CE : %ce, S : %s, D : %d) : !seq.clock, i1, i1, i1 -> i1
  %fdse_out = xlnx.fdse(C: %clock, CE: %ce, S: %s, D: %d) : !seq.clock, i1, i1, i1 -> i1
  
  // Using seq.compreg.ce, connect S to reset, set resetValue to 1 to simulate synchronous Set
  // CHECK: %{{.+}} = seq.compreg.ce %d, %clock, %ce reset %s, %true : i1
  %compreg_set_out = seq.compreg.ce %d, %clock, %ce reset %s, %c1_i1 : i1
  
  hw.output %fdse_out, %compreg_set_out : i1, i1
}

// CHECK-LABEL: hw.module @FDSECounter4Bit
hw.module @FDSECounter4Bit(in %clock: !seq.clock, in %ce: i1, in %s: i1, 
                           out out: i4) {
  // 4-bit counter implementation (note S behavior)
  // Corresponding Verilog code:
  // module FDSECounter4Bit(input wire clock, input wire ce, input wire s, output wire [3:0] out);
  //   reg [3:0] count;
  //   wire [3:0] next_count;
  //   assign next_count = count + 1'b1;
  //   FDSE #(.INIT(1'b0)) count_ff0 (.C(clock), .CE(ce), .S(s), .D(next_count[0]), .Q(count[0]));
  //   FDSE #(.INIT(1'b0)) count_ff1 (.C(clock), .CE(ce), .S(s), .D(next_count[1]), .Q(count[1]));
  //   FDSE #(.INIT(1'b0)) count_ff2 (.C(clock), .CE(ce), .S(s), .D(next_count[2]), .Q(count[2]));
  //   FDSE #(.INIT(1'b0)) count_ff3 (.C(clock), .CE(ce), .S(s), .D(next_count[3]), .Q(count[3]));
  //   assign out = count;
  // endmodule
  
  %c1_i4 = hw.constant 1 : i4
  
  // CHECK: %{{.+}} = xlnx.fdse(C : %clock, CE : %ce, S : %s, D : %{{.+}}) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %count0 = xlnx.fdse(C: %clock, CE: %ce, S: %s, D: %next_count0) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %count1 = xlnx.fdse(C: %clock, CE: %ce, S: %s, D: %next_count1) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %count2 = xlnx.fdse(C: %clock, CE: %ce, S: %s, D: %next_count2) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %count3 = xlnx.fdse(C: %clock, CE: %ce, S: %s, D: %next_count3) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  %count = comb.concat %count3, %count2, %count1, %count0 : i1, i1, i1, i1
  %next_count = comb.add %count, %c1_i4 : i4
  %next_count0 = comb.extract %next_count from 0 : (i4) -> i1
  %next_count1 = comb.extract %next_count from 1 : (i4) -> i1
  %next_count2 = comb.extract %next_count from 2 : (i4) -> i1
  %next_count3 = comb.extract %next_count from 3 : (i4) -> i1
  
  hw.output %count : i4
}

// CHECK-LABEL: hw.module @FDSEWithConditionalLogic
hw.module @FDSEWithConditionalLogic(in %clock: !seq.clock, in %ce: i1, in %s: i1, 
                                   in %d: i1, in %mode: i1, 
                                   out out: i1) {
  // Corresponding Verilog code (note S behavior):
  // module FDSEWithConditionalLogic(input wire clock, input wire ce, input wire s,
  //                                input wire d, input wire mode, output wire out);
  //   reg state;
  //   wire next_state;
  //   wire condition;
  //   assign condition = mode ^ state;
  //   assign next_state = mode ? d : condition;
  //   FDSE #(.INIT(1'b0)) state_ff (.C(clock), .CE(ce), .S(s), .D(next_state), .Q(state));
  //   assign out = state;
  // endmodule
  
  %c0_i1 = hw.constant 0 : i1
  %c1_i1 = hw.constant 1 : i1
  
  // CHECK: %{{.+}} = xlnx.fdse(C : %clock, CE : %ce, S : %s, D : %{{.+}}) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %state = xlnx.fdse(C: %clock, CE: %ce, S: %s, D: %next_state) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  %condition = comb.xor %mode, %state : i1
  %next_state = comb.mux %mode, %d, %condition : i1
  
  hw.output %state : i1
} 