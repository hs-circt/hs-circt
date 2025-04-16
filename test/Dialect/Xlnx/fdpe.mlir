// RUN: circt-opt %s -verify-diagnostics | FileCheck %s

// CHECK-LABEL: hw.module @FDPEBasic
hw.module @FDPEBasic(in %clock: !seq.clock, in %ce: i1, in %pre: i1, in %d: i1, 
                     out out: i1) {
  // Basic FDPE test
  // CHECK: %{{.+}} = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %d) : !seq.clock, i1, i1, i1 -> i1
  %q = xlnx.fdpe(C: %clock, CE: %ce, PRE: %pre, D: %d) : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @FDPEWithAttributes
hw.module @FDPEWithAttributes(in %clock: !seq.clock, in %ce: i1, in %pre: i1, in %d: i1, out out: i1) {
  // FDPE test with attributes
  // CHECK: %{{.+}} = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %d) {IS_C_INVERTED = 1 : ui1, IS_D_INVERTED = 1 : ui1, IS_PRE_INVERTED = 1 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %q = xlnx.fdpe(C: %clock, CE: %ce, PRE: %pre, D: %d) {INIT = 1 : ui1, IS_PRE_INVERTED = 1 : ui1, IS_C_INVERTED = 1 : ui1, IS_D_INVERTED = 1 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @FDPECounter
hw.module @FDPECounter(in %clock: !seq.clock, in %ce: i1, in %pre: i1, out out: i1) {
  // Corresponding Verilog code (note PRE behavior):
  // module FDPECounter(input wire clock, input wire ce, input wire pre, output wire out);
  //   reg state;
  //   wire next_state;
  //   assign next_state = ~state;
  //   FDPE #(.INIT(1'b0)) counter_ff (.C(clock), .CE(ce), .PRE(pre), .D(next_state), .Q(state));
  //   assign out = state;
  // endmodule
  
  %c1_i1 = hw.constant 1 : i1
  
  // CHECK: %{{.+}} = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %{{.+}}) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %state = xlnx.fdpe(C: %clock, CE: %ce, PRE: %pre, D: %next_state) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  %next_state = comb.xor %state, %c1_i1 : i1
  
  hw.output %state : i1
}

// CHECK-LABEL: hw.module @FDPEToggle
hw.module @FDPEToggle(in %clock: !seq.clock, in %ce: i1, in %pre: i1, in %toggle: i1, 
                     out out: i1) {
  // Corresponding Verilog code (note PRE behavior):
  // module FDPEToggle(input wire clock, input wire ce, input wire pre, input wire toggle, output wire out);
  //   reg state;
  //   wire next_state;
  //   wire should_toggle;
  //   assign should_toggle = toggle & ce;
  //   assign next_state = should_toggle ? ~state : state;
  //   FDPE #(.INIT(1'b0)) toggle_ff (.C(clock), .CE(ce), .PRE(pre), .D(next_state), .Q(state));
  //   assign out = state;
  // endmodule
  
  %c0_i1 = hw.constant 0 : i1
  %c1_i1 = hw.constant 1 : i1
  
  // CHECK: %{{.+}} = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %{{.+}}) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %state = xlnx.fdpe(C: %clock, CE: %ce, PRE: %pre, D: %next_state) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  %should_toggle = comb.and %toggle, %ce : i1
  %toggled_state = comb.xor %state, %c1_i1 : i1
  %next_state = comb.mux %should_toggle, %toggled_state, %state : i1
  
  hw.output %state : i1
}

// CHECK-LABEL: hw.module @FDPEWithPriorityLogic
hw.module @FDPEWithPriorityLogic(in %clock: !seq.clock, in %ce: i1, in %pre: i1, 
                               in %d1: i1, in %d2: i1, in %sel: i1, 
                               out out: i1) {
  %selected_data = comb.mux %sel, %d1, %d2 : i1
  
  // CHECK: %{{.+}} = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %q = xlnx.fdpe(C: %clock, CE: %ce, PRE: %pre, D: %selected_data) : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @FDPEMultiBit
hw.module @FDPEMultiBit(in %clock: !seq.clock, in %ce: i1, in %pre: i1, 
                       in %d0: i1, in %d1: i1, 
                       out out0: i1, out out1: i1) {
  // CHECK: %{{.+}} = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %d0) : !seq.clock, i1, i1, i1 -> i1
  %q0 = xlnx.fdpe(C: %clock, CE: %ce, PRE: %pre, D: %d0) : !seq.clock, i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %d1) : !seq.clock, i1, i1, i1 -> i1
  %q1 = xlnx.fdpe(C: %clock, CE: %ce, PRE: %pre, D: %d1) : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q0, %q1 : i1, i1
}

// CHECK-LABEL: hw.module @FDPEWithRegion
hw.module @FDPEWithRegion(in %clock: !seq.clock, in %ce: i1, in %pre: i1, in %d: i1, 
                          out out: i1) {
  // CHECK: %{{.+}} = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %d) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %q = xlnx.fdpe(C: %clock, CE: %ce, PRE: %pre, D: %d) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @ShiftRegisterFDPE
hw.module @ShiftRegisterFDPE(in %clock: !seq.clock, in %ce: i1, in %pre: i1, in %d: i1, 
                         out out: i1) {
  // Shift register created using FDPE (note PRE affects all stages)
  // CHECK: %{{.+}} = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %d) : !seq.clock, i1, i1, i1 -> i1
  %q1 = xlnx.fdpe(C: %clock, CE: %ce, PRE: %pre, D: %d) : !seq.clock, i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %q2 = xlnx.fdpe(C: %clock, CE: %ce, PRE: %pre, D: %q1) : !seq.clock, i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %q3 = xlnx.fdpe(C: %clock, CE: %ce, PRE: %pre, D: %q2) : !seq.clock, i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %q4 = xlnx.fdpe(C: %clock, CE: %ce, PRE: %pre, D: %q3) : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q4 : i1
}

// CHECK-LABEL: hw.module @FDPEWithCompRegLike
hw.module @FDPEWithCompRegLike(in %clock: !seq.clock, in %ce: i1, in %pre: i1, in %d: i1, 
                               out out_fdpe: i1, out out_compreg_preset: i1) {
  // Compare FDPE and seq.compreg.ce (simulate preset using reset)
  %c1_i1 = hw.constant 1 : i1
  
  // Using FDPE
  // CHECK: %{{.+}} = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %d) : !seq.clock, i1, i1, i1 -> i1
  %fdpe_out = xlnx.fdpe(C: %clock, CE: %ce, PRE: %pre, D: %d) : !seq.clock, i1, i1, i1 -> i1
  
  // Simulating asynchronous preset with seq.compreg.ce is difficult because compreg's reset is synchronous.
  // Here we connect the compreg's reset input to pre and set the reset value to 1 to simulate preset behavior,
  // but this is only a functional simulation, the timing is different.
  // Note: seq.compreg's reset is synchronous, while FDPE's PRE is asynchronous.
  // CHECK: %{{.+}} = seq.compreg.ce %d, %clock, %ce reset %pre, %true : i1
  %compreg_preset_out = seq.compreg.ce %d, %clock, %ce reset %pre, %c1_i1 : i1
  
  hw.output %fdpe_out, %compreg_preset_out : i1, i1
}

// CHECK-LABEL: hw.module @FDPECounter4Bit
hw.module @FDPECounter4Bit(in %clock: !seq.clock, in %ce: i1, in %pre: i1, 
                           out out: i4) {
  // 4-bit counter implementation (note PRE behavior)
  // Corresponding Verilog code:
  // module FDPECounter4Bit(input wire clock, input wire ce, input wire pre, output wire [3:0] out);
  //   reg [3:0] count;
  //   wire [3:0] next_count;
  //   assign next_count = count + 1'b1;
  //   FDPE #(.INIT(1'b0)) count_ff0 (.C(clock), .CE(ce), .PRE(pre), .D(next_count[0]), .Q(count[0]));
  //   FDPE #(.INIT(1'b0)) count_ff1 (.C(clock), .CE(ce), .PRE(pre), .D(next_count[1]), .Q(count[1]));
  //   FDPE #(.INIT(1'b0)) count_ff2 (.C(clock), .CE(ce), .PRE(pre), .D(next_count[2]), .Q(count[2]));
  //   FDPE #(.INIT(1'b0)) count_ff3 (.C(clock), .CE(ce), .PRE(pre), .D(next_count[3]), .Q(count[3]));
  //   assign out = count;
  // endmodule
  
  %c1_i4 = hw.constant 1 : i4
  
  // CHECK: %{{.+}} = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %{{.+}}) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %count0 = xlnx.fdpe(C: %clock, CE: %ce, PRE: %pre, D: %next_count0) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %count1 = xlnx.fdpe(C: %clock, CE: %ce, PRE: %pre, D: %next_count1) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %count2 = xlnx.fdpe(C: %clock, CE: %ce, PRE: %pre, D: %next_count2) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %count3 = xlnx.fdpe(C: %clock, CE: %ce, PRE: %pre, D: %next_count3) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  %count = comb.concat %count3, %count2, %count1, %count0 : i1, i1, i1, i1
  %next_count = comb.add %count, %c1_i4 : i4
  %next_count0 = comb.extract %next_count from 0 : (i4) -> i1
  %next_count1 = comb.extract %next_count from 1 : (i4) -> i1
  %next_count2 = comb.extract %next_count from 2 : (i4) -> i1
  %next_count3 = comb.extract %next_count from 3 : (i4) -> i1
  
  hw.output %count : i4
}

// CHECK-LABEL: hw.module @FDPEWithConditionalLogic
hw.module @FDPEWithConditionalLogic(in %clock: !seq.clock, in %ce: i1, in %pre: i1, 
                                   in %d: i1, in %mode: i1, 
                                   out out: i1) {
  // Corresponding Verilog code (note PRE behavior):
  // module FDPEWithConditionalLogic(input wire clock, input wire ce, input wire pre,
  //                                input wire d, input wire mode, output wire out);
  //   reg state;
  //   wire next_state;
  //   wire condition;
  //   assign condition = mode ^ state;
  //   assign next_state = mode ? d : condition;
  //   FDPE #(.INIT(1'b0)) state_ff (.C(clock), .CE(ce), .PRE(pre), .D(next_state), .Q(state));
  //   assign out = state;
  // endmodule
  
  %c0_i1 = hw.constant 0 : i1
  %c1_i1 = hw.constant 1 : i1
  
  // CHECK: %{{.+}} = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %{{.+}}) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %state = xlnx.fdpe(C: %clock, CE: %ce, PRE: %pre, D: %next_state) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  %condition = comb.xor %mode, %state : i1
  %next_state = comb.mux %mode, %d, %condition : i1
  
  hw.output %state : i1
} 