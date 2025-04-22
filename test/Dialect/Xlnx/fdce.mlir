// RUN: circt-opt %s -verify-diagnostics | FileCheck %s

// CHECK-LABEL: hw.module @FDCEBasic
hw.module @FDCEBasic(in %clock: !seq.clock, in %ce: i1, in %clr: i1, in %d: i1, 
                     out out: i1) {
  // Basic FDCE test
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %d) : !seq.clock, i1, i1, i1 -> i1
  %q = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %d) : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @FDCEWithAttributes
hw.module @FDCEWithAttributes(in %clock: !seq.clock, in %ce: i1, in %clr: i1, in %d: i1, out out: i1) {
  // FDCE test with attributes
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %d) {INIT = 1 : ui1, IS_CLR_INVERTED = 1 : ui1, IS_C_INVERTED = 1 : ui1, IS_D_INVERTED = 1 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %q = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %d) {INIT = 1 : ui1, IS_CLR_INVERTED = 1 : ui1, IS_C_INVERTED = 1 : ui1, IS_D_INVERTED = 1 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @FDCECounter
hw.module @FDCECounter(in %clock: !seq.clock, in %ce: i1, in %clr: i1, out out: i1) {
  // Corresponding Verilog code:
  // module FDCECounter(input wire clock, input wire ce, input wire clr, output wire out);
  //   reg state;
  //   wire next_state;
  //   // Invert current state as the next state
  //   assign next_state = ~state;
  //   // Flip-flop instance
  //   FDCE #(.INIT(1'b0)) counter_ff (.C(clock), .CE(ce), .CLR(clr), .D(next_state), .Q(state));
  //   assign out = state;
  // endmodule
  
  // Implementation logic
  %c1_i1 = hw.constant 1 : i1
  
  // Use FDCE to implement a simple flip-flop, output connected to input to form a counter
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %state = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %next_state) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  // Invert current state as the next state (toggle every clock cycle)
  %next_state = comb.xor %state, %c1_i1 : i1
  
  // Connect the output to the flip-flop's output
  hw.output %state : i1
}

// CHECK-LABEL: hw.module @FDCEToggle
hw.module @FDCEToggle(in %clock: !seq.clock, in %ce: i1, in %clr: i1, in %toggle: i1, 
                     out out: i1) {
  // Implement a simple toggle, state changes only when toggle signal is 1 and ce is active
  // Corresponding Verilog code:
  // module FDCEToggle(input wire clock, input wire ce, input wire clr, input wire toggle, output wire out);
  //   reg state;
  //   wire next_state;
  //   wire should_toggle;
  //   // Toggle only when both toggle and ce are active
  //   assign should_toggle = toggle & ce;
  //   // Conditionally invert state
  //   assign next_state = should_toggle ? ~state : state;
  //   // Implement flip-flop using FDCE
  //   FDCE #(.INIT(1'b0)) toggle_ff (.C(clock), .CE(ce), .CLR(clr), .D(next_state), .Q(state));
  //   assign out = state;
  // endmodule
  
  %c0_i1 = hw.constant 0 : i1
  %c1_i1 = hw.constant 1 : i1
  
  // Flip-flop with initial value 0
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %state = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %next_state) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  // Calculate the next state
  // Toggle state only when toggle signal is 1 and ce is active
  %should_toggle = comb.and %toggle, %ce : i1
  
  // Generate the inverted state
  %toggled_state = comb.xor %state, %c1_i1 : i1
  
  // Select whether the state should toggle
  %next_state = comb.mux %should_toggle, %toggled_state, %state : i1
  
  hw.output %state : i1
}

// CHECK-LABEL: hw.module @FDCEWithPriorityLogic
hw.module @FDCEWithPriorityLogic(in %clock: !seq.clock, in %ce: i1, in %clr: i1, 
                               in %d1: i1, in %d2: i1, in %sel: i1, 
                               out out: i1) {
  // Use selector logic to determine input data
  // Use 2:1 multiplexer to select input data
  %selected_data = comb.mux %sel, %d1, %d2 : i1
  
  // Use FDCE to register the selected data
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %q = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %selected_data) : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @FDCEMultiBit
hw.module @FDCEMultiBit(in %clock: !seq.clock, in %ce: i1, in %clr: i1, 
                       in %d0: i1, in %d1: i1, 
                       out out0: i1, out out1: i1) {
  // Multiple FDCE instances to build a multi-bit register
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %d0) : !seq.clock, i1, i1, i1 -> i1
  %q0 = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %d0) : !seq.clock, i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %d1) : !seq.clock, i1, i1, i1 -> i1
  %q1 = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %d1) : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q0, %q1 : i1, i1
}

// CHECK-LABEL: hw.module @FDCEWithRegion
hw.module @FDCEWithRegion(in %clock: !seq.clock, in %ce: i1, in %clr: i1, in %d: i1, 
                          out out: i1) {
  // Use FDCE operation with an initial value
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %d) : !seq.clock, i1, i1, i1 -> i1
  %q = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %d) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @ShiftRegister
hw.module @ShiftRegister(in %clock: !seq.clock, in %ce: i1, in %clr: i1, in %d: i1, 
                         out out: i1) {
  // Create a simple 4-stage shift register using FDCE
  // First stage
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %d) : !seq.clock, i1, i1, i1 -> i1
  %q1 = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %d) : !seq.clock, i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %q2 = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %q1) : !seq.clock, i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %q3 = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %q2) : !seq.clock, i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %q4 = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %q3) : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q4 : i1
}

// CHECK-LABEL: hw.module @FDCEWithCompReg
hw.module @FDCEWithCompReg(in %clock: !seq.clock, in %ce: i1, in %clr: i1, in %d: i1, 
                          out out_fdce: i1, out out_compreg: i1) {
  // For comparison, use both FDCE and seq.compreg
  %c0_i1 = hw.constant 0 : i1
  
  // Using FDCE
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %d) : !seq.clock, i1, i1, i1 -> i1
  %fdce_out = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %d) : !seq.clock, i1, i1, i1 -> i1
  
  // Using seq.compreg.ce, functionally equivalent to FDCE
  // CHECK: %{{.+}} = seq.compreg.ce %d, %clock, %ce reset %clr, %false : i1
  %compreg_out = seq.compreg.ce %d, %clock, %ce reset %clr, %c0_i1 : i1
  
  hw.output %fdce_out, %compreg_out : i1, i1
}

// CHECK-LABEL: hw.module @FDCECounter4Bit
hw.module @FDCECounter4Bit(in %clock: !seq.clock, in %ce: i1, in %clr: i1, 
                           out out: i4) {
  // 4-bit counter implementation
  // Corresponding Verilog code:
  // module FDCECounter4Bit(input wire clock, input wire ce, input wire clr, output wire [3:0] out);
  //   reg [3:0] count;
  //   wire [3:0] next_count;
  //   // Next count value (current value + 1)
  //   assign next_count = count + 1'b1;
  //   // 4 independent FDCE flip-flops, each storing one bit
  //   FDCE #(.INIT(1'b0)) count_ff0 (.C(clock), .CE(ce), .CLR(clr), .D(next_count[0]), .Q(count[0]));
  //   FDCE #(.INIT(1'b0)) count_ff1 (.C(clock), .CE(ce), .CLR(clr), .D(next_count[1]), .Q(count[1]));
  //   FDCE #(.INIT(1'b0)) count_ff2 (.C(clock), .CE(ce), .CLR(clr), .D(next_count[2]), .Q(count[2]));
  //   FDCE #(.INIT(1'b0)) count_ff3 (.C(clock), .CE(ce), .CLR(clr), .D(next_count[3]), .Q(count[3]));
  //   assign out = count;
  // endmodule
  
  // Constant for adding 1
  %c1_i4 = hw.constant 1 : i4
  
  // 4 FDCE flip-flops, each storing one bit of the counter
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %count0 = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %next_count0) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %count1 = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %next_count1) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %count2 = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %next_count2) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %count3 = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %next_count3) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  // Construct the current count value
  %count = comb.concat %count3, %count2, %count1, %count0 : i1, i1, i1, i1
  
  // Calculate the next count value
  %next_count = comb.add %count, %c1_i4 : i4
  
  // Extract individual bits for the next count
  %next_count0 = comb.extract %next_count from 0 : (i4) -> i1
  %next_count1 = comb.extract %next_count from 1 : (i4) -> i1
  %next_count2 = comb.extract %next_count from 2 : (i4) -> i1
  %next_count3 = comb.extract %next_count from 3 : (i4) -> i1
  
  hw.output %count : i4
}

// CHECK-LABEL: hw.module @FDCEWithConditionalLogic
hw.module @FDCEWithConditionalLogic(in %clock: !seq.clock, in %ce: i1, in %clr: i1, 
                                   in %d: i1, in %mode: i1, 
                                   out out: i1) {
  // Create an FDCE circuit with conditional logic
  // Corresponding Verilog code:
  // module FDCEWithConditionalLogic(input wire clock, input wire ce, input wire clr,
  //                                input wire d, input wire mode, output wire out);
  //   reg state;
  //   wire next_state;
  //   wire condition;
  //   // Select different behavior based on mode
  //   // mode=1: Use input d directly
  //   // mode=0: Use condition (mode XOR state)
  //   assign condition = mode ^ state;
  //   assign next_state = mode ? d : condition;
  //   // Use FDCE flip-flop to store state
  //   FDCE #(.INIT(1'b0)) state_ff (.C(clock), .CE(ce), .CLR(clr), .D(next_state), .Q(state));
  //   assign out = state;
  // endmodule
  
  %c0_i1 = hw.constant 0 : i1
  %c1_i1 = hw.constant 1 : i1
  
  // Current FDCE state
  %state = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %next_state) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  // Select different behavior based on mode
  %condition = comb.xor %mode, %state : i1
  %next_state = comb.mux %mode, %d, %condition : i1
  
  hw.output %state : i1
}
