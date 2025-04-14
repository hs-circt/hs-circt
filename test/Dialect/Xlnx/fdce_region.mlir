// RUN: circt-opt %s -verify-diagnostics | FileCheck %s

// CHECK-LABEL: module {
module {
// CHECK-LABEL: hw.module @FDCEWithRegion
hw.module @FDCEWithRegion(in %clock: !seq.clock, in %ce: i1, in %clr: i1, in %d: i1, 
                          out out: i1) {
  // 使用FDCE操作并带有初始值
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %d) : !seq.clock, i1, i1, i1 -> i1
  %q = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %d) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @ShiftRegister
hw.module @ShiftRegister(in %clock: !seq.clock, in %ce: i1, in %clr: i1, in %d: i1, 
                         out out: i1) {
  // 使用FDCE创建一个简单的4级移位寄存器
  // 第一级
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %d) : !seq.clock, i1, i1, i1 -> i1
  %q1 = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %d) : !seq.clock, i1, i1, i1 -> i1
  
  // 第二级
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %q2 = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %q1) : !seq.clock, i1, i1, i1 -> i1
  
  // 第三级
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %q3 = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %q2) : !seq.clock, i1, i1, i1 -> i1
  
  // 第四级
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %q4 = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %q3) : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q4 : i1
}

// CHECK-LABEL: hw.module @FDCEWithCompReg
hw.module @FDCEWithCompReg(in %clock: !seq.clock, in %ce: i1, in %clr: i1, in %d: i1, 
                          out out_fdce: i1, out out_compreg: i1) {
  // 为了对比，同时使用FDCE和seq.compreg
  %c0_i1 = hw.constant 0 : i1
  
  // 使用FDCE
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %d) : !seq.clock, i1, i1, i1 -> i1
  %fdce_out = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %d) : !seq.clock, i1, i1, i1 -> i1
  
  // 使用seq.compreg.ce，功能上等同于FDCE
  // CHECK: %{{.+}} = seq.compreg.ce %d, %clock, %ce reset %clr, %false : i1
  %compreg_out = seq.compreg.ce %d, %clock, %ce reset %clr, %c0_i1 : i1
  
  hw.output %fdce_out, %compreg_out : i1, i1
}

// CHECK-LABEL: hw.module @FDCECounter4Bit
hw.module @FDCECounter4Bit(in %clock: !seq.clock, in %ce: i1, in %clr: i1, 
                           out out: i4) {
  // 4位计数器实现
  // 对应的Verilog代码：
  // module FDCECounter4Bit(input wire clock, input wire ce, input wire clr, output wire [3:0] out);
  //   reg [3:0] count;
  //   wire [3:0] next_count;
  //   // 下一个计数值（当前值+1）
  //   assign next_count = count + 1'b1;
  //   // 4个独立的FDCE触发器，每个存储一位
  //   FDCE #(.INIT(1'b0)) count_ff0 (.C(clock), .CE(ce), .CLR(clr), .D(next_count[0]), .Q(count[0]));
  //   FDCE #(.INIT(1'b0)) count_ff1 (.C(clock), .CE(ce), .CLR(clr), .D(next_count[1]), .Q(count[1]));
  //   FDCE #(.INIT(1'b0)) count_ff2 (.C(clock), .CE(ce), .CLR(clr), .D(next_count[2]), .Q(count[2]));
  //   FDCE #(.INIT(1'b0)) count_ff3 (.C(clock), .CE(ce), .CLR(clr), .D(next_count[3]), .Q(count[3]));
  //   assign out = count;
  // endmodule
  
  // 加1的常量
  %c1_i4 = hw.constant 1 : i4
  
  // 4个FDCE触发器，每个存储计数器的一位
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %count0 = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %next_count0) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %count1 = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %next_count1) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %count2 = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %next_count2) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %count3 = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %next_count3) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  // 构建当前计数值
  %count = comb.concat %count3, %count2, %count1, %count0 : i1, i1, i1, i1
  
  // 计算下一个计数值
  %next_count = comb.add %count, %c1_i4 : i4
  
  // 分解下一个计数值到各个位
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
  // 创建一个有条件逻辑的FDCE电路
  // 对应的Verilog代码：
  // module FDCEWithConditionalLogic(input wire clock, input wire ce, input wire clr, 
  //                                input wire d, input wire mode, output wire out);
  //   reg state;
  //   wire next_state;
  //   wire condition;
  //   // 根据模式选择不同行为
  //   // mode=1: 直接使用输入d
  //   // mode=0: 使用condition (mode XOR state)
  //   assign condition = mode ^ state;
  //   assign next_state = mode ? d : condition;
  //   // 使用FDCE触发器存储状态
  //   FDCE #(.INIT(1'b0)) state_ff (.C(clock), .CE(ce), .CLR(clr), .D(next_state), .Q(state));
  //   assign out = state;
  // endmodule
  
  %c0_i1 = hw.constant 0 : i1
  %c1_i1 = hw.constant 1 : i1
  
  // 当前FDCE状态
  %state = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %next_state) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  // 根据模式选择不同行为
  %condition = comb.xor %mode, %state : i1
  %next_state = comb.mux %mode, %d, %condition : i1
  
  hw.output %state : i1
}
} 