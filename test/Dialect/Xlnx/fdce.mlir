// RUN: circt-opt %s -verify-diagnostics | FileCheck %s

// CHECK-LABEL: hw.module @FDCEBasic
hw.module @FDCEBasic(in %clock: !seq.clock, in %ce: i1, in %clr: i1, in %d: i1, 
                     out out: i1) {
  // 基本FDCE测试
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %d) : !seq.clock, i1, i1, i1 -> i1
  %q = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %d) : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @FDCEWithAttributes
hw.module @FDCEWithAttributes(in %clock: !seq.clock, in %ce: i1, in %clr: i1, in %d: i1, out out: i1) {
  // 带有属性的FDCE测试
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %d) {INIT = 1 : ui1, IS_CLR_INVERTED = 1 : ui1, IS_C_INVERTED = 1 : ui1, IS_D_INVERTED = 1 : ui1} : !seq.clock, i1, i1, i1 -> i1
  %q = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %d) {INIT = 1 : ui1, IS_CLR_INVERTED = 1 : ui1, IS_C_INVERTED = 1 : ui1, IS_D_INVERTED = 1 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @FDCECounter
hw.module @FDCECounter(in %clock: !seq.clock, in %ce: i1, in %clr: i1, out out: i1) {
  // 对应的Verilog代码：
  // module FDCECounter(input wire clock, input wire ce, input wire clr, output wire out);
  //   reg state;
  //   wire next_state;
  //   // 反转当前状态作为下一个状态
  //   assign next_state = ~state;
  //   // 触发器实例
  //   FDCE #(.INIT(1'b0)) counter_ff (.C(clock), .CE(ce), .CLR(clr), .D(next_state), .Q(state));
  //   assign out = state;
  // endmodule
  
  // 实现逻辑
  %c1_i1 = hw.constant 1 : i1
  
  // 使用FDCE实现一个简单的触发器，输出连接到输入形成一个计数器
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %state = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %next_state) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  // 反转当前状态作为下一个状态（每个时钟周期切换）
  %next_state = comb.xor %state, %c1_i1 : i1
  
  // 将输出连接到触发器的输出
  hw.output %state : i1
}

// CHECK-LABEL: hw.module @FDCEToggle
hw.module @FDCEToggle(in %clock: !seq.clock, in %ce: i1, in %clr: i1, in %toggle: i1, 
                     out out: i1) {
  // 实现一个简单的开关，只有当toggle信号为1且ce有效时才会改变状态
  // 对应的Verilog代码：
  // module FDCEToggle(input wire clock, input wire ce, input wire clr, input wire toggle, output wire out);
  //   reg state;
  //   wire next_state;
  //   wire should_toggle;
  //   // 只有当toggle和ce都有效时才切换
  //   assign should_toggle = toggle & ce;
  //   // 条件反转状态
  //   assign next_state = should_toggle ? ~state : state;
  //   // 使用FDCE实现触发器
  //   FDCE #(.INIT(1'b0)) toggle_ff (.C(clock), .CE(ce), .CLR(clr), .D(next_state), .Q(state));
  //   assign out = state;
  // endmodule
  
  %c0_i1 = hw.constant 0 : i1
  %c1_i1 = hw.constant 1 : i1
  
  // 初始值为0的触发器
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %state = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %next_state) {INIT = 0 : ui1} : !seq.clock, i1, i1, i1 -> i1
  
  // 计算下一个状态
  // 只有当toggle信号为1且ce有效时才会切换状态
  %should_toggle = comb.and %toggle, %ce : i1
  
  // 生成反转后的状态
  %toggled_state = comb.xor %state, %c1_i1 : i1
  
  // 选择是否应该切换状态
  %next_state = comb.mux %should_toggle, %toggled_state, %state : i1
  
  hw.output %state : i1
}

// CHECK-LABEL: hw.module @FDCEWithPriorityLogic
hw.module @FDCEWithPriorityLogic(in %clock: !seq.clock, in %ce: i1, in %clr: i1, 
                               in %d1: i1, in %d2: i1, in %sel: i1, 
                               out out: i1) {
  // 使用选择器逻辑决定输入数据
  // 使用2:1多路复用器选择输入数据
  %selected_data = comb.mux %sel, %d1, %d2 : i1
  
  // 使用FDCE寄存选择的数据
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %{{.+}}) : !seq.clock, i1, i1, i1 -> i1
  %q = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %selected_data) : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @FDCEMultiBit
hw.module @FDCEMultiBit(in %clock: !seq.clock, in %ce: i1, in %clr: i1, 
                       in %d0: i1, in %d1: i1, 
                       out out0: i1, out out1: i1) {
  // 多个FDCE实例用于构建多比特寄存器
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %d0) : !seq.clock, i1, i1, i1 -> i1
  %q0 = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %d0) : !seq.clock, i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %d1) : !seq.clock, i1, i1, i1 -> i1
  %q1 = xlnx.fdce(C: %clock, CE: %ce, CLR: %clr, D: %d1) : !seq.clock, i1, i1, i1 -> i1
  
  hw.output %q0, %q1 : i1, i1
}
