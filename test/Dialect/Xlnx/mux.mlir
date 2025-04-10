// RUN: circt-opt %s -verify-diagnostics | FileCheck %s

// 包含HW方言
module {
// CHECK-LABEL: hw.module @MuxExamples
hw.module @MuxExamples(in %select: i1, in %input0: i1, in %input1: i1, 
                      out out_muxf7: i1, out out_muxf8: i1, out out_muxf9: i1) {
  // 基本的MUXF7测试
  // CHECK: %{{.+}} = xlnx.muxf7(S : %select, I0 : %input0, I1 : %input1) : i1, i1, i1 -> i1
  %0 = xlnx.muxf7(S: %select, I0: %input0, I1: %input1) : i1, i1, i1 -> i1

  // 基本的MUXF8测试
  // CHECK: %{{.+}} = xlnx.muxf8(S : %select, I0 : %input0, I1 : %input1) : i1, i1, i1 -> i1
  %1 = xlnx.muxf8(S: %select, I0: %input0, I1: %input1) : i1, i1, i1 -> i1

  // 基本的MUXF9测试
  // CHECK: %{{.+}} = xlnx.muxf9(S : %select, I0 : %input0, I1 : %input1) : i1, i1, i1 -> i1
  %2 = xlnx.muxf9(S: %select, I0: %input0, I1: %input1) : i1, i1, i1 -> i1

  hw.output %0, %1, %2 : i1, i1, i1
}

// CHECK-LABEL: hw.module @CascadedMux
hw.module @CascadedMux(in %select1: i1, in %select2: i1, in %select3: i1, 
                      in %a: i1, in %b: i1, in %c: i1, in %d: i1,
                      out out: i1) {
  // 级联多个MUX操作以创建更复杂的多路复用电路
  
  // 第一级MUXF7
  // CHECK: %{{.+}} = xlnx.muxf7(S : %select1, I0 : %a, I1 : %b) : i1, i1, i1 -> i1
  %mux1 = xlnx.muxf7(S: %select1, I0: %a, I1: %b) : i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.muxf7(S : %select1, I0 : %c, I1 : %d) : i1, i1, i1 -> i1
  %mux2 = xlnx.muxf7(S: %select1, I0: %c, I1: %d) : i1, i1, i1 -> i1
  
  // 第二级MUXF8
  // CHECK: %{{.+}} = xlnx.muxf8(S : %select2, I0 : %{{.+}}, I1 : %{{.+}}) : i1, i1, i1 -> i1
  %mux3 = xlnx.muxf8(S: %select2, I0: %mux1, I1: %mux2) : i1, i1, i1 -> i1
  
  hw.output %mux3 : i1
}

// CHECK-LABEL: hw.module @ComplexMuxTree
hw.module @ComplexMuxTree(in %select1: i1, in %select2: i1, in %select3: i1,
                         in %a: i1, in %b: i1, in %c: i1, in %d: i1, 
                         in %e: i1, in %f: i1, in %g: i1, in %h: i1,
                         out out: i1) {
  // 创建一个复杂的MUX树，模拟8-1多路复用器
  
  // 第一级 - 4个MUXF7
  // CHECK: %{{.+}} = xlnx.muxf7(S : %select1, I0 : %a, I1 : %b) : i1, i1, i1 -> i1
  %mux1 = xlnx.muxf7(S: %select1, I0: %a, I1: %b) : i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.muxf7(S : %select1, I0 : %c, I1 : %d) : i1, i1, i1 -> i1
  %mux2 = xlnx.muxf7(S: %select1, I0: %c, I1: %d) : i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.muxf7(S : %select1, I0 : %e, I1 : %f) : i1, i1, i1 -> i1
  %mux3 = xlnx.muxf7(S: %select1, I0: %e, I1: %f) : i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.muxf7(S : %select1, I0 : %g, I1 : %h) : i1, i1, i1 -> i1
  %mux4 = xlnx.muxf7(S: %select1, I0: %g, I1: %h) : i1, i1, i1 -> i1
  
  // 第二级 - 2个MUXF8
  // CHECK: %{{.+}} = xlnx.muxf8(S : %select2, I0 : %{{.+}}, I1 : %{{.+}}) : i1, i1, i1 -> i1
  %mux5 = xlnx.muxf8(S: %select2, I0: %mux1, I1: %mux2) : i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.muxf8(S : %select2, I0 : %{{.+}}, I1 : %{{.+}}) : i1, i1, i1 -> i1
  %mux6 = xlnx.muxf8(S: %select2, I0: %mux3, I1: %mux4) : i1, i1, i1 -> i1
  
  // 第三级 - 1个MUXF9
  // CHECK: %{{.+}} = xlnx.muxf9(S : %select3, I0 : %{{.+}}, I1 : %{{.+}}) : i1, i1, i1 -> i1
  %mux7 = xlnx.muxf9(S: %select3, I0: %mux5, I1: %mux6) : i1, i1, i1 -> i1
  
  hw.output %mux7 : i1
}

// CHECK-LABEL: hw.module @MuxWithLuts
hw.module @MuxWithLuts(in %select: i1, in %a: i1, in %b: i1, in %c: i1, in %d: i1, 
                      in %e: i1, in %f: i1, out out: i1) {
  // 组合LUT6和MUXF7/F8/F9以创建复杂逻辑
  
  // 创建两个LUT6
  // CHECK: %{{.+}} = xlnx.lut6(I0 : %a, I1 : %b, I2 : %c, I3 : %d, I4 : %e, I5 : %f) {INIT = 8589934590 : ui64} : i1, i1, i1, i1, i1, i1 -> i1
  %lut1 = xlnx.lut6(I0: %a, I1: %b, I2: %c, I3: %d, I4: %e, I5: %f) {INIT = 8589934590 : ui64} : i1, i1, i1, i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.lut6(I0 : %a, I1 : %b, I2 : %c, I3 : %d, I4 : %e, I5 : %f) {INIT = 4294967295 : ui64} : i1, i1, i1, i1, i1, i1 -> i1
  %lut2 = xlnx.lut6(I0: %a, I1: %b, I2: %c, I3: %d, I4: %e, I5: %f) {INIT = 4294967295 : ui64} : i1, i1, i1, i1, i1, i1 -> i1
  
  // 使用MUXF7连接两个LUT6
  // CHECK: %{{.+}} = xlnx.muxf7(S : %select, I0 : %{{.+}}, I1 : %{{.+}}) : i1, i1, i1 -> i1
  %mux = xlnx.muxf7(S: %select, I0: %lut1, I1: %lut2) : i1, i1, i1 -> i1
  
  hw.output %mux : i1
}

// CHECK-LABEL: hw.module @MultiLevelMuxTree
hw.module @MultiLevelMuxTree(in %select1: i1, in %select2: i1, in %select3: i1,
                           in %a: i1, in %b: i1, in %c: i1, in %d: i1, in %e: i1, in %f: i1,
                           out out: i1) {
  // 创建多级MUX树的复杂示例
  
  // 第一级 - 使用MUXF7
  // CHECK: %{{.+}} = xlnx.muxf7(S : %select1, I0 : %a, I1 : %b) : i1, i1, i1 -> i1
  %mux1 = xlnx.muxf7(S: %select1, I0: %a, I1: %b) : i1, i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.muxf7(S : %select1, I0 : %c, I1 : %d) : i1, i1, i1 -> i1
  %mux2 = xlnx.muxf7(S: %select1, I0: %c, I1: %d) : i1, i1, i1 -> i1
  
  // 第二级 - 使用MUXF8
  // CHECK: %{{.+}} = xlnx.muxf8(S : %select2, I0 : %{{.+}}, I1 : %{{.+}}) : i1, i1, i1 -> i1
  %mux3 = xlnx.muxf8(S: %select2, I0: %mux1, I1: %mux2) : i1, i1, i1 -> i1
  
  // 第三级 - 使用MUXF9
  // CHECK: %{{.+}} = xlnx.muxf9(S : %select3, I0 : %{{.+}}, I1 : %e) : i1, i1, i1 -> i1
  %mux4 = xlnx.muxf9(S: %select3, I0: %mux3, I1: %e) : i1, i1, i1 -> i1
  
  hw.output %mux4 : i1
}
} 