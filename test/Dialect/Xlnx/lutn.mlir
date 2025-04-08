// RUN: circt-opt %s -verify-diagnostics | FileCheck %s

// CHECK-LABEL: hw.module @LutExamples
hw.module @LutExamples(in %a: i1, in %b: i1, in %c: i1, out out1: i1, out out2: i1, out out3: i1, out out4: i1) {
  // 1输入LUT (缓冲器，INIT=0x2，二进制10)
  // CHECK: %{{.+}} = xlnx.lutn(%a) {INIT = 2 : ui64} : (i1) -> i1
  %0 = xlnx.lutn(%a) {INIT = 2 : ui64} : (i1) -> i1

  // 2输入LUT (AND门，INIT=0x8，二进制1000)
  // CHECK: %{{.+}} = xlnx.lutn(%a, %b) {INIT = 8 : ui64} : (i1, i1) -> i1
  %1 = xlnx.lutn(%a, %b) {INIT = 8 : ui64} : (i1, i1) -> i1

  // 3输入LUT (OR门，INIT=0xFE，二进制11111110)
  // CHECK: %{{.+}} = xlnx.lutn(%a, %b, %c) {INIT = 254 : ui64} : (i1, i1, i1) -> i1
  %2 = xlnx.lutn(%a, %b, %c) {INIT = 254 : ui64} : (i1, i1, i1) -> i1

  // 使用相同的输入创建不同功能的LUT (XOR门，INIT=0x6，二进制0110)
  // CHECK: %{{.+}} = xlnx.lutn(%a, %b) {INIT = 6 : ui64} : (i1, i1) -> i1
  %3 = xlnx.lutn(%a, %b) {INIT = 6 : ui64} : (i1, i1) -> i1

  hw.output %0, %1, %2, %3 : i1, i1, i1, i1
}

// CHECK-LABEL: hw.module @CascadedLuts
hw.module @CascadedLuts(in %a: i1, in %b: i1, in %c: i1, in %d: i1, out out: i1) {
  // 创建一个AND门和一个OR门串联
  // CHECK: %{{.+}} = xlnx.lutn(%a, %b) {INIT = 8 : ui64} : (i1, i1) -> i1
  %0 = xlnx.lutn(%a, %b) {INIT = 8 : ui64} : (i1, i1) -> i1
  
  // CHECK: %{{.+}} = xlnx.lutn(%0, %c) {INIT = 14 : ui64} : (i1, i1) -> i1
  %1 = xlnx.lutn(%0, %c) {INIT = 14 : ui64} : (i1, i1) -> i1
  
  // 再添加一个XOR门
  // CHECK: %{{.+}} = xlnx.lutn(%1, %d) {INIT = 6 : ui64} : (i1, i1) -> i1
  %2 = xlnx.lutn(%1, %d) {INIT = 6 : ui64} : (i1, i1) -> i1
  
  hw.output %2 : i1
}

// CHECK-LABEL: hw.module @ParallelLuts
hw.module @ParallelLuts(in %a: i1, in %b: i1, in %c: i1, in %d: i1, 
                        out and_out: i1, out or_out: i1, out xor_out: i1) {
  // 并行创建多个LUT，实现不同功能
  // AND门
  // CHECK: %{{.+}} = xlnx.lutn(%a, %b) {INIT = 8 : ui64} : (i1, i1) -> i1
  %0 = xlnx.lutn(%a, %b) {INIT = 8 : ui64} : (i1, i1) -> i1
  
  // OR门
  // CHECK: %{{.+}} = xlnx.lutn(%b, %c) {INIT = 14 : ui64} : (i1, i1) -> i1
  %1 = xlnx.lutn(%b, %c) {INIT = 14 : ui64} : (i1, i1) -> i1
  
  // XOR门
  // CHECK: %{{.+}} = xlnx.lutn(%c, %d) {INIT = 6 : ui64} : (i1, i1) -> i1
  %2 = xlnx.lutn(%c, %d) {INIT = 6 : ui64} : (i1, i1) -> i1
  
  hw.output %0, %1, %2 : i1, i1, i1
}

// 测试6输入LUT，这是支持的最大输入数
// CHECK-LABEL: hw.module @SixInputLut
hw.module @SixInputLut(in %a: i1, in %b: i1, in %c: i1, in %d: i1, in %e: i1, in %f: i1, 
                        out out: i1) {
  // 6输入LUT
  // CHECK: %{{.+}} = xlnx.lutn(%a, %b, %c, %d, %e, %f) {INIT = 8589934590 : ui64} : (i1, i1, i1, i1, i1, i1) -> i1
  %0 = xlnx.lutn(%a, %b, %c, %d, %e, %f) {INIT = 8589934590 : ui64} : (i1, i1, i1, i1, i1, i1) -> i1
  
  hw.output %0 : i1
}
