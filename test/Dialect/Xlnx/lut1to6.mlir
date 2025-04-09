// RUN: circt-opt %s -verify-diagnostics | FileCheck %s

// CHECK-LABEL: hw.module @LutSpecificOps
hw.module @LutSpecificOps(in %a: i1, in %b: i1, in %c: i1, in %d: i1, in %e: i1, in %f: i1,
                         out out1: i1, out out2: i1, out out3: i1, out out4: i1, out out5: i1, out out6: i1) {
  // 测试 LUT1 - 缓冲器 (INIT=0x2，二进制10)
  // CHECK: %{{.+}} = xlnx.lut1(I0 : %a) {INIT = 2 : ui64} : i1 -> i1
  %0 = xlnx.lut1(I0: %a) {INIT = 2 : ui64} : i1 -> i1

  // 测试 LUT2 - AND门 (INIT=0x8，二进制1000)
  // CHECK: %{{.+}} = xlnx.lut2(I0 : %a, I1 : %b) {INIT = 8 : ui64} : i1, i1 -> i1
  %1 = xlnx.lut2(I0: %a, I1: %b) {INIT = 8 : ui64} : i1, i1 -> i1

  // 测试 LUT3 - OR门 (INIT=0xFE，二进制11111110)
  // CHECK: %{{.+}} = xlnx.lut3(I0 : %a, I1 : %b, I2 : %c) {INIT = 254 : ui64} : i1, i1, i1 -> i1
  %2 = xlnx.lut3(I0: %a, I1: %b, I2: %c) {INIT = 254 : ui64} : i1, i1, i1 -> i1

  // 测试 LUT4 - 4输入复杂函数
  // CHECK: %{{.+}} = xlnx.lut4(I0 : %a, I1 : %b, I2 : %c, I3 : %d) {INIT = 65535 : ui64} : i1, i1, i1, i1 -> i1
  %3 = xlnx.lut4(I0: %a, I1: %b, I2: %c, I3: %d) {INIT = 65535 : ui64} : i1, i1, i1, i1 -> i1

  // 测试 LUT5 - 5输入复杂函数
  // CHECK: %{{.+}} = xlnx.lut5(I0 : %a, I1 : %b, I2 : %c, I3 : %d, I4 : %e) {INIT = 4294967295 : ui64} : i1, i1, i1, i1, i1 -> i1
  %4 = xlnx.lut5(I0: %a, I1: %b, I2: %c, I3: %d, I4: %e) {INIT = 4294967295 : ui64} : i1, i1, i1, i1, i1 -> i1

  // 测试 LUT6 - 6输入复杂函数
  // CHECK: %{{.+}} = xlnx.lut6(I0 : %a, I1 : %b, I2 : %c, I3 : %d, I4 : %e, I5 : %f) {INIT = 8589934590 : ui64} : i1, i1, i1, i1, i1, i1 -> i1
  %5 = xlnx.lut6(I0: %a, I1: %b, I2: %c, I3: %d, I4: %e, I5: %f) {INIT = 8589934590 : ui64} : i1, i1, i1, i1, i1, i1 -> i1

  hw.output %0, %1, %2, %3, %4, %5 : i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: hw.module @LutCascaded
hw.module @LutCascaded(in %a: i1, in %b: i1, in %c: i1, in %d: i1, out out: i1) {
  // 使用各种具体类型的LUT级联
  // CHECK: %{{.+}} = xlnx.lut1(I0 : %a) {INIT = 2 : ui64} : i1 -> i1
  %0 = xlnx.lut1(I0: %a) {INIT = 2 : ui64} : i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.lut2(I0 : %0, I1 : %b) {INIT = 8 : ui64} : i1, i1 -> i1
  %1 = xlnx.lut2(I0: %0, I1: %b) {INIT = 8 : ui64} : i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.lut3(I0 : %1, I1 : %c, I2 : %d) {INIT = 254 : ui64} : i1, i1, i1 -> i1
  %2 = xlnx.lut3(I0: %1, I1: %c, I2: %d) {INIT = 254 : ui64} : i1, i1, i1 -> i1
  
  hw.output %2 : i1
}

// 测试常见逻辑门的实现
// CHECK-LABEL: hw.module @CommonLogicGates
hw.module @CommonLogicGates(in %a: i1, in %b: i1, 
                           out buffer_out: i1, out not_out: i1, out and_out: i1, 
                           out or_out: i1, out xor_out: i1, out nand_out: i1) {
  // Buffer - LUT1 (缓冲器，INIT=0x2，二进制10)
  // CHECK: %{{.+}} = xlnx.lut1(I0 : %a) {INIT = 2 : ui64} : i1 -> i1
  %buf = xlnx.lut1(I0: %a) {INIT = 2 : ui64} : i1 -> i1
  
  // NOT - LUT1 (非门，INIT=0x1，二进制01)
  // CHECK: %{{.+}} = xlnx.lut1(I0 : %a) {INIT = 1 : ui64} : i1 -> i1
  %not = xlnx.lut1(I0: %a) {INIT = 1 : ui64} : i1 -> i1
  
  // AND - LUT2 (与门，INIT=0x8，二进制1000)
  // CHECK: %{{.+}} = xlnx.lut2(I0 : %a, I1 : %b) {INIT = 8 : ui64} : i1, i1 -> i1
  %and = xlnx.lut2(I0: %a, I1: %b) {INIT = 8 : ui64} : i1, i1 -> i1
  
  // OR - LUT2 (或门，INIT=0xE，二进制1110)
  // CHECK: %{{.+}} = xlnx.lut2(I0 : %a, I1 : %b) {INIT = 14 : ui64} : i1, i1 -> i1
  %or = xlnx.lut2(I0: %a, I1: %b) {INIT = 14 : ui64} : i1, i1 -> i1
  
  // XOR - LUT2 (异或门，INIT=0x6，二进制0110)
  // CHECK: %{{.+}} = xlnx.lut2(I0 : %a, I1 : %b) {INIT = 6 : ui64} : i1, i1 -> i1
  %xor = xlnx.lut2(I0: %a, I1: %b) {INIT = 6 : ui64} : i1, i1 -> i1
  
  // NAND - LUT2 (与非门，INIT=0x7，二进制0111)
  // CHECK: %{{.+}} = xlnx.lut2(I0 : %a, I1 : %b) {INIT = 7 : ui64} : i1, i1 -> i1
  %nand = xlnx.lut2(I0: %a, I1: %b) {INIT = 7 : ui64} : i1, i1 -> i1
  
  hw.output %buf, %not, %and, %or, %xor, %nand : i1, i1, i1, i1, i1, i1
}

// 测试不同LUT形式的混合使用
// CHECK-LABEL: hw.module @MixedLutTypes
hw.module @MixedLutTypes(in %a: i1, in %b: i1, in %c: i1, in %d: i1, out out: i1) {
  // 使用lutn格式
  // CHECK: %{{.+}} = xlnx.lutn(%a, %b) {INIT = 8 : ui64} : (i1, i1) -> i1
  %0 = xlnx.lutn(%a, %b) {INIT = 8 : ui64} : (i1, i1) -> i1
  
  // 使用lut2格式
  // CHECK: %{{.+}} = xlnx.lut2(I0 : %c, I1 : %d) {INIT = 14 : ui64} : i1, i1 -> i1
  %1 = xlnx.lut2(I0: %c, I1: %d) {INIT = 14 : ui64} : i1, i1 -> i1
  
  // 将两种格式混合使用
  // CHECK: %{{.+}} = xlnx.lut2(I0 : %0, I1 : %1) {INIT = 6 : ui64} : i1, i1 -> i1
  %2 = xlnx.lut2(I0: %0, I1: %1) {INIT = 6 : ui64} : i1, i1 -> i1
  
  hw.output %2 : i1
} 