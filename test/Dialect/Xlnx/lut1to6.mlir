// RUN: circt-opt %s -verify-diagnostics | FileCheck %s

// CHECK-LABEL: hw.module @LutSpecificOps
hw.module @LutSpecificOps(in %a: i1, in %b: i1, in %c: i1, in %d: i1, in %e: i1, in %f: i1,
                         out out1: i1, out out2: i1, out out3: i1, out out4: i1, out out5: i1, out out6: i1) {
  // Test LUT1 - Buffer (INIT=0x2, binary 10)
  // CHECK: %{{.+}} = xlnx.lut1(I0 : %a) {INIT = 2 : ui2} : i1 -> i1
  %0 = xlnx.lut1(I0: %a) {INIT = 2 : ui2} : i1 -> i1

  // Test LUT2 - AND gate (INIT=0x8, binary 1000)
  // CHECK: %{{.+}} = xlnx.lut2(I0 : %a, I1 : %b) {INIT = 8 : ui4} : i1, i1 -> i1
  %1 = xlnx.lut2(I0: %a, I1: %b) {INIT = 8 : ui4} : i1, i1 -> i1

  // Test LUT3 - OR gate (INIT=0xFE, binary 11111110)
  // CHECK: %{{.+}} = xlnx.lut3(I0 : %a, I1 : %b, I2 : %c) {INIT = 254 : ui8} : i1, i1, i1 -> i1
  %2 = xlnx.lut3(I0: %a, I1: %b, I2: %c) {INIT = 254 : ui8} : i1, i1, i1 -> i1

  // Test LUT4 - 4 input complex function
  // CHECK: %{{.+}} = xlnx.lut4(I0 : %a, I1 : %b, I2 : %c, I3 : %d) {INIT = 65535 : ui16} : i1, i1, i1, i1 -> i1
  %3 = xlnx.lut4(I0: %a, I1: %b, I2: %c, I3: %d) {INIT = 65535 : ui16} : i1, i1, i1, i1 -> i1

  // Test LUT5 - 5 input complex function
  // CHECK: %{{.+}} = xlnx.lut5(I0 : %a, I1 : %b, I2 : %c, I3 : %d, I4 : %e) {INIT = 4294967295 : ui32} : i1, i1, i1, i1, i1 -> i1
  %4 = xlnx.lut5(I0: %a, I1: %b, I2: %c, I3: %d, I4: %e) {INIT = 4294967295 : ui32} : i1, i1, i1, i1, i1 -> i1

  // Test LUT6 - 6 input complex function
  // CHECK: %{{.+}} = xlnx.lut6(I0 : %a, I1 : %b, I2 : %c, I3 : %d, I4 : %e, I5 : %f) {INIT = 8589934590 : ui64} : i1, i1, i1, i1, i1, i1 -> i1
  %5 = xlnx.lut6(I0: %a, I1: %b, I2: %c, I3: %d, I4: %e, I5: %f) {INIT = 8589934590 : ui64} : i1, i1, i1, i1, i1, i1 -> i1

  hw.output %0, %1, %2, %3, %4, %5 : i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: hw.module @LutCascaded
hw.module @LutCascaded(in %a: i1, in %b: i1, in %c: i1, in %d: i1, out out: i1) {
  // Use different specific types of LUTs in cascade
  // CHECK: %{{.+}} = xlnx.lut1(I0 : %a) {INIT = 2 : ui2} : i1 -> i1
  %0 = xlnx.lut1(I0: %a) {INIT = 2 : ui2} : i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.lut2(I0 : %0, I1 : %b) {INIT = 8 : ui4} : i1, i1 -> i1
  %1 = xlnx.lut2(I0: %0, I1: %b) {INIT = 8 : ui4} : i1, i1 -> i1
  
  // CHECK: %{{.+}} = xlnx.lut3(I0 : %1, I1 : %c, I2 : %d) {INIT = 254 : ui8} : i1, i1, i1 -> i1
  %2 = xlnx.lut3(I0: %1, I1: %c, I2: %d) {INIT = 254 : ui8} : i1, i1, i1 -> i1
  
  hw.output %2 : i1
}

// Test implementation of common logic gates
// CHECK-LABEL: hw.module @CommonLogicGates
hw.module @CommonLogicGates(in %a: i1, in %b: i1, 
                           out buffer_out: i1, out not_out: i1, out and_out: i1, 
                           out or_out: i1, out xor_out: i1, out nand_out: i1) {
  // Buffer - LUT1 (Buffer, INIT=0x2, binary 10)
  // CHECK: %{{.+}} = xlnx.lut1(I0 : %a) {INIT = 2 : ui2} : i1 -> i1
  %buf = xlnx.lut1(I0: %a) {INIT = 2 : ui2} : i1 -> i1
  
  // NOT - LUT1 (NOT gate, INIT=0x1, binary 01)
  // CHECK: %{{.+}} = xlnx.lut1(I0 : %a) {INIT = 1 : ui2} : i1 -> i1
  %not = xlnx.lut1(I0: %a) {INIT = 1 : ui2} : i1 -> i1
  
  // AND - LUT2 (AND gate, INIT=0x8, binary 1000)
  // CHECK: %{{.+}} = xlnx.lut2(I0 : %a, I1 : %b) {INIT = 8 : ui4} : i1, i1 -> i1
  %and = xlnx.lut2(I0: %a, I1: %b) {INIT = 8 : ui4} : i1, i1 -> i1
  
  // OR - LUT2 (OR gate, INIT=0xE, binary 1110)
  // CHECK: %{{.+}} = xlnx.lut2(I0 : %a, I1 : %b) {INIT = 14 : ui4} : i1, i1 -> i1
  %or = xlnx.lut2(I0: %a, I1: %b) {INIT = 14 : ui4} : i1, i1 -> i1
  
  // XOR - LUT2 (XOR gate, INIT=0x6, binary 0110)
  // CHECK: %{{.+}} = xlnx.lut2(I0 : %a, I1 : %b) {INIT = 6 : ui4} : i1, i1 -> i1
  %xor = xlnx.lut2(I0: %a, I1: %b) {INIT = 6 : ui4} : i1, i1 -> i1
  
  // NAND - LUT2 (NAND gate, INIT=0x7, binary 0111)
  // CHECK: %{{.+}} = xlnx.lut2(I0 : %a, I1 : %b) {INIT = 7 : ui4} : i1, i1 -> i1
  %nand = xlnx.lut2(I0: %a, I1: %b) {INIT = 7 : ui4} : i1, i1 -> i1
  
  hw.output %buf, %not, %and, %or, %xor, %nand : i1, i1, i1, i1, i1, i1
}

// Test mixed use of different LUT formats
// CHECK-LABEL: hw.module @MixedLutTypes
hw.module @MixedLutTypes(in %a: i1, in %b: i1, in %c: i1, in %d: i1, out out: i1) {
  // Use lutn format
  // CHECK: %{{.+}} = xlnx.lutn(%a, %b) {INIT = 8 : ui64} : (i1, i1) -> i1
  %0 = xlnx.lutn(%a, %b) {INIT = 8 : ui64} : (i1, i1) -> i1
  
  // Use lut2 format
  // CHECK: %{{.+}} = xlnx.lut2(I0 : %c, I1 : %d) {INIT = 14 : ui4} : i1, i1 -> i1
  %1 = xlnx.lut2(I0: %c, I1: %d) {INIT = 14 : ui4} : i1, i1 -> i1
  
  // Mix two formats
  // CHECK: %{{.+}} = xlnx.lut2(I0 : %0, I1 : %1) {INIT = 6 : ui4} : i1, i1 -> i1
  %2 = xlnx.lut2(I0: %0, I1: %1) {INIT = 6 : ui4} : i1, i1 -> i1
  
  hw.output %2 : i1
} 