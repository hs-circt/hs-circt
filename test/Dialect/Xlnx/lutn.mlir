// RUN: circt-opt %s -verify-diagnostics | FileCheck %s

// CHECK-LABEL: hw.module @LutExamples
hw.module @LutExamples(in %a: i1, in %b: i1, in %c: i1, out out1: i1, out out2: i1, out out3: i1, out out4: i1) {
  // 1 Input LUT (Buffer, INIT=0x2, binary 10)
  // CHECK: %{{.+}} = xlnx.lutn(%a) {INIT = 2 : ui64} : (i1) -> i1
  %0 = xlnx.lutn(%a) {INIT = 2 : ui64} : (i1) -> i1

  // 2 Input LUT (AND gate, INIT=0x8, binary 1000)
  // CHECK: %{{.+}} = xlnx.lutn(%a, %b) {INIT = 8 : ui64} : (i1, i1) -> i1
  %1 = xlnx.lutn(%a, %b) {INIT = 8 : ui64} : (i1, i1) -> i1

  // 3 Input LUT (OR gate, INIT=0xFE, binary 11111110)
  // CHECK: %{{.+}} = xlnx.lutn(%a, %b, %c) {INIT = 254 : ui64} : (i1, i1, i1) -> i1
  %2 = xlnx.lutn(%a, %b, %c) {INIT = 254 : ui64} : (i1, i1, i1) -> i1

  // Create different LUTs with the same inputs (XOR gate, INIT=0x6, binary 0110)
  // CHECK: %{{.+}} = xlnx.lutn(%a, %b) {INIT = 6 : ui64} : (i1, i1) -> i1
  %3 = xlnx.lutn(%a, %b) {INIT = 6 : ui64} : (i1, i1) -> i1

  hw.output %0, %1, %2, %3 : i1, i1, i1, i1
}

// CHECK-LABEL: hw.module @CascadedLuts
hw.module @CascadedLuts(in %a: i1, in %b: i1, in %c: i1, in %d: i1, out out: i1) {
  // Create a cascaded AND gate and OR gate
  // CHECK: %{{.+}} = xlnx.lutn(%a, %b) {INIT = 8 : ui64} : (i1, i1) -> i1
  %0 = xlnx.lutn(%a, %b) {INIT = 8 : ui64} : (i1, i1) -> i1
  
  // CHECK: %{{.+}} = xlnx.lutn(%0, %c) {INIT = 14 : ui64} : (i1, i1) -> i1
  %1 = xlnx.lutn(%0, %c) {INIT = 14 : ui64} : (i1, i1) -> i1
  
  // Add another XOR gate
  // CHECK: %{{.+}} = xlnx.lutn(%1, %d) {INIT = 6 : ui64} : (i1, i1) -> i1
  %2 = xlnx.lutn(%1, %d) {INIT = 6 : ui64} : (i1, i1) -> i1
  
  hw.output %2 : i1
}

// CHECK-LABEL: hw.module @ParallelLuts
hw.module @ParallelLuts(in %a: i1, in %b: i1, in %c: i1, in %d: i1, 
                        out and_out: i1, out or_out: i1, out xor_out: i1) {
  // Create multiple LUTs in parallel to implement different functions
  // AND gate
  // CHECK: %{{.+}} = xlnx.lutn(%a, %b) {INIT = 8 : ui64} : (i1, i1) -> i1
  %0 = xlnx.lutn(%a, %b) {INIT = 8 : ui64} : (i1, i1) -> i1
  
  // OR gate
  // CHECK: %{{.+}} = xlnx.lutn(%b, %c) {INIT = 14 : ui64} : (i1, i1) -> i1
  %1 = xlnx.lutn(%b, %c) {INIT = 14 : ui64} : (i1, i1) -> i1
  
  // XOR gate
  // CHECK: %{{.+}} = xlnx.lutn(%c, %d) {INIT = 6 : ui64} : (i1, i1) -> i1
  %2 = xlnx.lutn(%c, %d) {INIT = 6 : ui64} : (i1, i1) -> i1
  
  hw.output %0, %1, %2 : i1, i1, i1
}

// CHECK-LABEL: hw.module @SixInputLut
hw.module @SixInputLut(in %a: i1, in %b: i1, in %c: i1, in %d: i1, in %e: i1, in %f: i1, 
                        out out: i1) {
  // 6 Input LUT
  // CHECK: %{{.+}} = xlnx.lutn(%a, %b, %c, %d, %e, %f) {INIT = 8589934590 : ui64} : (i1, i1, i1, i1, i1, i1) -> i1
  %0 = xlnx.lutn(%a, %b, %c, %d, %e, %f) {INIT = 8589934590 : ui64} : (i1, i1, i1, i1, i1, i1) -> i1
  
  hw.output %0 : i1
}
