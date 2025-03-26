// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @test_logical_ops
func.func @test_logical_ops(%arg0: i1, %arg1: i1) -> (i1, i1) {
  // CHECK-NEXT: %0 = vfc.and %arg0, %arg1 : i1, i1 -> i1
  %0 = vfc.and %arg0, %arg1 : i1, i1 -> i1
  
  // CHECK-NEXT: %1 = vfc.or %arg0, %arg1 : i1, i1 -> i1
  %1 = vfc.or %arg0, %arg1 : i1, i1 -> i1
  
  // CHECK-NEXT: return %0, %1 : i1, i1
  return %0, %1 : i1, i1
}

// Test with constant values
// CHECK-LABEL: func @test_const_logical_ops
func.func @test_const_logical_ops() -> (i1, i1, i1, i1) {
  // CHECK-NEXT: %true = hw.constant true
  %true = hw.constant true
  // CHECK-NEXT: %false = hw.constant false
  %false = hw.constant false
  
  // CHECK-NEXT: %0 = vfc.and %true, %true : i1, i1 -> i1
  %0 = vfc.and %true, %true : i1, i1 -> i1
  
  // CHECK-NEXT: %1 = vfc.and %true, %false : i1, i1 -> i1
  %1 = vfc.and %true, %false : i1, i1 -> i1
  
  // CHECK-NEXT: %2 = vfc.or %false, %false : i1, i1 -> i1
  %2 = vfc.or %false, %false : i1, i1 -> i1
  
  // CHECK-NEXT: %3 = vfc.or %true, %false : i1, i1 -> i1
  %3 = vfc.or %true, %false : i1, i1 -> i1
  
  // CHECK-NEXT: return %0, %1, %2, %3 : i1, i1, i1, i1
  return %0, %1, %2, %3 : i1, i1, i1, i1
} 