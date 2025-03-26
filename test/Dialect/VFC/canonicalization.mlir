// RUN: circt-opt %s -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func @canonicalize_and_true
func.func @canonicalize_and_true(%arg0: i1) -> i1 {
  // CHECK-NEXT: %true = hw.constant true
  %true = hw.constant true
  // CHECK-NEXT: %0 = vfc.and %arg0, %true : i1, i1 -> i1
  %0 = vfc.and %arg0, %true : i1, i1 -> i1
  // CHECK-NEXT: return %0 : i1
  return %0 : i1
}

// -----

// CHECK-LABEL: func @canonicalize_and_false
func.func @canonicalize_and_false(%arg0: i1) -> i1 {
  // CHECK-NEXT: %false = hw.constant false
  %false = hw.constant false
  // CHECK-NEXT: %0 = vfc.and %arg0, %false : i1, i1 -> i1
  %0 = vfc.and %arg0, %false : i1, i1 -> i1
  // CHECK-NEXT: return %0 : i1
  return %0 : i1
}

// -----

// CHECK-LABEL: func @canonicalize_or_true
func.func @canonicalize_or_true(%arg0: i1) -> i1 {
  // CHECK-NEXT: %true = hw.constant true
  %true = hw.constant true
  // CHECK-NEXT: %0 = vfc.or %arg0, %true : i1, i1 -> i1
  %0 = vfc.or %arg0, %true : i1, i1 -> i1
  // CHECK-NEXT: return %0 : i1
  return %0 : i1
}

// -----

// CHECK-LABEL: func @canonicalize_or_false
func.func @canonicalize_or_false(%arg0: i1) -> i1 {
  // CHECK-NEXT: %false = hw.constant false
  %false = hw.constant false
  // CHECK-NEXT: %0 = vfc.or %arg0, %false : i1, i1 -> i1
  %0 = vfc.or %arg0, %false : i1, i1 -> i1
  // CHECK-NEXT: return %0 : i1
  return %0 : i1
} 