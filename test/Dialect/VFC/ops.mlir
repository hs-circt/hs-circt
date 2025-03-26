// RUN: circt-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// Basic operation tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_and_i1
func.func @test_and_i1(%arg0: i1, %arg1: i1) -> i1 {
  %0 = vfc.and %arg0, %arg1 : i1, i1 -> i1
  return %0 : i1
}

// CHECK-LABEL: func @test_and_i8
func.func @test_and_i8(%arg0: i8, %arg1: i8) -> i8 {
  %0 = vfc.and %arg0, %arg1 : i8, i8 -> i8
  return %0 : i8
}

// CHECK-LABEL: func @test_and_i32
func.func @test_and_i32(%arg0: i32, %arg1: i32) -> i32 {
  %0 = vfc.and %arg0, %arg1 : i32, i32 -> i32
  return %0 : i32
}

// CHECK-LABEL: func @test_or_i1
func.func @test_or_i1(%arg0: i1, %arg1: i1) -> i1 {
  %0 = vfc.or %arg0, %arg1 : i1, i1 -> i1
  return %0 : i1
}

// CHECK-LABEL: func @test_or_i8
func.func @test_or_i8(%arg0: i8, %arg1: i8) -> i8 {
  %0 = vfc.or %arg0, %arg1 : i8, i8 -> i8
  return %0 : i8
}

// CHECK-LABEL: func @test_or_i32
func.func @test_or_i32(%arg0: i32, %arg1: i32) -> i32 {
  %0 = vfc.or %arg0, %arg1 : i32, i32 -> i32
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// Operation verification tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_and_mixed_types
func.func @test_and_mixed_types(%arg0: i8, %arg1: i16) -> i16 {
  // expected-error @+1 {{requires the same type for all operands and results}}
  %0 = vfc.and %arg0, %arg1 : i8, i16 -> i16
  return %0 : i16
}

// -----

// CHECK-LABEL: func @test_or_mixed_types
func.func @test_or_mixed_types(%arg0: i8, %arg1: i16) -> i16 {
  // expected-error @+1 {{requires the same type for all operands and results}}
  %0 = vfc.or %arg0, %arg1 : i8, i16 -> i16
  return %0 : i16
} 