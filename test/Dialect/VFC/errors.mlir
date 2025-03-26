// RUN: circt-opt %s -split-input-file -verify-diagnostics

func.func @test_wrong_type_and(%arg0: i32, %arg1: i1) {
  // expected-error @+1 {{requires the same type for all operands and results}}
  %0 = vfc.and %arg0, %arg1 : i32, i1 -> i1
  return
}

// -----

func.func @test_wrong_type_or(%arg0: i1, %arg1: i32) {
  // expected-error @+1 {{requires the same type for all operands and results}}
  %0 = vfc.or %arg0, %arg1 : i1, i32 -> i1
  return
}

// -----

func.func @test_wrong_result_type(%arg0: i1, %arg1: i1) {
  // expected-error @+1 {{requires the same type for all operands and results}}
  %0 = vfc.and %arg0, %arg1 : i1, i1 -> i32
  return
} 