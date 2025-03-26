// RUN: circt-opt %s | FileCheck %s

// CHECK-LABEL: func @test_nand
func.func @test_nand(%arg0: i1, %arg1: i1) -> i1 {
  // CHECK: %[[RES:.*]] = gtech.nand %arg0, %arg1 : i1
  %0 = gtech.nand %arg0, %arg1 : i1
  // CHECK: return %[[RES]]
  return %0 : i1
}