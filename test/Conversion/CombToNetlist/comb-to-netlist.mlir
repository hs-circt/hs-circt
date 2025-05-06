// RUN: circt-opt %s --convert-comb-to-netlist | FileCheck %s

// CHECK-LABEL: @test
hw.module @test(in %a: i1, in %b: i1, in %c: i1, in %d: i1, out out: i1)  {
  // CHECK-NEXT: %[[LUT:.+]] = xlnx.lutn(%a, %b, %c, %d) {INIT = 63624 : ui64} : (i1, i1, i1, i1) -> i1
  // CHECK-NEXT: hw.output %[[LUT]] : i1
  %0 = comb.and %a, %b : i1
  %1 = comb.and %c, %d : i1
  %2 = comb.or %0, %1 : i1
  hw.output %2 : i1
}
