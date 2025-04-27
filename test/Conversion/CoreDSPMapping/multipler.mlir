module {
  hw.module @top(in %a : i2, in %b : i2, out c : i4) {
    %0 = comb.extract %a from 1 : (i2) -> i1
    %1 = comb.extract %a from 0 : (i2) -> i1
    %2 = comb.extract %b from 1 : (i2) -> i1
    %3 = comb.extract %b from 0 : (i2) -> i1
    %4 = comb.concat %3, %2 : i1, i1
    %5 = comb.concat %1, %0 : i1, i1
    %mult_3.o = hw.instance "mult_3" @mult_2u_2u(i1: %5: i2, i2: %4: i2) -> (o: i4) {input1Size = 2 : i32, input2Size = 2 : i32, outputSize = 4 : i32}
    %6 = comb.extract %mult_3.o from 3 : (i4) -> i1
    %7 = comb.extract %mult_3.o from 2 : (i4) -> i1
    %8 = comb.extract %mult_3.o from 1 : (i4) -> i1
    %9 = comb.extract %mult_3.o from 0 : (i4) -> i1
    %10 = comb.concat %6, %7, %8, %9 : i1, i1, i1, i1
    hw.output %10 : i4
  }
  hw.generator.schema @OperMult, "OPERMULT", ["input1Size", "input2Size", "outputSize"]
  hw.module.generated @mult_2u_2u, @OperMult(in %i1 : i2, in %i2 : i2, out o : i4) attributes {input1Size = 2 : ui32, input2Size = 2 : ui32, outputSize = 4 : ui32}
}
