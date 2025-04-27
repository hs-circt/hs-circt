module {
  hw.module @top(in %a : i2, in %b : i2, out c : i3) {
    %false = hw.constant false
    %0 = comb.extract %a from 1 : (i2) -> i1
    %1 = comb.extract %a from 0 : (i2) -> i1
    %2 = comb.extract %b from 1 : (i2) -> i1
    %3 = comb.extract %b from 0 : (i2) -> i1
    %4 = comb.concat %3, %2 : i1, i1
    %5 = comb.concat %1, %0 : i1, i1
    %add_3.o, %add_3.cout = hw.instance "add_3" @add_2u_2u(cin: %false: i1, i1: %5: i2, i2: %4: i2) -> (o: i2, cout: i1)
    %6 = comb.extract %add_3.o from 1 : (i2) -> i1
    %7 = comb.extract %add_3.o from 0 : (i2) -> i1
    %8 = comb.concat %add_3.cout, %6, %7 : i1, i1, i1
    hw.output %8 : i3
  }
  hw.generator.schema @OperAdder, "OPERADDER", ["input1Size", "input2Size", "outputSize"]
  hw.module.generated @add_2u_2u, @OperAdder(in %cin : i1, in %i1 : i2, in %i2 : i2, out o : i2, out cout : i1)
  attributes {input1Size = 2 : ui32, input2Size = 2 : ui32, outputSize = 2 : ui32}
}
