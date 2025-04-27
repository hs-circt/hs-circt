module {
  hw.module @LessThan_4u_4u(in %cin : i1, in %a : i4, in %b : i4, out o : i1) {
    %0 = comb.extract %a from 3 : (i4) -> i1
    %1 = comb.extract %a from 2 : (i4) -> i1
    %2 = comb.extract %a from 1 : (i4) -> i1
    %3 = comb.extract %a from 0 : (i4) -> i1
    %4 = comb.extract %b from 3 : (i4) -> i1
    %5 = comb.extract %b from 2 : (i4) -> i1
    %6 = comb.extract %b from 1 : (i4) -> i1
    %7 = comb.extract %b from 0 : (i4) -> i1
    %8 = comb.xor bin %3, %7 {veri_inst = "i1"} : i1
    %9 = comb.mux %8, %cin, %7 {veri_inst = "i2"} : i1
    %10 = comb.xor bin %2, %6 {veri_inst = "i3"} : i1
    %11 = comb.mux %10, %9, %6 {veri_inst = "i4"} : i1
    %12 = comb.xor bin %1, %5 {veri_inst = "i5"} : i1
    %13 = comb.mux %12, %11, %5 {veri_inst = "i6"} : i1
    %14 = comb.xor bin %0, %4 {veri_inst = "i7"} : i1
    %15 = comb.mux %14, %13, %4 {veri_inst = "i8"} : i1
    hw.output %15 : i1
  }

  hw.module @LessThan_3u_3u(in %cin : i1, in %a : i3, in %b : i3, out o : i1) {
    %0 = comb.extract %a from 2 : (i3) -> i1
    %1 = comb.extract %a from 1 : (i3) -> i1
    %2 = comb.extract %a from 0 : (i3) -> i1
    %3 = comb.extract %b from 2 : (i3) -> i1
    %4 = comb.extract %b from 1 : (i3) -> i1
    %5 = comb.extract %b from 0 : (i3) -> i1
    %6 = comb.xor bin %2, %5 {veri_inst = "i1"} : i1
    %7 = comb.mux %6, %cin, %5 {veri_inst = "i2"} : i1
    %8 = comb.xor bin %1, %4 {veri_inst = "i3"} : i1
    %9 = comb.mux %8, %7, %4 {veri_inst = "i4"} : i1
    %10 = comb.xor bin %0, %3 {veri_inst = "i5"} : i1
    %11 = comb.mux %10, %9, %3 {veri_inst = "i6"} : i1
    hw.output %11 : i1
  }
  hw.module @LessThan_2u_2u(in %cin : i1, in %a : i2, in %b : i2, out o : i1) {
  %0 = comb.extract %a from 1 : (i2) -> i1
  %1 = comb.extract %a from 0 : (i2) -> i1
  %2 = comb.extract %b from 1 : (i2) -> i1
  %3 = comb.extract %b from 0 : (i2) -> i1
  %4 = comb.xor bin %1, %3 {veri_inst = "i1"} : i1
  %5 = comb.mux %4, %cin, %3 {veri_inst = "i2"} : i1
  %6 = comb.xor bin %0, %2 {veri_inst = "i3"} : i1
  %7 = comb.mux %6, %5, %2 {veri_inst = "i4"} : i1
  hw.output %7 : i1
}

hw.module @top(in %a : i2, in %b : i2, out c : i2) {
  %0 = comb.extract %a from 1 : (i2) -> i1
  %1 = comb.extract %a from 0 : (i2) -> i1
  %2 = comb.extract %b from 1 : (i2) -> i1
  %3 = comb.extract %b from 0 : (i2) -> i1
  %4 = comb.concat %3, %2 : i1, i1
  %5 = comb.concat %1, %0 : i1, i1
  %div_3_0 = hw.instance "div_2u_2u" @div_2u_2u(i1: %5: i2, i2: %4: i2) -> (o: i2) {input1Size = 2 : i32, input2Size = 2 : i32, outputSize = 2 : i32}
  %6 = comb.extract %div_3_0 from 1 : (i2) -> i1
  %7 = comb.extract %div_3_0 from 0 : (i2) -> i1
  %8 = comb.concat %6, %7 : i1, i1
  hw.output %8 : i2
}

hw.generator.schema @OperDiv, "OPERDIV", ["input1Size", "input2Size", "outputSize"]
hw.module.generated @div_2u_2u, @OperDiv(in %i1 : i2, in %i2 : i2, out o : i2) attributes {input1Size = 2 : ui32, input2Size = 2 : ui32, outputSize = 2 : ui32}

}
