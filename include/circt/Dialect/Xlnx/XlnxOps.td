//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Bytecode/BytecodeOpInterface.td"
include "circt/Dialect/Xlnx/XlnxDialect.td"

// Base class for Xlnx operations
class XlnxOp<string mnemonic, list<Trait> traits = []> : Op<XlnxDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// LUT Operation
//===----------------------------------------------------------------------===//

// Base class for LUT operations
class XlnxLutBase<string mnemonic, list<Trait> traits = []> : XlnxOp<mnemonic, !listconcat(traits, [])> {
  let summary = "Lookup table operation";
  let results = (outs I1:$result);
  let hasVerifier = 1;
  let description = [{
    Lookup table operation with a single output bit. The INIT attribute defines 
    the lookup table contents as a 64-bit integer where each bit represents the output 
    value for the corresponding input pattern.

    Example:
    ```mlir
    // 2-input LUT
    %result = xlnx.lutn(%a, %b) <{INIT = 8 : ui64}> : (i1, i1) -> i1
    // or
    %result = xlnx.lut2(%a, %b) <{INIT = 8 : ui64}> : (i1, i1) -> i1

    // 3-input LUT
    %result = xlnx.lutn(%a, %b, %c) <{INIT = 254 : ui64}> : (i1, i1, i1) -> i1
    // or
    %result = xlnx.lut3(%a, %b, %c) <{INIT = 254 : ui64}> : (i1, i1, i1) -> i1
    ```

    The number of inputs must be between 1 and 6 inclusive in xlnx.lutn.
  }];

}

def XlnxLutNOp : XlnxLutBase<"lutn"> {
  let arguments = (ins Variadic<I1>:$inputs, UI64Attr:$INIT);
  let results = (outs I1:$result);
  let assemblyFormat = "`(` $inputs `)` attr-dict `:` functional-type($inputs, $result)";
}

def XlnxLut1Op : XlnxLutBase<"lut1"> {
  let arguments = (ins I1:$input, UI64Attr:$INIT);
  let results = (outs I1:$result);
  let assemblyFormat = "`(` `I0` `:` $input `)` attr-dict `:` type($input) `->` type($result)";
  let extraClassDeclaration = [{
    // #define LUT1_NAMED_ASSOCIATED_BUILD
    // template <typename ... Actions>
    // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Actions &&... actions);
  }];
}

def XlnxLut2Op : XlnxLutBase<"lut2"> {
  let arguments = (ins I1:$input0, I1:$input1, UI64Attr:$INIT);
  let results = (outs I1:$result);
  let assemblyFormat = "`(` `I0` `:` $input0 `,` `I1` `:` $input1 `)` attr-dict `:` type($input0) `,` type($input1) `->` type($result)";
}

def XlnxLut3Op : XlnxLutBase<"lut3"> {
  let arguments = (ins I1:$input0, I1:$input1, I1:$input2, UI64Attr:$INIT);
  let results = (outs I1:$result);
  let assemblyFormat = "`(` `I0` `:` $input0 `,` `I1` `:` $input1 `,` `I2` `:` $input2 `)` attr-dict `:` type($input0) `,` type($input1) `,` type($input2) `->` type($result)";
}

def XlnxLut4Op : XlnxLutBase<"lut4"> {
  let arguments = (ins I1:$input0, I1:$input1, I1:$input2, I1:$input3, UI64Attr:$INIT);
  let results = (outs I1:$result);
  let assemblyFormat = "`(` `I0` `:` $input0 `,` `I1` `:` $input1 `,` `I2` `:` $input2 `,` `I3` `:` $input3 `)` attr-dict `:` type($input0) `,` type($input1) `,` type($input2) `,` type($input3) `->` type($result)";
}

def XlnxLut5Op : XlnxLutBase<"lut5"> {
  let arguments = (ins I1:$input0, I1:$input1, I1:$input2, I1:$input3, I1:$input4, UI64Attr:$INIT);
  let results = (outs I1:$result);
  let assemblyFormat = "`(` `I0` `:` $input0 `,` `I1` `:` $input1 `,` `I2` `:` $input2 `,` `I3` `:` $input3 `,` `I4` `:` $input4 `)` attr-dict `:` type($input0) `,` type($input1) `,` type($input2) `,` type($input3) `,` type($input4) `->` type($result)";
}

def XlnxLut6Op : XlnxLutBase<"lut6"> {
  let arguments = (ins I1:$input0, I1:$input1, I1:$input2, I1:$input3, I1:$input4, I1:$input5, UI64Attr:$INIT);
  let results = (outs I1:$result);
  let assemblyFormat = "`(` `I0` `:` $input0 `,` `I1` `:` $input1 `,` `I2` `:` $input2 `,` `I3` `:` $input3 `,` `I4` `:` $input4 `,` `I5` `:` $input5 `)` attr-dict `:` type($input0) `,` type($input1) `,` type($input2) `,` type($input3) `,` type($input4) `,` type($input5) `->` type($result)";
}
