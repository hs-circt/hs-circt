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

def XlnxLutNOp : XlnxOp<"lutn", [Commutative]> {
  let summary = "Lookup table operation";
  let description = [{
    Lookup table operation with a single output bit. The INIT attribute defines 
    the lookup table contents as a 64-bit integer where each bit represents the output 
    value for the corresponding input pattern.

    Example:
    ```mlir
    // 2-input AND gate
    %result = xlnx.lutn(%a, %b) <{INIT = 8 : ui64}> : (i1, i1) -> i1
    // 3-input OR gate  
    %result = xlnx.lutn(%a, %b, %c) <{INIT = 254 : ui64}> : (i1, i1, i1) -> i1
    ```

    The number of inputs must be between 1 and 6 inclusive.
  }];

  let arguments = (ins 
    Variadic<I1>:$inputs,
    UI64Attr:$INIT
  );
  let results = (outs I1:$result);

  let assemblyFormat = [{
    `(` $inputs `)` attr-dict `:` functional-type($inputs, $result)
  }];

  let hasVerifier = 1;

  let builders = [
    // Build from inputs and INIT value
    OpBuilder<(ins "::mlir::ValueRange":$inputs, "uint64_t":$init), [{
      $_state.addOperands(inputs);
      $_state.addAttribute("INIT", $_builder.getIntegerAttr($_builder.getIntegerType(64, false), init));
      $_state.addTypes($_builder.getI1Type());
    }]>
  ];
}