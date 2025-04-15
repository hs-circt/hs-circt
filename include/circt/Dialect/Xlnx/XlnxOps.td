//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO:
// - Add interface to get primitive module name
// - Add attribute to LutN to specify N number of inputs
// - Modify INIT attribute size to match input size, should not be 64-bit
// - Implement FlipFlop primitive
// - Implement Latch primitive

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/Xlnx/XlnxOpInterfaces.td"
include "circt/Dialect/Seq/SeqTypes.td"
include "mlir/Bytecode/BytecodeOpInterface.td"
include "circt/Dialect/Xlnx/XlnxDialect.td"
include "circt/Dialect/Xlnx/XlnxTypes.td"

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
    %result = xlnx.lut2(I0 : %a, I1 : %b) <{INIT = 8 : ui4}> : (i1, i1) -> i1

    // 3-input LUT
    %result = xlnx.lutn(%a, %b, %c) <{INIT = 254 : ui64}> : (i1, i1, i1) -> i1
    // or
    %result = xlnx.lut3(I0 : %a, I1 : %b, I2 : %c) <{INIT = 254 : ui8}> : (i1, i1, i1) -> i1
    ```

    The number of inputs must be between 1 and 6 inclusive in xlnx.lutn.
  }];

}

def XlnxLutNOp : XlnxLutBase<"lutn"> {
  let arguments = (ins UI64Attr:$INIT, Variadic<I1>:$inputs);
  let assemblyFormat = "`(` $inputs `)` attr-dict `:` functional-type($inputs, $result)";
}

def XlnxLut1Op : XlnxLutBase<"lut1"> {
  let arguments = (ins UI2Attr:$INIT, I1:$input);
  let assemblyFormat = "`(` `I0` `:` $input `)` attr-dict `:` type($input) `->` type($result)";
  let extraClassDeclaration = [{
    // #define LUT1_NAMED_ASSOCIATED_BUILD
    // template <typename ... Actions>
    // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Actions &&... actions);
  }];
}

def XlnxLut2Op : XlnxLutBase<"lut2"> {
  let arguments = (ins UI4Attr:$INIT, I1:$input0, I1:$input1);
  let assemblyFormat = "`(` `I0` `:` $input0 `,` `I1` `:` $input1 `)` attr-dict `:` type($input0) `,` type($input1) `->` type($result)";
}

def XlnxLut3Op : XlnxLutBase<"lut3"> {
  let arguments = (ins UI8Attr:$INIT, I1:$input0, I1:$input1, I1:$input2);
  let assemblyFormat = "`(` `I0` `:` $input0 `,` `I1` `:` $input1 `,` `I2` `:` $input2 `)` attr-dict `:` type($input0) `,` type($input1) `,` type($input2) `->` type($result)";
}

def XlnxLut4Op : XlnxLutBase<"lut4"> {
  let arguments = (ins UI16Attr:$INIT, I1:$input0, I1:$input1, I1:$input2, I1:$input3);
  let assemblyFormat = "`(` `I0` `:` $input0 `,` `I1` `:` $input1 `,` `I2` `:` $input2 `,` `I3` `:` $input3 `)` attr-dict `:` type($input0) `,` type($input1) `,` type($input2) `,` type($input3) `->` type($result)";
}

def XlnxLut5Op : XlnxLutBase<"lut5"> {
  let arguments = (ins UI32Attr:$INIT, I1:$input0, I1:$input1, I1:$input2, I1:$input3, I1:$input4);
  let assemblyFormat = "`(` `I0` `:` $input0 `,` `I1` `:` $input1 `,` `I2` `:` $input2 `,` `I3` `:` $input3 `,` `I4` `:` $input4 `)` attr-dict `:` type($input0) `,` type($input1) `,` type($input2) `,` type($input3) `,` type($input4) `->` type($result)";
}

def XlnxLut6Op : XlnxLutBase<"lut6"> {
  let arguments = (ins UI64Attr:$INIT, I1:$input0, I1:$input1, I1:$input2, I1:$input3, I1:$input4, I1:$input5);
  let assemblyFormat = "`(` `I0` `:` $input0 `,` `I1` `:` $input1 `,` `I2` `:` $input2 `,` `I3` `:` $input3 `,` `I4` `:` $input4 `,` `I5` `:` $input5 `)` attr-dict `:` type($input0) `,` type($input1) `,` type($input2) `,` type($input3) `,` type($input4) `,` type($input5) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// MUX Operation
//===----------------------------------------------------------------------===//

// Base class for MUX operations
class XlnxMuxBase<string mnemonic, list<Trait> traits = []> : XlnxOp<mnemonic, !listconcat(traits, [])> {
  let summary = "Multiplexing operation";
  let arguments = (ins I1:$select, I1:$input0, I1:$input1);
  let results = (outs I1:$result);
  let assemblyFormat = "`(` `S` `:` $select `,` `I0` `:` $input0 `,` `I1` `:` $input1 `)` attr-dict `:` type($select) `,` type($input0) `,` type($input1) `->` type($result)";
  let hasVerifier = 1;
  let description = [{
    Multiplexing operation with a single output bit. The SELECT pin is the selection input,
    which determines which input is passed through to the output.

    Example:
    ```mlir
    %result = xlnx.mux(S : %select, I0 : %input0, I1 : %input1) : (i1, i1, i1) -> i1
    ```
  }];
}

def XlnxMuxF7Op : XlnxMuxBase<"muxf7"> {
  let description = [{
    CLB MUX to connect two LUT6's Together.

    This design element is a two input multiplexer which, in combination with two LUT6 elements
    will let you create any 7-input logic function, an 8-to-1 multiplexer, or other logic
    functions up to 13-bits wide all within a single CLB. Outputs of the LUT6 elements are
    connected to the I0 and I1 inputs of the MUXF7. The S input is driven from any net. When Low,
    S selects I0. When High, S selects I1.
  }];
}

def XlnxMuxF8Op : XlnxMuxBase<"muxf8"> {
  let description = [{
    CLB MUX to connect two MUXF7's Together.

    This design element is a two input multiplexer which, in combination with two MUXF7 and four
    LUT6 elements will let you create any 8-input logic function, an 16-to-1 multiplexer, or other
    logic functions up to 27-bits wide all within a single CLB. Outputs of the MUXF7 elements are
    connected to the I0 and I1 inputs of the MUXF8. The S input is driven from any net. When Low,
    S selects I0. When High, S selects I1.
  }];
}

def XlnxMuxF9Op : XlnxMuxBase<"muxf9"> {
  let description = [{
    CLB MUX to connect two MUXF8's Together.

    This design element is a two input multiplexer which, in combination with two MUXF8s, four
    MUXF7s and eight LUT6 elements will let you create any 9-input logic function, a 32-to-1
    multiplexer, or other logic functions up to 55-bits wide all within a single CLB. Outputs of
    the MUXF8 elements are connected to the I0 and I1 inputs of the MUXF9. The S input is driven
    from any net. When Low, S selects I0. When High, S selects I1.
  }];
}

//===----------------------------------------------------------------------===//
// Flip-Flop Operations
//===----------------------------------------------------------------------===//

// Base class for Xilinx D Flip-Flops with Clock Enable and Control
class XlnxFDCtrlBase<string mnemonic, list<Trait> traits = []>
    : XlnxOp<mnemonic, !listconcat(traits, [
        Pure,
        Clocked,
        ClockEnabled
      ])> {
  let summary = "Base class for Xilinx D Flip-Flops with Clock Enable and Control";
  let description = [{
    Base class for Xilinx D-type flip-flop primitives with clock enable (CE)
    and an additional control input (e.g., clear, preset, reset, set).

    Common Features:
    - Clock enable (CE) controls data capture
    - Programmable inversion on C and D pins
    - Initialization via GSR or configuration
  }];

  let results = (outs I1:$dataOutput); // Q

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FDCE Operation
//===----------------------------------------------------------------------===//

def XlnxFDCEOp : XlnxFDCtrlBase<"fdce", [AsynchronousControl]> {
  let summary = "D Flip-Flop with Clock Enable and Asynchronous Clear";
  let description = [{
    Xilinx D-type flip-flop primitive with clock enable and asynchronous clear.

    Features:
    - Asynchronous clear (CLR) has priority over other inputs and clears Q Low
    - Clock enable (CE) controls data capture
    - Programmable inversion on C/CLR/D pins
    - Initialization via GSR or configuration

    Verilog instantiation template:
    ```verilog
    FDCE #(
      .INIT(1'b0), .IS_C_INVERTED(1'b0), .IS_CLR_INVERTED(1'b0), .IS_D_INVERTED(1'b0)
    ) FDCE_inst (
      .Q(Q), .C(C), .CE(CE), .CLR(CLR), .D(D)
    );
    ```
  }];

  let arguments = (ins
    ClockType:$clock,
    I1:$clockEnable,
    I1:$asyncClear, // CLR
    I1:$dataInput,

    // Configuration Attributes
    DefaultValuedAttr<UI1Attr, "0">:$INIT, // Default for FDCE is 0
    DefaultValuedAttr<UI1Attr, "0">:$IS_C_INVERTED, // Inherited from base
    DefaultValuedAttr<UI1Attr, "0">:$IS_D_INVERTED, // Inherited from base
    DefaultValuedAttr<UI1Attr, "0">:$IS_CLR_INVERTED
  );

  let assemblyFormat = [{
    `(` `C` `:` $clock `,` `CE` `:` $clockEnable `,` `CLR` `:` $asyncClear `,` `D` `:` $dataInput `)`
    attr-dict
    `:` type($clock) `,` type($clockEnable) `,` type($asyncClear) `,` type($dataInput) `->` type($dataOutput)
  }];

  let extraClassDeclaration = [{
    bool isAsynchronous() const { return true; }
  }];
}

//===----------------------------------------------------------------------===//
// FDPE Operation
//===----------------------------------------------------------------------===//

def XlnxFDPEOp : XlnxFDCtrlBase<"fdpe", [AsynchronousControl]> {
  let summary = "D Flip-Flop with Clock Enable and Asynchronous Preset";
  let description = [{
    Xilinx D-type flip-flop primitive with clock enable and asynchronous preset.

    Features:
    - Asynchronous preset (PRE) has priority over other inputs and presets Q High
    - Clock enable (CE) controls data capture
    - Programmable inversion on C/PRE/D pins
    - Initialization via GSR or configuration

    Verilog instantiation template:
    ```verilog
    FDPE #(
      .INIT(1'b1), .IS_C_INVERTED(1'b0), .IS_D_INVERTED(1'b0), .IS_PRE_INVERTED(1'b0)
    ) FDPE_inst (
      .Q(Q), .C(C), .CE(CE), .D(D), .PRE(PRE)
    );
    ```
  }];

  let arguments = (ins
    ClockType:$clock,
    I1:$clockEnable,
    I1:$asyncPreset,  // PRE
    I1:$dataInput,

    // Configuration Attributes
    DefaultValuedAttr<UI1Attr, "1">:$INIT, // Default for FDPE is 1
    DefaultValuedAttr<UI1Attr, "0">:$IS_C_INVERTED, // Inherited from base
    DefaultValuedAttr<UI1Attr, "0">:$IS_D_INVERTED, // Inherited from base
    DefaultValuedAttr<UI1Attr, "0">:$IS_PRE_INVERTED
  );


  let assemblyFormat = [{
    `(` `C` `:` $clock `,` `CE` `:` $clockEnable `,` `PRE` `:` $asyncPreset `,` `D` `:` $dataInput `)`
    attr-dict
    `:` type($clock) `,` type($clockEnable) `,` type($asyncPreset) `,` type($dataInput) `->` type($dataOutput)
  }];


  let extraClassDeclaration = [{
    bool isAsynchronous() const { return true; }
  }];
}

//===----------------------------------------------------------------------===//
// FDRE Operation
//===----------------------------------------------------------------------===//

def XlnxFDREOp : XlnxFDCtrlBase<"fdre", [SynchronousControl]> {
  let summary = "D Flip-Flop with Clock Enable and Synchronous Reset";
  let description = [{
    Xilinx D-type flip-flop primitive with clock enable and synchronous reset.

    Features:
    - Synchronous reset (R) overrides other inputs on the clock edge and resets Q Low
    - Clock enable (CE) controls data capture
    - Programmable inversion on C/R/D pins
    - Initialization via GSR or configuration

    Verilog instantiation template:
    ```verilog
    FDRE #(
      .INIT(1'b0), .IS_C_INVERTED(1'b0), .IS_D_INVERTED(1'b0), .IS_R_INVERTED(1'b0)
    ) FDRE_inst (
      .Q(Q), .C(C), .CE(CE), .D(D), .R(R)
    );
    ```
  }];

  let arguments = (ins
    ClockType:$clock,
    I1:$clockEnable,
    I1:$syncReset,    // R
    I1:$dataInput,

    // Configuration Attributes
    DefaultValuedAttr<UI1Attr, "0">:$INIT, // Default for FDRE is 0
    DefaultValuedAttr<UI1Attr, "0">:$IS_C_INVERTED, // Inherited from base
    DefaultValuedAttr<UI1Attr, "0">:$IS_D_INVERTED, // Inherited from base
    DefaultValuedAttr<UI1Attr, "0">:$IS_R_INVERTED
  );

  let assemblyFormat = [{
    `(` `C` `:` $clock `,` `CE` `:` $clockEnable `,` `R` `:` $syncReset `,` `D` `:` $dataInput `)`
    attr-dict
    `:` type($clock) `,` type($clockEnable) `,` type($syncReset) `,` type($dataInput) `->` type($dataOutput)
  }];

  let extraClassDeclaration = [{
    bool isSynchronous() const { return true; }
  }];
}

//===----------------------------------------------------------------------===//
// FDSE Operation
//===----------------------------------------------------------------------===//

def XlnxFDSEOp : XlnxFDCtrlBase<"fdse", [SynchronousControl]> {
  let summary = "D Flip-Flop with Clock Enable and Synchronous Set";
  let description = [{
    Xilinx D-type flip-flop primitive with clock enable and synchronous set.

    Features:
    - Synchronous set (S) overrides other inputs on the clock edge and sets Q High
    - Clock enable (CE) controls data capture
    - Programmable inversion on C/S/D pins
    - Initialization via GSR or configuration

    Verilog instantiation template:
    ```verilog
    FDSE #(
      .INIT(1'b1), .IS_C_INVERTED(1'b0), .IS_D_INVERTED(1'b0), .IS_S_INVERTED(1'b0)
    ) FDSE_inst (
      .Q(Q), .C(C), .CE(CE), .D(D), .S(S)
    );
    ```
  }];

  let arguments = (ins
    ClockType:$clock,
    I1:$clockEnable,
    I1:$syncSet,      // S
    I1:$dataInput,

    // Configuration Attributes
    DefaultValuedAttr<UI1Attr, "1">:$INIT, // Default for FDSE is 1
    DefaultValuedAttr<UI1Attr, "0">:$IS_C_INVERTED, // Inherited from base
    DefaultValuedAttr<UI1Attr, "0">:$IS_D_INVERTED, // Inherited from base
    DefaultValuedAttr<UI1Attr, "0">:$IS_S_INVERTED
  );

  let assemblyFormat = [{
    `(` `C` `:` $clock `,` `CE` `:` $clockEnable `,` `S` `:` $syncSet `,` `D` `:` $dataInput `)`
    attr-dict
    `:` type($clock) `,` type($clockEnable) `,` type($syncSet) `,` type($dataInput) `->` type($dataOutput)
  }];

  let extraClassDeclaration = [{
    bool isSynchronous() const { return true; }
  }];
}

