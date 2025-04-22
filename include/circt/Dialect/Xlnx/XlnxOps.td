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
class XlnxOp<string mnemonic, list<Trait> traits = []> : Op<XlnxDialect, mnemonic, !listconcat(traits, [HWInstable])> {
  let extraClassDeclaration = [{
    llvm::StringRef getGateType();
    llvm::StringMap<mlir::Value> getPortDict();
    llvm::StringMap<mlir::Attribute> getAttrDict();
  }];
}

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
  let extraClassDefinition = [{
    llvm::StringRef XlnxLutNOp::getGateType() {
      return "LUT" + std::to_string(getNumOperands());
    }
    llvm::StringMap<mlir::Value> XlnxLutNOp::getPortDict() {
      llvm::StringMap<mlir::Value> ports;
      for (auto it : llvm::enumerate(getInputs())) {
        ports["I" + std::to_string(it.index())] = it.value();
      }
      return ports;
    }
    llvm::StringMap<mlir::Attribute> XlnxLutNOp::getAttrDict() {
      llvm::StringMap<mlir::Attribute> attrs;
      attrs["INIT"] = getINITAttr();
      return attrs;
    }
  }];
}

def XlnxLut1Op : XlnxLutBase<"lut1"> {
  let arguments = (ins UI2Attr:$INIT, I1:$input);
  let assemblyFormat = "`(` `I0` `:` $input `)` attr-dict `:` type($input) `->` type($result)";
  // let extraClassDeclaration = [{
  //   // #define LUT1_NAMED_ASSOCIATED_BUILD
  //   // template <typename ... Actions>
  //   // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Actions &&... actions);
  // }];

  let extraClassDefinition = [{
    llvm::StringRef XlnxLut1Op::getGateType() {
      return "LUT1";
    }
    llvm::StringMap<mlir::Value> XlnxLut1Op::getPortDict() {
      llvm::StringMap<mlir::Value> ports;
      ports["I0"] = getInput();
      return ports;
    }
    llvm::StringMap<mlir::Attribute> XlnxLut1Op::getAttrDict() {
      llvm::StringMap<mlir::Attribute> attrs;
      attrs["INIT"] = getINITAttr();
      return attrs;
    }
  }];
}

def XlnxLut2Op : XlnxLutBase<"lut2"> {
  let arguments = (ins UI4Attr:$INIT, I1:$input0, I1:$input1);
  let assemblyFormat = "`(` `I0` `:` $input0 `,` `I1` `:` $input1 `)` attr-dict `:` type($input0) `,` type($input1) `->` type($result)";
  let extraClassDefinition = [{
    llvm::StringRef XlnxLut2Op::getGateType() {
      return "LUT2";
    }
    llvm::StringMap<mlir::Value> XlnxLut2Op::getPortDict() {
      llvm::StringMap<mlir::Value> ports;
      ports["I0"] = getInput0();
      ports["I1"] = getInput1();
      return ports;
    }
    llvm::StringMap<mlir::Attribute> XlnxLut2Op::getAttrDict() {
      llvm::StringMap<mlir::Attribute> attrs;
      attrs["INIT"] = getINITAttr();
      return attrs;
    }
  }];
}

def XlnxLut3Op : XlnxLutBase<"lut3"> {
  let arguments = (ins UI8Attr:$INIT, I1:$input0, I1:$input1, I1:$input2);
  let assemblyFormat = "`(` `I0` `:` $input0 `,` `I1` `:` $input1 `,` `I2` `:` $input2 `)` attr-dict `:` type($input0) `,` type($input1) `,` type($input2) `->` type($result)";
  let extraClassDefinition = [{
    llvm::StringRef XlnxLut3Op::getGateType() {
      return "LUT3";
    }
    llvm::StringMap<mlir::Value> XlnxLut3Op::getPortDict() {
      llvm::StringMap<mlir::Value> ports;
      ports["I0"] = getInput0();
      ports["I1"] = getInput1();
      ports["I2"] = getInput2();
      return ports;
    }
    llvm::StringMap<mlir::Attribute> XlnxLut3Op::getAttrDict() {
      llvm::StringMap<mlir::Attribute> attrs;
      attrs["INIT"] = getINITAttr();
      return attrs;
    }
  }];
}

def XlnxLut4Op : XlnxLutBase<"lut4"> {
  let arguments = (ins UI16Attr:$INIT, I1:$input0, I1:$input1, I1:$input2, I1:$input3);
  let assemblyFormat = "`(` `I0` `:` $input0 `,` `I1` `:` $input1 `,` `I2` `:` $input2 `,` `I3` `:` $input3 `)` attr-dict `:` type($input0) `,` type($input1) `,` type($input2) `,` type($input3) `->` type($result)";
  let extraClassDefinition = [{
    llvm::StringRef XlnxLut4Op::getGateType() {
      return "LUT4";
    }
    llvm::StringMap<mlir::Value> XlnxLut4Op::getPortDict() {
      llvm::StringMap<mlir::Value> ports;
      ports["I0"] = getInput0();
      ports["I1"] = getInput1();
      ports["I2"] = getInput2();
      ports["I3"] = getInput3();
      return ports;
    }
    llvm::StringMap<mlir::Attribute> XlnxLut4Op::getAttrDict() {
      llvm::StringMap<mlir::Attribute> attrs;
      attrs["INIT"] = getINITAttr();
      return attrs;
    }
  }];
}

def XlnxLut5Op : XlnxLutBase<"lut5"> {
  let arguments = (ins UI32Attr:$INIT, I1:$input0, I1:$input1, I1:$input2, I1:$input3, I1:$input4);
  let assemblyFormat = "`(` `I0` `:` $input0 `,` `I1` `:` $input1 `,` `I2` `:` $input2 `,` `I3` `:` $input3 `,` `I4` `:` $input4 `)` attr-dict `:` type($input0) `,` type($input1) `,` type($input2) `,` type($input3) `,` type($input4) `->` type($result)";
  let extraClassDefinition = [{
    llvm::StringRef XlnxLut5Op::getGateType() {
      return "LUT5";
    }
    llvm::StringMap<mlir::Value> XlnxLut5Op::getPortDict() {
      llvm::StringMap<mlir::Value> ports;
      ports["I0"] = getInput0();
      ports["I1"] = getInput1();
      ports["I2"] = getInput2();
      ports["I3"] = getInput3();
      ports["I4"] = getInput4();
      return ports;
    }
    llvm::StringMap<mlir::Attribute> XlnxLut5Op::getAttrDict() {
      llvm::StringMap<mlir::Attribute> attrs;
      attrs["INIT"] = getINITAttr();
      return attrs;
    }
  }];
}

def XlnxLut6Op : XlnxLutBase<"lut6"> {
  let arguments = (ins UI64Attr:$INIT, I1:$input0, I1:$input1, I1:$input2, I1:$input3, I1:$input4, I1:$input5);
  let assemblyFormat = "`(` `I0` `:` $input0 `,` `I1` `:` $input1 `,` `I2` `:` $input2 `,` `I3` `:` $input3 `,` `I4` `:` $input4 `,` `I5` `:` $input5 `)` attr-dict `:` type($input0) `,` type($input1) `,` type($input2) `,` type($input3) `,` type($input4) `,` type($input5) `->` type($result)";
  let extraClassDefinition = [{
    llvm::StringRef XlnxLut6Op::getGateType() {
      return "LUT6";
    }
    llvm::StringMap<mlir::Value> XlnxLut6Op::getPortDict() {
      llvm::StringMap<mlir::Value> ports;
      ports["I0"] = getInput0();
      ports["I1"] = getInput1();
      ports["I2"] = getInput2();
      ports["I3"] = getInput3();
      ports["I4"] = getInput4();
      ports["I5"] = getInput5();
      return ports;
    }
    llvm::StringMap<mlir::Attribute> XlnxLut6Op::getAttrDict() {
      llvm::StringMap<mlir::Attribute> attrs;
      attrs["INIT"] = getINITAttr();
      return attrs;
    }
  }];
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

  let extraClassDefinition = [{
    llvm::StringRef XlnxMuxF7Op::getGateType() {
      return "MUXF7";
    }
    llvm::StringMap<mlir::Value> XlnxMuxF7Op::getPortDict() {
      llvm::StringMap<mlir::Value> ports;
      ports["S"] = getSelect();
      ports["I0"] = getInput0();
      ports["I1"] = getInput1();
      return ports;
    }
    llvm::StringMap<mlir::Attribute> XlnxMuxF7Op::getAttrDict() {
      llvm::StringMap<mlir::Attribute> attrs;
      return attrs;
    }
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

  let extraClassDefinition = [{
    llvm::StringRef XlnxMuxF8Op::getGateType() {
      return "MUXF8";
    }
    llvm::StringMap<mlir::Value> XlnxMuxF8Op::getPortDict() {
      llvm::StringMap<mlir::Value> ports;
      ports["S"] = getSelect();
      ports["I0"] = getInput0();
      ports["I1"] = getInput1();
      return ports;
    }
    llvm::StringMap<mlir::Attribute> XlnxMuxF8Op::getAttrDict() {
      llvm::StringMap<mlir::Attribute> attrs;
      return attrs;
    }
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

  let extraClassDefinition = [{
    llvm::StringRef XlnxMuxF9Op::getGateType() {
      return "MUXF9";
    }
    llvm::StringMap<mlir::Value> XlnxMuxF9Op::getPortDict() {
      llvm::StringMap<mlir::Value> ports;
      ports["S"] = getSelect();
      ports["I0"] = getInput0();
      ports["I1"] = getInput1();
      return ports;
    }
    llvm::StringMap<mlir::Attribute> XlnxMuxF9Op::getAttrDict() {
      llvm::StringMap<mlir::Attribute> attrs;
      return attrs;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Flip-Flop Operations
//===----------------------------------------------------------------------===//

// Base class for Xilinx D Flip-Flops with Clock Enable and Control
class XlnxFDCtrlBase<string mnemonic, list<Trait> traits = []>
    : XlnxOp<mnemonic, !listconcat(traits, [
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
    // Ports
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

  let extraClassDefinition = [{
    llvm::StringRef XlnxFDCEOp::getGateType() {
      return "FDCE";
    }
    llvm::StringMap<mlir::Value> XlnxFDCEOp::getPortDict() {
      llvm::StringMap<mlir::Value> ports;
      ports["C"] = getClock();
      ports["CE"] = getClockEnable();
      ports["CLR"] = getAsyncClear();
      ports["D"] = getDataInput();
      return ports;
    }
    llvm::StringMap<mlir::Attribute> XlnxFDCEOp::getAttrDict() {
      llvm::StringMap<mlir::Attribute> attrs;
      attrs["INIT"] = getINITAttr();
      attrs["IS_C_INVERTED"] = getIS_C_INVERTEDAttr();
      attrs["IS_D_INVERTED"] = getIS_D_INVERTEDAttr();
      attrs["IS_CLR_INVERTED"] = getIS_CLR_INVERTEDAttr();
      return attrs;
    }
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
    // Ports
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

  let extraClassDefinition = [{
    llvm::StringRef XlnxFDPEOp::getGateType() {
      return "FDPE";
    }
    llvm::StringMap<mlir::Value> XlnxFDPEOp::getPortDict() {
      llvm::StringMap<mlir::Value> ports;
      ports["C"] = getClock();
      ports["CE"] = getClockEnable();
      ports["PRE"] = getAsyncPreset();
      ports["D"] = getDataInput();
      return ports;
    }
    llvm::StringMap<mlir::Attribute> XlnxFDPEOp::getAttrDict() {
      llvm::StringMap<mlir::Attribute> attrs;
      attrs["INIT"] = getINITAttr();
      attrs["IS_C_INVERTED"] = getIS_C_INVERTEDAttr();
      attrs["IS_D_INVERTED"] = getIS_D_INVERTEDAttr();
      attrs["IS_PRE_INVERTED"] = getIS_PRE_INVERTEDAttr();
      return attrs;
    }
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
    // Ports
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

  let extraClassDefinition = [{
    llvm::StringRef XlnxFDREOp::getGateType() {
      return "FDRE";
    }
    llvm::StringMap<mlir::Value> XlnxFDREOp::getPortDict() {
      llvm::StringMap<mlir::Value> ports;
      ports["C"] = getClock();
      ports["CE"] = getClockEnable();
      ports["R"] = getSyncReset();
      ports["D"] = getDataInput();
      return ports;
    }
    llvm::StringMap<mlir::Attribute> XlnxFDREOp::getAttrDict() {
      llvm::StringMap<mlir::Attribute> attrs;
      attrs["INIT"] = getINITAttr();
      attrs["IS_C_INVERTED"] = getIS_C_INVERTEDAttr();
      attrs["IS_D_INVERTED"] = getIS_D_INVERTEDAttr();
      attrs["IS_R_INVERTED"] = getIS_R_INVERTEDAttr();
      return attrs;
    }
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
    // Ports
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

  let extraClassDefinition = [{
    llvm::StringRef XlnxFDSEOp::getGateType() {
      return "FDSE";
    }
    llvm::StringMap<mlir::Value> XlnxFDSEOp::getPortDict() {
      llvm::StringMap<mlir::Value> ports;
      ports["C"] = getClock();
      ports["CE"] = getClockEnable();
      ports["S"] = getSyncSet();
      ports["D"] = getDataInput();
      return ports;
    }
    llvm::StringMap<mlir::Attribute> XlnxFDSEOp::getAttrDict() {
      llvm::StringMap<mlir::Attribute> attrs;
      attrs["INIT"] = getINITAttr();
      attrs["IS_C_INVERTED"] = getIS_C_INVERTEDAttr();
      attrs["IS_D_INVERTED"] = getIS_D_INVERTEDAttr();
      attrs["IS_S_INVERTED"] = getIS_S_INVERTEDAttr();
      return attrs;
    }
  }];
}

