# Xlnx Dialect Design Rationale

This document describes various design points of the Xlnx dialect, why they are designed the way they are, and the current status. This follows the spirit of the other [MLIR Design Rationale documents](https://mlir.llvm.org/docs/Rationale/).

- [Xlnx Dialect Design Rationale](#xlnx Dialect Design Rationale)
- [Introduction](#Introduction)
- [Dialect Scope](#Dialect Scope)
- [Operation Design](#Operation Design)
- [LUT Operation Hierarchy](#LUT Operation Hierarchy)
- [Generic LUT and Specific LUT](#Generic LUT and Specific LUT)
- [Verifier Design](#Verifier Design)
- [Future Development Directions](#Future Development Directions)
- [FAQ](#FAQ)

## Introduction

Xilinx FPGAs are widely used programmable logic devices with rich hardware primitives and architectural features. To allow the CIRCT toolchain to fully exploit these specific hardware resources, we designed the Xlnx dialect, which provides direct representation and manipulation of Xilinx specific hardware primitives.

The initial design focuses on the UltraScale+ architecture family, which is a mainstream FPGA architecture from Xilinx. As the dialect evolves, we plan to expand support for more Xilinx device families.

## Dialect Scope

The goal of the Xlnx dialect is to provide a low-level representation of Xilinx FPGA-specific primitives and features, rather than fully emulating the full functionality of the Xilinx toolchain. The dialect focuses on hardware primitives that are difficult to express or optimize in other CIRCT dialects.

Currently, the dialect supports the following primitives:

1.  **Lookup Table (LUT)** primitives (`xlnx.lutn`, `xlnx.lut1`-`lut6`), the core combinatorial logic elements.
2.  **Multiplexer (Mux)** primitives (`xlnx.muxf7`, `xlnx.muxf8`, `xlnx.muxf9`), representing hardened multiplexers within the CLB.
3.  **Flip-Flop (FF)** primitives (`xlnx.fdce`), representing basic sequential elements like D-type flip-flops with clock enable and asynchronous clear.

In the future, the dialect will be expanded to include:

1. Block RAM and distributed RAM primitives
2. DSP primitives
3. Clock management primitives
4. I/O primitives
5. Other Xilinx-specific hardware resources

## Operation Design

### LUT Operation Hierarchy

To provide clear levels of abstraction and flexibility, we designed a hierarchy of LUT operations:

1. A base class (`XlnxLutBase`) provides common semantics and validator logic for all LUT operations
2. A generic LUT operation (`xlnx.lutn`) that accepts a variable number of inputs
3. A series of specific LUT operations (`xlnx.lut1` to `xlnx.lut6`), each corresponding to a specific number of inputs

This design allows users to choose the level of abstraction that best suits their usage scenario:

- Generic `lutn` provides concise syntax and flexibility
- Specific `lut1` to `lut6` provide clear input tagging and stricter type checking

### Generic LUT and Specific LUT

We provide two styles of LUT operations to meet different usage needs:

1. **Generic LUT (xlnx.lutn)**

- Accepts variable number of inputs (1-6)
- Uses concise syntax
- Good for dynamically generated code or when flexibility is desired

2. **Specific LUTs (xlnx.lut1-xlnx.lut6)**
- Accepts fixed number of inputs per operation
- Uses explicit input tags (I0-I5)
- Provides one-to-one mapping with Xilinx hardware
- Good for when explicit control of input order is desired

These two styles are equivalent and users can choose which to use based on their preference.

## Validator Design

To ensure that the generated code is compatible with Xilinx hardware, we implemented comprehensive validation rules:

1. **Input Quantity Validation**: Ensure that the number of LUT inputs is between 1 and 6
2. **INIT Value Range Validation**: Ensure that the INIT attribute does not exceed the maximum value allowed by the number of inputs
3. **Type Validation**: Ensure that all inputs and outputs are of Boolean type (i1)

The validation logic is implemented using templates so that the same validation rules are shared between all LUT operations.

## Future Development Directions

The ongoing development plan for the Xlnx dialect includes:

1.  **Continued primitive support extension**: Building upon the initial LUT, Mux, and FF support, add representations for more Xilinx hardware primitives such as:
    *   Block RAM and distributed RAM primitives
    *   DSP primitives
    *   Other Xilinx-specific hardware resources
2.  **Architecture features**: Implement architecture-specific features and constraints
3.  **Optimization**: Provide specific optimization for Xilinx FPGAs
4.  **Integration with other dialects**: Improve integration with Comb, SV and other dialects
5.  **Expression conversion**: Automatically generate the best LUT representation from high-level expressions

## FAQ

**Q: Why has the INIT attribute design been changed from a uniform 64-bit integer to specific bit widths for each LUT type?**

A: The updated design uses the exact bit width required for each LUT size (2^N bits for an N-input LUT), which provides the following benefits:
1. More precise type checking and error detection at the dialect level
2. Better representation of the actual hardware constraints
3. More efficient memory usage and processing within the compiler
4. Clearer documentation and user interface

The generic `xlnx.lutn` operation still uses a 64-bit integer for backward compatibility and flexibility when the number of inputs is determined at runtime.

**Q: Why don't you use an array or a more intuitive truth table representation instead of an integer INIT value?**

A: The integer INIT value directly corresponds to how the Xilinx toolchain and hardware are represented, making integration with existing tools more seamless. In addition, the integer representation is more efficient in terms of internal processing and storage.

**Q: Why is there no routing or placement information included in the dialect?**

A: The Xlnx dialect currently focuses on logical representation, not physical implementation details. Routing and placement information is typically handled by downstream Xilinx tools, or may be added as a separate abstraction layer in future extensions.