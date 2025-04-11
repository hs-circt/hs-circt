# Xlnx Dialect

This dialect contains specific operations for Xilinx FPGAs, with the current version only supporting UltraScale+ series devices.

## Introduction

The Xlnx dialect provides a set of operations for representing and manipulating specific hardware primitives in Xilinx FPGAs. The initial implementation focuses on lookup table (LUT) primitives, which are the most basic combinational logic units in Xilinx FPGA architecture.

The Xlnx dialect provides two styles of LUT operations:
1. General `xlnx.lutn` operations, supporting a variable number of inputs from 1 to 6
2. Specialized `xlnx.lut1` to `xlnx.lut6` operations, with a fixed number of inputs and a more explicit interface syntax

## List of Operations

### Lookup Table (LUT) Operations

```mlir
%result = xlnx.lutn(%input1, %input2, ..., %inputN) {INIT = value : ui64} : (i1, i1, ..., i1) -> i1
```

Where:
- `%input1` to `%inputN` are 1 to 6 Boolean (i1) inputs
- `INIT` is a 64-bit unsigned integer property that defines the truth table of the LUT
- The operation returns a Boolean (i1) result

### Specific LUTs (xlnx.lut1 to xlnx.lut6)

```mlir
// LUT1 (1 input)
%result = xlnx.lut1(I0: %input) {INIT = value : ui2} : i1 -> i1

// LUT2 (2 inputs)
%result = xlnx.lut2(I0: %input0, I1: %input1) {INIT = value : ui4} : i1, i1 -> i1

// LUT3 (3 inputs)
%result = xlnx.lut3(I0: %input0, I1: %input1, I2: %input2) {INIT = value : ui8} : i1, i1, i1 -> i1

// LUT4 (4 inputs)
%result = xlnx.lut4(I0: %input0, I1: %input1, I2: %input2, I3: %input3) {INIT = value : ui16} : i1, i1, i1, i1 -> i1

// LUT5 (5 inputs)
%result = xlnx.lut5(I0: %input0, I1: %input1, I2: %input2, I3: %input3, I4: %input4) {INIT = value : ui32} : i1, i1, i1, i1, i1 -> i1

// LUT6 (6 inputs)
%result = xlnx.lut6(I0: %input0, I1: %input1, I2: %input2, I3: %input3, I4: %input4, I5: %input5) {INIT = value : ui64} : i1, i1, i1, i1, i1, i1 -> i1
```

Where:
- `I0` through `I5` are input location labels that correspond to specific input pins of the Xilinx LUT cell
- Each LUT operation accepts a fixed number of Boolean (i1) inputs and returns a Boolean (i1) result
- `INIT` is an unsigned integer property that defines the truth table of the LUT, with the bit width matching the requirements for each LUT size (ui2 for LUT1, ui4 for LUT2, etc.)

## INIT Properties

The `INIT` attribute is an unsigned integer that defines the functionality of the LUT. It represents a truth table where each bit corresponds to the output value for a specific input combination. The bit width of the INIT attribute varies based on the number of inputs:

- For 1-input LUT (LUT1): 2-bit INIT (ui2)
- For 2-input LUT (LUT2): 4-bit INIT (ui4)
- For 3-input LUT (LUT3): 8-bit INIT (ui8)
- For 4-input LUT (LUT4): 16-bit INIT (ui16)
- For 5-input LUT (LUT5): 32-bit INIT (ui32)
- For 6-input LUT (LUT6): 64-bit INIT (ui64)
- For generic LUT (lutn): 64-bit INIT (ui64), with only the lower 2^N bits used for an N-input LUT

For an N-input LUT, the INIT attribute uses 2^N bits. The index of each bit corresponds to the binary encoding of the input, following the following rules:
- For the combination of inputs (I0, I1, ..., I(N-1)), the corresponding bit index is: (I(N-1) << (N-1)) | ... | (I1 << 1) | I0

For example, for a 2-input LUT, the 4 bits of the `INIT` attribute are used, mapping as follows:
- INIT[0]: Output when input is (I0=0, I1=0)
- INIT[1]: Output when input is (I0=1, I1=0)
- INIT[2]: Output when input is (I0=0, I1=1)
- INIT[3]: Output when input is (I0=1, I1=1)

### INIT values ​​for common logic functions

Here are some common INIT values ​​for 2-input logic functions:
- AND gate: INIT = 8 (binary 1000) - This is a ui4 value for LUT2
- OR gate: INIT = 14 (binary 1110) - This is a ui4 value for LUT2
- XOR gate: INIT = 6 (binary 0110) - This is a ui4 value for LUT2
- NAND gate: INIT = 7 (binary 0111) - This is a ui4 value for LUT2
- NOR gate: INIT = 1 (binary 0001) - This is a ui4 value for LUT2
- XNOR gate: INIT = 9 (binary 1001) - This is a ui4 value for LUT2

## Example

### Basic LUT Usage

```mlir
// Create an AND gate
%and = xlnx.lut2(I0: %a, I1: %b) {INIT = 8 : ui4} : i1, i1 -> i1

// Create the same AND gate using a generic LUT
%and_generic = xlnx.lutn(%a, %b) {INIT = 8 : ui64} : (i1, i1) -> i1

// Create an XOR gate
%xor = xlnx.lut2(I0: %a, I1: %b) {INIT = 6 : ui4} : i1, i1 -> i1

// Create a 3-input majority voter (output 1 if 2 or more of the inputs are 1)
%majority = xlnx.lut3(I0: %a, I1: %b, I2: %c) {INIT = 232: ui8}: i1, i1, i1 -> i1
```

### LUT Cascading

```mlir
// Create an AND gate
%and = xlnx.lut2(I0: %a, I1: %b) {INIT = 8 : ui4} : i1, i1 -> i1

// Create an OR gate that takes the output of the previous LUT as one of its inputs
%or = xlnx.lut2(I0: %and, I1: %c) {INIT = 14 : ui4} : i1, i1 -> i1
```

## Validation Rules

All LUT operations have the following validation rules:
1. The number of inputs must be between 1 and 6
2. The INIT value cannot exceed the maximum allowed by the number of inputs (for N-input LUTs, the maximum INIT value is 2^(2^N)-1)

## API

LUT operations provide the following common interface:
- Return type is always a single boolean value (i1)
- All inputs must be boolean values ​​(i1)
- All LUT operations record and use the INIT attribute internally
- Validator ensures that the INIT attribute is valid for a given number of inputs