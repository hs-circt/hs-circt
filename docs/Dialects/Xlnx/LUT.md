# Xlnx Dialect - Lookup Table Primitives

This document describes the Lookup Table (LUT) primitive operations available in the Xlnx dialect, with the current version supporting UltraScale+ series devices.

## Introduction

The Lookup Table (LUT) is the fundamental combinational logic building block in Xilinx FPGAs. In UltraScale+ devices, LUTs are 6-input elements capable of implementing any arbitrary 6-input boolean function. These LUTs are physically implemented within Configurable Logic Blocks (CLBs) and serve as the primary resource for implementing combinational logic.

The Xlnx dialect provides two styles of LUT operations:
1. General `xlnx.lutn` operations, supporting a variable number of inputs from 1 to 6
2. Specialized `xlnx.lut1` to `xlnx.lut6` operations, with a fixed number of inputs and a more explicit interface syntax

## Operation Definitions

### General Lookup Table Operation

```mlir
%result = xlnx.lutn(%input1, %input2, ..., %inputN) {INIT = value : ui64} : (i1, i1, ..., i1) -> i1
```

**Operands:**
- `%input1` to `%inputN`: 1 to 6 Boolean (i1) inputs
- `INIT`: A 64-bit unsigned integer attribute that defines the truth table of the LUT

**Result:**
- `%result`: A Boolean (i1) output representing the result of the logic function

### Specific LUT Operations

#### `xlnx.lut1` (1-input LUT)

```mlir
%result = xlnx.lut1(I0: %input) {INIT = value : ui2} : i1 -> i1
```

**Operands:**
- `I0` (%input): Boolean input (i1)
- `INIT`: A 2-bit unsigned integer attribute (ui2) that defines the truth table

**Result:**
- `%result`: Boolean output (i1)

#### `xlnx.lut2` (2-input LUT)

```mlir
%result = xlnx.lut2(I0: %input0, I1: %input1) {INIT = value : ui4} : i1, i1 -> i1
```

**Operands:**
- `I0` (%input0): First Boolean input (i1)
- `I1` (%input1): Second Boolean input (i1)
- `INIT`: A 4-bit unsigned integer attribute (ui4) that defines the truth table

**Result:**
- `%result`: Boolean output (i1)

#### `xlnx.lut3` (3-input LUT)

```mlir
%result = xlnx.lut3(I0: %input0, I1: %input1, I2: %input2) {INIT = value : ui8} : i1, i1, i1 -> i1
```

**Operands:**
- `I0` to `I2`: Boolean inputs (i1)
- `INIT`: An 8-bit unsigned integer attribute (ui8) that defines the truth table

**Result:**
- `%result`: Boolean output (i1)

#### `xlnx.lut4` (4-input LUT)

```mlir
%result = xlnx.lut4(I0: %input0, I1: %input1, I2: %input2, I3: %input3) {INIT = value : ui16} : i1, i1, i1, i1 -> i1
```

**Operands:**
- `I0` to `I3`: Boolean inputs (i1)
- `INIT`: A 16-bit unsigned integer attribute (ui16) that defines the truth table

**Result:**
- `%result`: Boolean output (i1)

#### `xlnx.lut5` (5-input LUT)

```mlir
%result = xlnx.lut5(I0: %input0, I1: %input1, I2: %input2, I3: %input3, I4: %input4) {INIT = value : ui32} : i1, i1, i1, i1, i1 -> i1
```

**Operands:**
- `I0` to `I4`: Boolean inputs (i1)
- `INIT`: A 32-bit unsigned integer attribute (ui32) that defines the truth table

**Result:**
- `%result`: Boolean output (i1)

#### `xlnx.lut6` (6-input LUT)

```mlir
%result = xlnx.lut6(I0: %input0, I1: %input1, I2: %input2, I3: %input3, I4: %input4, I5: %input5) {INIT = value : ui64} : i1, i1, i1, i1, i1, i1 -> i1
```

**Operands:**
- `I0` to `I5`: Boolean inputs (i1)
- `INIT`: A 64-bit unsigned integer attribute (ui64) that defines the truth table

**Result:**
- `%result`: Boolean output (i1)

## INIT Attribute Specification

The `INIT` attribute is a crucial parameter that defines the functionality of the LUT. It represents a truth table where each bit corresponds to the output value for a specific input combination. The bit width of the INIT attribute varies based on the number of inputs:

| LUT Type | Inputs | INIT Bit Width | INIT Type | Max Value                           |
|----------|--------|----------------|-----------|-------------------------------------|
| LUT1     | 1      | 2              | ui2       | 3                                   |
| LUT2     | 2      | 4              | ui4       | 15                                  |
| LUT3     | 3      | 8              | ui8       | 255                                 |
| LUT4     | 4      | 16             | ui16      | 65,535                              |
| LUT5     | 5      | 32             | ui32      | 4,294,967,295                       |
| LUT6     | 6      | 64             | ui64      | 18,446,744,073,709,551,615          |
| LUTN     | N      | 64             | ui64      | Same as above (lower 2^N bits used) |

For an N-input LUT, the INIT attribute uses 2^N bits. The index of each bit corresponds to the binary encoding of the input, following these rules:
- For the combination of inputs (I0, I1, ..., I(N-1)), the corresponding bit index is: (I(N-1) << (N-1)) | ... | (I1 << 1) | I0

For example, for a 2-input LUT, the 4 bits of the `INIT` attribute are used, mapping as follows:
- INIT[0]: Output when input is (I0=0, I1=0)
- INIT[1]: Output when input is (I0=1, I1=0)
- INIT[2]: Output when input is (I0=0, I1=1)
- INIT[3]: Output when input is (I0=1, I1=1)

## Examples

### Basic Logic Gates

```mlir
// Create an AND gate
%and = xlnx.lut2(I0: %a, I1: %b) {INIT = 8 : ui4} : i1, i1 -> i1

// Create an OR gate
%or = xlnx.lut2(I0: %a, I1: %b) {INIT = 14 : ui4} : i1, i1 -> i1

// Create an XOR gate
%xor = xlnx.lut2(I0: %a, I1: %b) {INIT = 6 : ui4} : i1, i1 -> i1

// Create a NAND gate
%nand = xlnx.lut2(I0: %a, I1: %b) {INIT = 7 : ui4} : i1, i1 -> i1
```

### Complex Functions

```mlir
// Create a 3-input majority voter (output 1 if 2 or more inputs are 1)
// INIT = 11101000 binary = 232 decimal
%majority = xlnx.lut3(I0: %a, I1: %b, I2: %c) {INIT = 232 : ui8} : i1, i1, i1 -> i1

// Create a 4-input function that outputs 1 if exactly 2 inputs are 1
// INIT pattern needs to match this specific condition
%exactly_two = xlnx.lut4(I0: %a, I1: %b, I2: %c, I3: %d) {INIT = 5736 : ui16} : i1, i1, i1, i1 -> i1
```

### LUT Cascading

```mlir
// Create an AND gate
%and = xlnx.lut2(I0: %a, I1: %b) {INIT = 8 : ui4} : i1, i1 -> i1

// Create an OR gate that takes the output of the previous LUT as one of its inputs
%or = xlnx.lut2(I0: %and, I1: %c) {INIT = 14 : ui4} : i1, i1 -> i1
```

## Hardware Implementation

In UltraScale+ devices, each LUT is physically implemented as a 6-input LUT (LUT6) that can be configured as:
- One 6-input LUT
- Two 5-input LUTs with shared inputs
- Split into even smaller LUTs in certain configurations

The physical LUT structure is based on a memory array of 64 bits (2^6 = 64 entries) that can be addressed by the 6 inputs. Each LUT6 primitive has one general output (represented by the operation's result) and can also have additional outputs when configured in dual-output modes.

## Validation Rules

All LUT operations have the following validation rules:
1. The number of inputs must be between 1 and 6
2. The INIT value cannot exceed the maximum allowed by the number of inputs (for N-input LUTs, the maximum INIT value is 2^(2^N)-1)
3. All inputs must be of type i1 (boolean)
4. The output must be of type i1 (boolean)

## Design Considerations

When using LUT primitives, consider the following:

1. **Resource Utilization**: Each physical LUT6 can implement one 6-input function or multiple smaller functions (e.g., two 5-input functions with shared inputs).

2. **Timing Impact**: Larger LUTs (especially LUT6) may have slightly longer delays than smaller LUTs.

3. **Power Consumption**: The INIT pattern can affect power consumption; patterns with frequent output transitions (like XOR) may consume more dynamic power.

4. **Synthesis Optimizations**: While these operations provide direct control over the LUT implementation, synthesis tools may optimize or combine multiple LUT operations for better resource utilization.

## API

LUT operations provide the following common interface:

- All inputs must be boolean values (i1)
- Return type is always a single boolean value (i1)
- All LUT operations use the INIT attribute to define their truth table
- The validator ensures that the INIT attribute is valid for a given number of inputs
- For specialized LUT operations (lut1-lut6), the input pins are labeled with the Xilinx convention: I0, I1, ..., I5