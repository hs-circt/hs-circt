# Xlnx Dialect - Multiplexer Primitives

This document describes the Multiplexer (Mux) primitive operations available in the Xlnx dialect, specifically focusing on the hardened MUXFx primitives found in Xilinx UltraScale+ devices.

## Introduction

Xilinx FPGAs, particularly the UltraScale+ family, incorporate dedicated, hardened multiplexer primitives within their Configurable Logic Blocks (CLBs). These multiplexers (`MUXF7`, `MUXF8`, `MUXF9`) provide an efficient way to implement wider logic functions by combining the outputs of Lookup Tables (LUTs) or other logic. The Xlnx dialect provides direct representations for these primitives.

Using these specific Mux operations allows for a more direct mapping to the underlying hardware, potentially leading to better optimization and resource utilization compared to implementing the same logic purely with LUTs.

## List of Operations

The Xlnx dialect currently supports the following Mux primitives:

- `xlnx.muxf7`: Represents the F7 multiplexer.
- `xlnx.muxf8`: Represents the F8 multiplexer.
- `xlnx.muxf9`: Represents the F9 multiplexer.

These typically form a cascade within the CLB slice, where the output of MUXF7 can feed into MUXF8, and MUXF8 into MUXF9, enabling the construction of wider multiplexers or logic functions.

### `xlnx.muxf7`

This operation models the MUXF7 primitive, a 2-to-1 multiplexer.

```mlir
%out = xlnx.muxf7(%in0, %in1, %sel) : (i1, i1, i1) -> i1
```

- `%in0`: First data input (i1). Output when `%sel` is 0.
- `%in1`: Second data input (i1). Output when `%sel` is 1.
- `%sel`: Select input (i1).
- `%out`: Output (i1).

### `xlnx.muxf8`

This operation models the MUXF8 primitive, also a 2-to-1 multiplexer, typically placed after MUXF7 in the cascade.

```mlir
%out = xlnx.muxf8(%in0, %in1, %sel) : (i1, i1, i1) -> i1
```

- `%in0`: First data input (i1). Output when `%sel` is 0.
- `%in1`: Second data input (i1). Output when `%sel` is 1.
- `%sel`: Select input (i1).
- `%out`: Output (i1).

### `xlnx.muxf9`

This operation models the MUXF9 primitive, a 2-to-1 multiplexer, potentially placed after MUXF8. Support for MUXF9 might vary depending on the specific Xilinx architecture details, but it follows the same pattern.

```mlir
%out = xlnx.muxf9(%in0, %in1, %sel) : (i1, i1, i1) -> i1
```

- `%in0`: First data input (i1). Output when `%sel` is 0.
- `%in1`: Second data input (i1). Output when `%sel` is 1.
- `%sel`: Select input (i1).
- `%out`: Output (i1).

## Example

Building a 4-to-1 Multiplexer using MUXF7 and MUXF8:

```mlir
// Assume %d0, %d1, %d2, %d3 are data inputs (i1)
// Assume %s0, %s1 are select inputs (i1)

// First level muxes
%mux01 = xlnx.muxf7(%d0, %d1, %s0) : (i1, i1, i1) -> i1
%mux23 = xlnx.muxf7(%d2, %d3, %s0) : (i1, i1, i1) -> i1

// Second level mux
%out4to1 = xlnx.muxf8(%mux01, %mux23, %s1) : (i1, i1, i1) -> i1
```

## Validation Rules

- All operands (`in0`, `in1`, `sel`) must be of `i1` type.
- The operation must have exactly three operands.
- The operation must produce exactly one result of `i1` type.

## API

These Mux operations follow standard MLIR operation conventions. They have no specific attributes or special interfaces beyond operand and result types. 