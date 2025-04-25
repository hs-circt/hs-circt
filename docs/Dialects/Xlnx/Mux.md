# Xlnx Dialect - Multiplexer Primitives

This document describes the Multiplexer (Mux) primitive operations available in the Xlnx dialect, specifically focusing on the hardened MUXFx primitives found in Xilinx UltraScale+ devices.

## Introduction

Xilinx FPGAs, particularly the UltraScale+ family, incorporate dedicated, hardened multiplexer primitives within their Configurable Logic Blocks (CLBs). These multiplexers (`MUXF7`, `MUXF8`, `MUXF9`) provide an efficient way to implement wider logic functions by combining the outputs of Lookup Tables (LUTs) or other logic. The Xlnx dialect provides direct representations for these primitives.

Using these specific Mux operations allows for a more direct mapping to the underlying hardware, potentially leading to better optimization and resource utilization compared to implementing the same logic purely with LUTs.

## Operation Definitions

The Xlnx dialect currently supports the following Mux primitives:

### `xlnx.muxf7`

This operation models the MUXF7 primitive, a 2-to-1 multiplexer that is typically used to combine outputs from two LUT6 elements.

```mlir
%out = xlnx.muxf7(%in0, %in1, %sel) : (i1, i1, i1) -> i1
```

**Operands:**
- `%in0`: First data input (i1). Selected when `%sel` is 0.
- `%in1`: Second data input (i1). Selected when `%sel` is 1.
- `%sel`: Select input (i1). Controls which input is routed to the output.

**Result:**
- `%out`: Output (i1). Equals `%in0` when `%sel` is 0, or `%in1` when `%sel` is 1.

**Logic Table:**

| Inputs |||  Output |
|--------|--------|--------|--------|
| sel | in0 | in1 || out |
| 0 | x | 0 || x |
| 0 | x | 1 || x |
| 1 | 0 | x || x |
| 1 | 1 | x || x |

Where 'x' represents the value that is passed through to the output based on the select signal.

### `xlnx.muxf8`

This operation models the MUXF8 primitive, a 2-to-1 multiplexer that is typically placed after MUXF7 in the cascade to create wider multiplexers.

```mlir
%out = xlnx.muxf8(%in0, %in1, %sel) : (i1, i1, i1) -> i1
```

**Operands:**
- `%in0`: First data input (i1). Selected when `%sel` is 0.
- `%in1`: Second data input (i1). Selected when `%sel` is 1.
- `%sel`: Select input (i1). Controls which input is routed to the output.

**Result:**
- `%out`: Output (i1). Equals `%in0` when `%sel` is 0, or `%in1` when `%sel` is 1.

**Logic Table:**

| Inputs |||  Output |
|--------|--------|--------|--------|
| sel | in0 | in1 || out |
| 0 | x | 0 || x |
| 0 | x | 1 || x |
| 1 | 0 | x || x |
| 1 | 1 | x || x |

### `xlnx.muxf9`

This operation models the MUXF9 primitive, a 2-to-1 multiplexer, potentially placed after MUXF8 to create even wider multiplexers or logic functions.

```mlir
%out = xlnx.muxf9(%in0, %in1, %sel) : (i1, i1, i1) -> i1
```

**Operands:**
- `%in0`: First data input (i1). Selected when `%sel` is 0.
- `%in1`: Second data input (i1). Selected when `%sel` is 1.
- `%sel`: Select input (i1). Controls which input is routed to the output.

**Result:**
- `%out`: Output (i1). Equals `%in0` when `%sel` is 0, or `%in1` when `%sel` is 1.

**Logic Table:**

| Inputs |||  Output |
|--------|--------|--------|--------|
| sel | in0 | in1 || out |
| 0 | x | 0 || x |
| 0 | x | 1 || x |
| 1 | 0 | x || x |
| 1 | 1 | x || x |

## Hardware Implementation

In UltraScale+ FPGAs, the MUXFx primitives are hardened multiplexers built into the CLB structure. They are physically located adjacent to the LUT elements and form part of the carry chain and wide function implementation architecture.

The typical implementation hierarchy is:
1. LUT6 elements provide 6-input combinational logic
2. MUXF7 combines two LUT6 outputs, controlled by a 7th input signal
3. MUXF8 combines two MUXF7 outputs, controlled by an 8th input signal
4. MUXF9 (where available) combines two MUXF8 outputs, controlled by a 9th input signal

This creates a cascade that can efficiently implement:
- 7-input functions (using MUXF7)
- 8-input functions (using MUXF7 + MUXF8)
- 9-input functions (using MUXF7 + MUXF8 + MUXF9)
- Wide multiplexers (4:1, 8:1, 16:1, etc.)

## Examples

### Building a 4-to-1 Multiplexer

```mlir
// Assume %d0, %d1, %d2, %d3 are data inputs (i1)
// Assume %s0, %s1 are select inputs (i1)

// First level muxes
%mux01 = xlnx.muxf7(%d0, %d1, %s0) : (i1, i1, i1) -> i1
%mux23 = xlnx.muxf7(%d2, %d3, %s0) : (i1, i1, i1) -> i1

// Second level mux
%out4to1 = xlnx.muxf8(%mux01, %mux23, %s1) : (i1, i1, i1) -> i1
```

### 8-to-1 Multiplexer Using MUXFx Cascade

```mlir
// Assume %d0 through %d7 are data inputs (i1)
// Assume %s0, %s1, %s2 are select inputs (i1)

// First level: Four MUXF7 primitives
%mux01 = xlnx.muxf7(%d0, %d1, %s0) : (i1, i1, i1) -> i1
%mux23 = xlnx.muxf7(%d2, %d3, %s0) : (i1, i1, i1) -> i1
%mux45 = xlnx.muxf7(%d4, %d5, %s0) : (i1, i1, i1) -> i1
%mux67 = xlnx.muxf7(%d6, %d7, %s0) : (i1, i1, i1) -> i1

// Second level: Two MUXF8 primitives
%mux0123 = xlnx.muxf8(%mux01, %mux23, %s1) : (i1, i1, i1) -> i1
%mux4567 = xlnx.muxf8(%mux45, %mux67, %s1) : (i1, i1, i1) -> i1

// Third level: Final output
%out8to1 = xlnx.muxf9(%mux0123, %mux4567, %s2) : (i1, i1, i1) -> i1
```

## Validation Rules

For all MUXFx operations:
- All operands (`in0`, `in1`, `sel`) must be of `i1` type.
- The operation must have exactly three operands.
- The operation must produce exactly one result of `i1` type.

## Design Considerations

When using MUXFx primitives, consider the following:

1. **Physical Mapping**: These operations map directly to physical multiplexer resources in the FPGA, which are a limited resource in the CLB architecture.

2. **Cascading**: MUXFx primitives are typically used in a cascade to build wider multiplexers or implement complex functions.

3. **Optimization**: Using these primitives can result in more efficient implementation of wide functions than using LUTs alone.

4. **LUT+MUX Integration**: In actual hardware, the MUXFx elements are tightly integrated with the LUT architecture, often taking inputs directly from adjacent LUTs.

5. **Carry Chain Interaction**: The MUXFx resources share physical structure with the carry chain logic, so there may be resource conflicts in highly utilized designs.

## API

These Mux operations follow standard MLIR operation conventions. They have no specific attributes beyond the operand and result types. The operations directly model the behavior of the hardware multiplexer primitives in Xilinx FPGAs. 