# Xlnx Dialect

This dialect contains specific operations for Xilinx FPGAs, and the current version only supports the UltraScale+ family of devices.

## Introduction

The Xlnx dialect provides a set of operations for representing and manipulating specific hardware primitives in Xilinx FPGAs. The initial implementation focuses on the Lookup Table (LUT) primitive, which is the most basic combinatorial logic unit in the Xilinx FPGA architecture.

Currently, the Xlnx dialect implements full support for 1-6 input lookup tables (LUT1-LUT6), which are the core logic elements in the UltraScale+ architecture.

## Operation List

### Lookup Table (LUT) Operations

- `xlnx.lutn` - Generic lookup table operation that accepts 1 to 6 inputs
- `xlnx.lut1` - 1-input lookup table
- `xlnx.lut2` - 2-input lookup table
- `xlnx.lut3` - 3-input lookup table
- `xlnx.lut4` - 4-input lookup table
- `xlnx.lut5` - 5-input lookup table
- `xlnx.lut6` - 6-input lookup table

Each LUT operation has its truth table defined by the `INIT` attribute, which is an unsigned integer with a bit width that corresponds to the LUT size (2^N bits for an N-input LUT). For specific LUT operations, the bit widths are:
- LUT1: 2-bit INIT (ui2)
- LUT2: 4-bit INIT (ui4)
- LUT3: 8-bit INIT (ui8)
- LUT4: 16-bit INIT (ui16)
- LUT5: 32-bit INIT (ui32)
- LUT6: 64-bit INIT (ui64)

For the generic `xlnx.lutn` operation, a 64-bit unsigned integer is used, with only the lower 2^N bits being relevant for an N-input LUT.

Each bit of the `INIT` attribute corresponds to the output value for a specific input combination.

### Multiplexer (Mux) Operations

The `Xlnx` dialect includes operations representing the hardened multiplexer primitives commonly found within Xilinx CLBs, such as `MUXF7`, `MUXF8`, and `MUXF9`. These are typically used to combine LUT outputs efficiently.

- `xlnx.muxf7` - Represents the F7 multiplexer primitive, usually a 2-to-1 mux controlled by a select signal.
- `xlnx.muxf8` - Represents the F8 multiplexer primitive, often cascading from `MUXF7`.
- `xlnx.muxf9` - Represents the F9 multiplexer primitive, potentially cascading from `MUXF8`.

These operations typically take two data inputs (`i1`), one select input (`i1`), and produce one output (`i1`). Example:

```mlir
%out_f7 = xlnx.muxf7(%in0, %in1, %select) : (i1, i1, i1) -> i1
```

### Flip-Flop (FF) Operations

The dialect supports common flip-flop primitives used for sequential logic.

- `xlnx.fdce` - Represents the FDCE primitive: a D-type flip-flop with Clock Enable (`CE`) and asynchronous Clear (`CLR`).

The `FDCE` primitive captures the data input `D` on the rising edge of the clock `C` only when the clock enable `CE` is active (typically high). The asynchronous clear `CLR`, when active (typically high), resets the output `Q` to 0 regardless of other inputs.

Its MLIR representation is:

```mlir
%q = xlnx.fdce(%d, %c, %ce, %clr) : (i1, seq.clock, i1, i1) -> i1
```

Where `%d`, `%ce`, `%clr` are `i1` type, `%c` is `seq.clock`, and the output `%q` is `i1`.