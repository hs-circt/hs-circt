# Xlnx Dialect

This dialect contains specific operations for Xilinx FPGAs, and the current version only supports the UltraScale+ family of devices.

## Introduction

The Xlnx dialect provides a set of operations for representing and manipulating specific hardware primitives in Xilinx FPGAs. The initial implementation focused on Lookup Tables (LUTs), Multiplexers (Mux), and Flip-Flops (FFs), which are fundamental building blocks in the Xilinx FPGA architecture (initially targeting UltraScale+).

Currently, the Xlnx dialect implements full support for:
- 1-6 input lookup tables (`xlnx.lut1` to `xlnx.lut6`, and `xlnx.lutn`)
- Hardened multiplexers (`xlnx.muxf7`, `xlnx.muxf8`, `xlnx.muxf9`)
- D-type flip-flops with clock enable and various controls (`xlnx.fdce`, `xlnx.fdpe`, `xlnx.fdre`, `xlnx.fdse`)

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

The dialect supports common D-type flip-flop primitives (FDxE series) used for sequential logic, all featuring a clock input (`C`), data input (`D`), clock enable (`CE`), and one specific control signal.

- `xlnx.fdce` - D-type flip-flop with Clock Enable (`CE`) and **asynchronous Clear** (`CLR`).
  - When `CLR` is active (high), output `Q` is forced to 0 immediately.
  ```mlir
  %q = xlnx.fdce(C: %c, CE: %ce, CLR: %clr, D: %d) : (!seq.clock, i1, i1, i1) -> i1
  ```

- `xlnx.fdpe` - D-type flip-flop with Clock Enable (`CE`) and **asynchronous Preset** (`PRE`).
  - When `PRE` is active (high), output `Q` is forced to 1 immediately.
  ```mlir
  %q = xlnx.fdpe(C: %c, CE: %ce, PRE: %pre, D: %d) : (!seq.clock, i1, i1, i1) -> i1
  ```

- `xlnx.fdre` - D-type flip-flop with Clock Enable (`CE`) and **synchronous Reset** (`R`).
  - When `R` and `CE` are active (high) at the clock edge, output `Q` becomes 0.
  ```mlir
  %q = xlnx.fdre(C: %c, CE: %ce, R: %r, D: %d) : (!seq.clock, i1, i1, i1) -> i1
  ```

- `xlnx.fdse` - D-type flip-flop with Clock Enable (`CE`) and **synchronous Set** (`S`).
  - When `S` and `CE` are active (high) at the clock edge, output `Q` becomes 1.
  ```mlir
  %q = xlnx.fdse(C: %c, CE: %ce, S: %s, D: %d) : (!seq.clock, i1, i1, i1) -> i1
  ```

All FF operations sample the data input `D` on the active clock edge (typically rising edge) only when the clock enable `CE` is active (high) and the respective control signal is inactive. The clock input (`C`) uses the `!seq.clock` type from the `seq` dialect, while all other inputs and the output (`Q`) are `i1`.