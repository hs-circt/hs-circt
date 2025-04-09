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

Each LUT operation has its truth table defined by the `INIT` attribute, which is a 64-bit unsigned integer. Each bit of the `INIT` attribute corresponds to the output value for a specific input combination.