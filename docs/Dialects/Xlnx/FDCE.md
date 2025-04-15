# Xlnx Dialect - FDCE Flip-Flop Primitive

This document describes the FDCE (D-type Flip-Flop with Clock Enable and Asynchronous Clear) primitive operation available in the Xlnx dialect.

## Introduction

The `FDCE` primitive is a fundamental sequential element found in Xilinx FPGAs, including the UltraScale+ family. It represents a positive-edge-triggered D-type flip-flop with additional controls for clock enable (`CE`) and asynchronous clear (`CLR`). This primitive is essential for implementing registered logic, state machines, pipelines, and other sequential circuits.

The `xlnx.fdce` operation in the Xlnx dialect provides a direct mapping to this hardware primitive, allowing for precise modeling of sequential behavior. It corresponds to the FDCE primitive described in the Xilinx UltraScale Architecture Libraries Guide (UG974) [1].

*[1] Xilinx, Inc. UltraScale Architecture Libraries Guide (UG974). (URL might be similar to the provided https://docs.amd.com/r/en-US/ug974-vivado-ultrascale-libraries/FDPE, referencing the FDCE or similar FF primitives)*


## Operation Definition

### `xlnx.fdce`

Models the FDCE flip-flop primitive.

```mlir
%q = xlnx.fdce(%d, %c, %ce, %clr) : (i1, seq.clock, i1, i1) -> i1
```

**Operands:**

- `%d` (Data Input, `i1`): The value to be captured by the flip-flop on the clock edge when enabled.
- `%c` (Clock Input, `seq.clock`): The clock signal. The flip-flop typically captures data on the positive (rising) edge of this clock. The `seq.clock` type comes from the CIRCT `seq` dialect.
- `%ce` (Clock Enable, `i1`): Controls when the flip-flop is active. If `CE` is high (1), the flip-flop samples the `D` input on the active clock edge. If `CE` is low (0), the flip-flop holds its current value, ignoring the `D` input and the clock edge.
- `%clr` (Asynchronous Clear, `i1`): Resets the flip-flop output. If `CLR` is high (1), the output `Q` is immediately forced to 0, regardless of the clock, `D`, or `CE` inputs. The clear takes precedence over other inputs. If `CLR` is low (0), the clear is inactive.

**Result:**

- `%q` (Output, `i1`): The registered output of the flip-flop.

**Behavior Summary:**

1.  **Asynchronous Clear (CLR high):** `Q` immediately becomes 0.
2.  **CLR low, CE high, Positive Clock Edge:** `Q` takes the value of `D`.
3.  **CLR low, CE low, Positive Clock Edge:** `Q` retains its previous value.
4.  **CLR low, No Positive Clock Edge:** `Q` retains its previous value.

## Example

```mlir
// Assume %data_in, %clk, %enable, %reset are MLIR values
// %clk should be of type seq.clock
// %data_in, %enable, %reset should be of type i1

%registered_data = xlnx.fdce(%data_in, %clk, %enable, %reset) : (i1, seq.clock, i1, i1) -> i1
```

This example shows instantiating an FDCE flip-flop. The output `%registered_data` will hold the value of `%data_in` from the previous clock cycle, provided `%enable` was high and `%reset` was low during that cycle's rising edge. If `%reset` goes high, `%registered_data` will become 0 asynchronously.

## Validation Rules

- Operand types must match the definition: `i1`, `seq.clock`, `i1`, `i1`.
- The operation must have exactly four operands.
- The operation must produce exactly one result of `i1` type.

## API

The `xlnx.fdce` operation follows standard MLIR operation conventions. It has no operation-specific attributes. 