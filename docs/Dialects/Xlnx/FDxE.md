# Xlnx Dialect - FDxE Flip-Flop Primitives

This document describes the FDxE series flip-flop primitive operations available in the Xlnx dialect, including FDCE, FDPE, FDSE, and FDRE.

## Introduction

FDxE primitives are fundamental timing elements in Xilinx FPGAs (including the UltraScale+ series). They represent D-type flip-flops with clock enable (`CE`) and various control signals. These primitives are essential for implementing register logic, state machines, pipelines, and other sequential circuits.

The corresponding operations in the Xlnx dialect provide direct mappings to these hardware primitives, allowing precise modeling of sequential behavior. They correspond to the respective primitives described in the Xilinx UltraScale Architecture Libraries Guide (UG974).

*[1] Xilinx, Inc. UltraScale Architecture Libraries Guide (UG974).*

## Operation Definitions

### `xlnx.fdce`

Models the FDCE flip-flop primitive (D flip-flop with clock enable and asynchronous clear). This operation inherits from `XlnxFDCtrlBase` and implements the `AsynchronousControl` trait.

```mlir
%q = xlnx.fdce(C: %c, CE: %ce, CLR: %clr, D: %d) : (!seq.clock, i1, i1, i1) -> i1
```

**Operands:**
- `%d` (data input, `i1`): Value captured by the flip-flop at the clock edge when enabled.
- `%c` (clock input, `seq.clock`): Clock signal. The flip-flop typically captures data on the rising edge of this clock. The `seq.clock` type comes from CIRCT's `seq` dialect.
- `%ce` (clock enable, `i1`): Controls when the flip-flop is active. If `CE` is high (1), the flip-flop samples the `D` input at the active clock edge. If `CE` is low (0), the flip-flop retains its current value, ignoring the `D` input and clock edges.
- `%clr` (asynchronous clear, `i1`): Resets the flip-flop output. If `CLR` is high (1), the output `Q` is immediately forced to 0, regardless of the clock, `D`, or `CE` inputs. Clear takes precedence over other inputs. If `CLR` is low (0), the clear function is inactive.

**Results:**
- `%q` (output, `i1`): Registered output of the flip-flop.

**Behavior Summary:**
1. **Asynchronous clear (CLR high):** `Q` immediately becomes 0.
2. **CLR low, CE high, clock rising edge:** `Q` takes the value of `D`.
3. **CLR low, CE low, clock rising edge:** `Q` retains its previous value.
4. **CLR low, no clock rising edge:** `Q` retains its previous value.

**Logic Table:**

| CLR    | CE   | D    | C    | Q         |
|--------|------|------|------|-----------|
| 1      | X    | X    | X    | 0         |
| 0      | 0    | X    | X    | No Change |
| 0      | 1    | D    | ↑    | D         |

**Attributes:**
- `INIT` (`ui1`, default: `0`): Initial value of the flip-flop. This value is loaded into the flip-flop output on power-up or assertion of GSR (Global Set/Reset).
- `IS_C_INVERTED` (`ui1`, default: `0`): Specifies whether the clock signal is inverted. When set to 1, creates a falling-edge triggered flip-flop.
- `IS_D_INVERTED` (`ui1`, default: `0`): Specifies whether the data input is inverted.
- `IS_CLR_INVERTED` (`ui1`, default: `0`): Specifies whether the clear signal is inverted. When set to 1, the clear function becomes active-low.

### `xlnx.fdpe`

Models the FDPE flip-flop primitive (D flip-flop with clock enable and asynchronous preset). This operation inherits from `XlnxFDCtrlBase` and implements the `AsynchronousControl` trait.

```mlir
%q = xlnx.fdpe(C: %c, CE: %ce, PRE: %pre, D: %d) : (!seq.clock, i1, i1, i1) -> i1
```

**Operands:**
- `%d` (data input, `i1`): Value captured by the flip-flop at the clock edge when enabled.
- `%c` (clock input, `seq.clock`): Clock signal. The flip-flop typically captures data on the rising edge of this clock.
- `%ce` (clock enable, `i1`): Controls when the flip-flop is active. If `CE` is high (1), the flip-flop samples the `D` input at the active clock edge. If `CE` is low (0), the flip-flop retains its current value.
- `%pre` (asynchronous preset, `i1`): Presets the flip-flop output. If `PRE` is high (1), the output `Q` is immediately forced to 1, regardless of the clock, `D`, or `CE` inputs. Preset takes precedence over other inputs. If `PRE` is low (0), the preset function is inactive.

**Results:**
- `%q` (output, `i1`): Registered output of the flip-flop.

**Behavior Summary:**
1. **Asynchronous preset (PRE high):** `Q` immediately becomes 1.
2. **PRE low, CE high, clock rising edge:** `Q` takes the value of `D`.
3. **PRE low, CE low, clock rising edge:** `Q` retains its previous value.
4. **PRE low, no clock rising edge:** `Q` retains its previous value.

**Logic Table:**

| PRE    | CE   | D    | C    | Q         |
|--------|------|------|------|-----------|
| 1      | X    | X    | X    | 1         |
| 0      | 0    | X    | X    | No Change |
| 0      | 1    | D    | ↑    | D         |

**Attributes:**
- `INIT` (`ui1`, default: `1`): Initial value of the flip-flop. This value is loaded into the flip-flop output on power-up or assertion of GSR (Global Set/Reset).
- `IS_C_INVERTED` (`ui1`, default: `0`): Specifies whether the clock signal is inverted. When set to 1, creates a falling-edge triggered flip-flop.
- `IS_D_INVERTED` (`ui1`, default: `0`): Specifies whether the data input is inverted.
- `IS_PRE_INVERTED` (`ui1`, default: `0`): Specifies whether the preset signal is inverted. When set to 1, the preset function becomes active-low.

### `xlnx.fdse`

Models the FDSE flip-flop primitive (D flip-flop with clock enable and synchronous set).

```mlir
%q = xlnx.fdse(C: %c, CE: %ce, S: %s, D: %d) : (!seq.clock, i1, i1, i1) -> i1
```

**Operands:**
- `%d` (data input, `i1`): Value captured by the flip-flop at the clock edge when enabled.
- `%c` (clock input, `seq.clock`): Clock signal. The flip-flop typically captures data on the rising edge of this clock.
- `%ce` (clock enable, `i1`): Controls when the flip-flop is active. If `CE` is high (1), the flip-flop samples the `D` input at the active clock edge or performs the set operation. If `CE` is low (0), the flip-flop retains its current value.
- `%s` (synchronous set, `i1`): If both `S` and `CE` are high and a clock rising edge is encountered, the output `Q` becomes 1. The set operation occurs at the clock edge and overrides the data input value.

**Results:**
- `%q` (output, `i1`): Registered output of the flip-flop.

**Behavior Summary:**
1. **CE high, S high, clock rising edge:** `Q` becomes 1.
2. **CE high, S low, clock rising edge:** `Q` takes the value of `D`.
3. **CE low, clock rising edge:** `Q` retains its previous value.
4. **No clock rising edge:** `Q` retains its previous value.

**Logic Table:**

| S      | CE   | D    | C    | Q         |
|--------|------|------|------|-----------|
| X      | 0    | X    | X    | No Change |
| 0      | 1    | D    | ↑    | D         |
| 1      | 1    | X    | ↑    | 1         |

**Attributes:**
- `INIT` (`ui1`, default: `1`): Initial value of the flip-flop. This value is loaded into the flip-flop output on power-up or assertion of GSR (Global Set/Reset).
- `IS_C_INVERTED` (`ui1`, default: `0`): Specifies whether the clock signal is inverted. When set to 1, creates a falling-edge triggered flip-flop.
- `IS_D_INVERTED` (`ui1`, default: `0`): Specifies whether the data input is inverted.
- `IS_S_INVERTED` (`ui1`, default: `0`): Specifies whether the set signal is inverted. When set to 1, the set function becomes active-low.

### `xlnx.fdre`

Models the FDRE flip-flop primitive (D flip-flop with clock enable and synchronous reset).

```mlir
%q = xlnx.fdre(C: %c, CE: %ce, R: %r, D: %d) : (!seq.clock, i1, i1, i1) -> i1
```

**Operands:**
- `%d` (data input, `i1`): Value captured by the flip-flop at the clock edge when enabled.
- `%c` (clock input, `seq.clock`): Clock signal. The flip-flop typically captures data on the rising edge of this clock.
- `%ce` (clock enable, `i1`): Controls when the flip-flop is active. If `CE` is high (1), the flip-flop samples the `D` input at the active clock edge or performs the reset operation. If `CE` is low (0), the flip-flop retains its current value.
- `%r` (synchronous reset, `i1`): If both `R` and `CE` are high and a clock rising edge is encountered, the output `Q` becomes 0. The reset operation occurs at the clock edge and overrides the data input value.

**Results:**
- `%q` (output, `i1`): Registered output of the flip-flop.

**Behavior Summary:**
1. **CE high, R high, clock rising edge:** `Q` becomes 0.
2. **CE high, R low, clock rising edge:** `Q` takes the value of `D`.
3. **CE low, clock rising edge:** `Q` retains its previous value.
4. **No clock rising edge:** `Q` retains its previous value.

**Logic Table:**

| R      | CE   | D    | C    | Q         |
|--------|------|------|------|-----------|
| X      | 0    | X    | X    | No Change |
| 0      | 1    | D    | ↑    | D         |
| 1      | 1    | X    | ↑    | 0         |

**Attributes:**
- `INIT` (`ui1`, default: `0`): Initial value of the flip-flop. This value is loaded into the flip-flop output on power-up or assertion of GSR (Global Set/Reset).
- `IS_C_INVERTED` (`ui1`, default: `0`): Specifies whether the clock signal is inverted. When set to 1, creates a falling-edge triggered flip-flop.
- `IS_D_INVERTED` (`ui1`, default: `0`): Specifies whether the data input is inverted.
- `IS_R_INVERTED` (`ui1`, default: `0`): Specifies whether the reset signal is inverted. When set to 1, the reset function becomes active-low.

## FDxE Primitive Comparison

| Primitive | Type                 | Control Signal | Control Behavior                            | Default INIT |
|-----------|----------------------|----------------|---------------------------------------------|--------------|
| FDCE      | Asynchronous Control | CLR (Clear)    | High level immediately forces output to 0   | 0            |
| FDPE      | Asynchronous Control | PRE (Preset)   | High level immediately forces output to 1   | 1            |
| FDRE      | Synchronous Control  | R (Reset)      | High level forces output to 0 at clock edge | 0            |
| FDSE      | Synchronous Control  | S (Set)        | High level forces output to 1 at clock edge | 1            |

All primitives have clock enable (CE) functionality, which ignores clock edges when CE is low.

## Examples

```mlir
// FDCE example - Flip-flop with asynchronous clear
%q1 = xlnx.fdce(C: %clk, CE: %enable, CLR: %reset, D: %data_in) : (!seq.clock, i1, i1, i1) -> i1

// FDPE example - Flip-flop with asynchronous preset
%q2 = xlnx.fdpe(C: %clk, CE: %enable, PRE: %preset, D: %data_in) : (!seq.clock, i1, i1, i1) -> i1

// FDSE example - Flip-flop with synchronous set
%q3 = xlnx.fdse(C: %clk, CE: %enable, S: %set, D: %data_in) : (!seq.clock, i1, i1, i1) -> i1

// FDRE example - Flip-flop with synchronous reset
%q4 = xlnx.fdre(C: %clk, CE: %enable, R: %reset, D: %data_in) : (!seq.clock, i1, i1, i1) -> i1
```

## Verification Rules

For all FDxE primitives:
- Operand types must match the definitions: `!seq.clock`, `i1`, `i1`, `i1`.
- Operations must have four operands.
- Operations must produce a result of type `i1`.

## Interface Implementations

All FDxE operations implement the following interfaces:
- `Clocked`: Provides access to the clock signal.
- `ClockEnabled`: Provides access to the clock enable signal.
- `HWInstable`: Supports conversion to hardware instances, with the following methods:
  - `getGateType()`: Returns the corresponding gate type ("FDCE", "FDPE", "FDSE", or "FDRE").
  - `getPortDict()`: Returns a mapping of port names to values (C, CE, control input, D, Q).
  - `getSelectAttrDict(uint32_t idx)`: Returns attribute values for a given index.

FDCE and FDPE operations also inherit the `AsynchronousControl` trait, indicating they have asynchronous control capabilities. FDRE and FDSE operations inherit the `SynchronousControl` trait, indicating they have synchronous control capabilities.

## Assembly Format

The assembly format for each operation follows a similar pattern, with slight variations based on its control signals:

### FDCE
```
`(` `C` `:` $clock `,` `CE` `:` $clockEnable `,` `CLR` `:` $asyncClear `,` `D` `:` $dataInput `)`
attr-dict
`:` type($clock) `,` type($clockEnable) `,` type($asyncClear) `,` type($dataInput) `->` type($dataOutput)
```

### FDPE
```
`(` `C` `:` $clock `,` `CE` `:` $clockEnable `,` `PRE` `:` $asyncPreset `,` `D` `:` $dataInput `)`
attr-dict
`:` type($clock) `,` type($clockEnable) `,` type($asyncPreset) `,` type($dataInput) `->` type($dataOutput)
```

### FDSE
```
`(` `C` `:` $clock `,` `CE` `:` $clockEnable `,` `S` `:` $syncSet `,` `D` `:` $dataInput `)`
attr-dict
`:` type($clock) `,` type($clockEnable) `,` type($syncSet) `,` type($dataInput) `->` type($dataOutput)
```

### FDRE
```
`(` `C` `:` $clock `,` `CE` `:` $clockEnable `,` `R` `:` $syncReset `,` `D` `:` $dataInput `)`
attr-dict
`:` type($clock) `,` type($clockEnable) `,` type($syncReset) `,` type($dataInput) `->` type($dataOutput)
```

## API

Each FDxE operation follows standard MLIR operation conventions. FDCE and FDPE inherit from the `XlnxFDCtrlBase` class, which provides common functionality for Xilinx D flip-flops with clock enable and control inputs. FDRE and FDSE have their own base classes to support their synchronous control features.

## Design Considerations

When choosing the appropriate flip-flop type in a design, consider the following factors:

1. **Control Signal Type**: If an immediate response to reset/set is needed, choose FDCE/FDPE with asynchronous control; if control signals should be synchronized with the clock to avoid metastability and glitch issues, choose FDRE/FDSE.

2. **Default State**: FDPE and FDSE have a default initial state of 1, while FDCE and FDRE have a default initial state of 0.

3. **Timing Analysis**: Asynchronous control signals (CLR/PRE) are not constrained by the clock, which may complicate timing analysis, while synchronous controls (R/S) are related to the clock, making timing analysis easier.

4. **Power Considerations**: In large designs, clock enable (CE) can be used to reduce dynamic power consumption by disabling clock sampling in inactive areas. 