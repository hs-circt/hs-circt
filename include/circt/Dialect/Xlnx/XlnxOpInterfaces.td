//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_XILINX_XILINXOPINTERFACES
#define CIRCT_DIALECT_XILINX_XILINXOPINTERFACES

include "mlir/IR/OpBase.td"

def Clocked : OpInterface<"Clocked"> {
  let cppNamespace = "circt::xlnx";
  let description = "Trait for clocked elements";

  let methods = [
    InterfaceMethod<"Get the clock signal", "mlir::Value", "getClock">
  ];
}

def ClockEnabled : OpInterface<"ClockEnabled"> {
  let cppNamespace = "circt::xlnx";
  let description = "Trait for ClockEnabled elements";

  let methods = [
    InterfaceMethod<"Get the clock enable signal", "mlir::Value", "getClockEnable">
  ];
}

def HWInstable : OpInterface<"HWInstable"> {
  let cppNamespace = "circt::xlnx";
  let description = "Pack ports and attributes into two dictionaries";

  let methods = [
    InterfaceMethod<
      "Get the gate type",
      "mlir::StringRef",
      "getGateType"
    >,
    InterfaceMethod<
      "Get the port dictionary (port name -> port value)",
      "llvm::StringMap<mlir::Value>",
      "getPortDict"
    >,
    InterfaceMethod<
      "Get the attribute dictionary (attribute name -> attribute value)",
      "llvm::StringMap<mlir::Attribute>",
      "getAttrDict"
    >
  ];
}

#endif  // CIRCT_DIALECT_XILINX_XILINXOPINTERFACES
