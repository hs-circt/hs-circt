//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_XLNX_XLNXOPS_H
#define CIRCT_DIALECT_XLNX_XLNXOPS_H

#include "circt/Dialect/Xlnx/XlnxDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/ValueRange.h"

#define GET_OP_CLASSES
#include "circt/Dialect/Xlnx/Xlnx.h.inc"

#endif // CIRCT_DIALECT_XLNX_XLNXOPS_H
