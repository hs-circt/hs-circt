//===- CoreToXlnx.h - Core to Xlnx Conversion Pass -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for converting Core dialects (Seq, Comb, HW)
// operations to the Xlnx dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_CORETOXLNX_CORETOXLNX_H
#define CIRCT_CONVERSION_CORETOXLNX_CORETOXLNX_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {

/// Creates a pass that converts Core dialect operations (Seq, Comb, HW)
/// to the Xlnx dialect.
#define GEN_PASS_DECL_CORETOXLNX
#include "circt/Conversion/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertCoreToXlnxPass();

} // namespace circt

#endif // CIRCT_CONVERSION_CORETOXLNX_CORETOXLNX_H 