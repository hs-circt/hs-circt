//===- CoreToXlnx.h - Core to Xlnx Conversion Pass -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// XlnxToHW
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_XLNXTOHW_H
#define CIRCT_CONVERSION_XLNXTOHW_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {

#define GEN_PASS_DECL_XLNXTOHW
#include "circt/Conversion/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertXlnxToHWPass();

} // namespace circt

#endif // CIRCT_CONVERSION_XLNXTOHW_H