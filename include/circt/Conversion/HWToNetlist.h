//===- HWToLLVM.h - HW to LLVM pass entry point -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the HWToNetlist pass
// constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_HWTONETLIST_HWTONETLIST_H
#define CIRCT_CONVERSION_HWTONETLIST_HWTONETLIST_H

#include "circt/Dialect/HW/HWTypes.h"
#include <memory>

namespace mlir {
template <typename T>
class OperationPass;
} // namespace mlir

namespace circt {
namespace hw {
class HWModuleOp;
} // namespace hw
} // namespace circt

namespace circt {

#define GEN_PASS_DECL_LOWERHWTONETLIST
#include "circt/Conversion/Passes.h.inc"

/// Get the HW to Netlist type conversions.
// void populateHWToNetlistTypeConversions(FPGANetlistTypeConverter &converter);

/// Get the HW to Netlist conversion patterns.
// void populateHWToNetlistConversionPatterns(
//     FPGANetlistTypeConverter &converter, RewritePatternSet &patterns);

/// Create an HW to Netlist conversion pass.
std::unique_ptr<OperationPass<hw::HWModuleOp>> createLowerHWToNetlistPass();

} // namespace circt

#endif // CIRCT_CONVERSION_HWTONETLIST_HWTONETLIST_H
