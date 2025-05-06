//===- HWToLLVM.h - HW to LLVM pass entry point -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the CombToNetlist pass
// constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_HWTONETLIST_HWTONETLIST_H
#define CIRCT_CONVERSION_HWTONETLIST_HWTONETLIST_H

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

#define GEN_PASS_DECL_LOWERCOMBTONETLIST
#include "circt/Conversion/Passes.h.inc"

/// Get the HW to Netlist type conversions.
// void populateCombToNetlistTypeConversions(FPGANetlistTypeConverter &converter);

/// Get the HW to Netlist conversion patterns.
// void populateCombToNetlistConversionPatterns(
//     FPGANetlistTypeConverter &converter, RewritePatternSet &patterns);

/// Create an HW to Netlist conversion pass.
std::unique_ptr<OperationPass<hw::HWModuleOp>> createLowerCombToNetlistPass();

} // namespace circt

#endif // CIRCT_CONVERSION_HWTONETLIST_HWTONETLIST_H
