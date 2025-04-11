//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_XLNX_TYPES_H
#define CIRCT_DIALECT_XLNX_TYPES_H

// clang-format off
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Xlnx/XlnxTypes.h.inc"
// clang-format on

namespace circt {
namespace xlnx {} // end namespace xlnx
} // end namespace circt

#endif // CIRCT_DIALECT_XLNX_TYPES_H
