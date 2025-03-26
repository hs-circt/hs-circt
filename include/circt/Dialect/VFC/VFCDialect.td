//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_VFC_DIALECT
#define CIRCT_DIALECT_VFC_DIALECT

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def VFC_Dialect : Dialect {
  let name = "vfc";
  let cppNamespace = "::circt::vfc";
  let summary = "VFC dialect for verification and formal checking";
}

#endif // CIRCT_DIALECT_VFC_DIALECT 