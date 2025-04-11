//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_XLNX_XLNXDIALECT_TD
#define CIRCT_DIALECT_XLNX_XLNXDIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def XlnxDialect : Dialect {
    let name = "xlnx";
    let summary = "Xilinx dialect (UltraScale+ Only)";
    let description = "This dialect contains Xilinx operations for UltraScale+ devices";
    let cppNamespace = "::circt::xlnx";
    let dependentDialects = ["circt::hw::HWDialect", "circt::seq::SeqDialect"];
}

#endif // CIRCT_DIALECT_XLNX_XLNXDIALECT_TD
