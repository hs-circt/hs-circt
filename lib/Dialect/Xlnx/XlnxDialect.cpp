//===- XlnxDialect.cpp - Xlnx dialect implementation ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Xlnx/XlnxDialect.h"
#include "circt/Dialect/Xlnx/XlnxOps.h"

using namespace mlir;
using namespace circt;
using namespace circt::xlnx;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void XlnxDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Xlnx/Xlnx.cpp.inc"
      >();
}

#include "circt/Dialect/Xlnx/XlnxDialect.cpp.inc" 