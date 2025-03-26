//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/VFC/VFCDialect.h"
#include "circt/Dialect/VFC/VFCOps.h"

void circt::vfc::VFCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/VFC/VFCOps.cpp.inc"
      >();
}

#include "circt/Dialect/VFC/VFCDialect.cpp.inc" 