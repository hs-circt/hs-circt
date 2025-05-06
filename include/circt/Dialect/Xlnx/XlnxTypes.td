//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"

def UI2 : UI<2>;
def UI4 : UI<4>;

def UI2Attr : TypedUnsignedIntegerAttrBase<
    UI2, "uint8_t", "2-bit unsigned integer attribute">;
def UI4Attr : TypedUnsignedIntegerAttrBase<
    UI4, "uint8_t", "4-bit unsigned integer attribute">;

def AsynchronousControl : NativeOpTrait<"AsynchronousControl"> {
  let cppNamespace = "::circt::xlnx";
}

def SynchronousControl : NativeOpTrait<"SynchronousControl"> {
  let cppNamespace = "::circt::xlnx";
}
