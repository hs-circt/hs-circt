//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_VFC_OPS
#define CIRCT_DIALECT_VFC_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "circt/Dialect/VFC/VFCDialect.td"

// Base class for VFC operations
class VFCOp<string mnemonic, list<Trait> traits = []> :
    Op<VFC_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//

def VFC_AndOp : VFCOp<"and", [Commutative, SameOperandsAndResultType]> {
  let summary = "Logical AND operation";
  
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$result);
  
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

def VFC_OrOp : VFCOp<"or", [Commutative, SameOperandsAndResultType]> {
  let summary = "Logical OR operation";
  
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$result);
  
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

#endif // CIRCT_DIALECT_VFC_OPS 