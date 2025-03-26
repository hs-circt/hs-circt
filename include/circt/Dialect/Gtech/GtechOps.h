#ifndef CIRCT_DIALECT_GTECH_GTECHOPS_H
#define CIRCT_DIALECT_GTECH_GTECHOPS_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"


#include "circt/Dialect/Gtech/GtechDialect.h"

#define GET_OP_CLASSES
#include "circt/Dialect/Gtech/Gtech.h.inc"

#endif // CIRCT_DIALECT_GTECH_GTECHOPS_H