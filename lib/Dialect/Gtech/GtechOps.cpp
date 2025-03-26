#include "circt/Dialect/Gtech/GtechOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace circt::gtech;

#define GET_OP_CLASSES
#include "circt/Dialect/Gtech/Gtech.cpp.inc"