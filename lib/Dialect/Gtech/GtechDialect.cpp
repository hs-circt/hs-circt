#include "circt/Dialect/Gtech/GtechDialect.h"
#include "circt/Dialect/Gtech/GtechOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace circt;
using namespace circt::gtech;


void GtechDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Gtech/Gtech.cpp.inc"
      >();
}


#include "circt/Dialect/Gtech/GtechDialect.cpp.inc"
