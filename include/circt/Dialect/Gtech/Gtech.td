#ifndef CIRCT_DIALECT_GTECH_TD
#define CIRCT_DIALECT_GTECH_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def Gtech_Dialect : Dialect {
  let name = "gtech";
  let cppNamespace = "::circt::gtech";
  let summary = "Representation of Gtech";
}

include "circt/Dialect/Gtech/GtechOps.td"


#endif // CIRCT_DIALECT_GTECH_TD