#ifndef CIRCT_DIALECT_GTECH_OPS_TD
#define CIRCT_DIALECT_GTECH_OPS_TD

include "circt/Dialect/Gtech/Gtech.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class GtechOp<string mnemonic, list<Trait> traits = []> :
  Op<Gtech_Dialect, mnemonic, traits>;


def NAndOp : GtechOp<"nand", [Pure]> {
  let summary = "NAnd";
  let description = [{
    `nand`
    
    example：
    ```mlir
    %result = gtech.nand %a, %b : i1
    ```
  }];

  let arguments = (ins I1:$lhs, I1:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";

}

#endif // CIRCT_DIALECT_GTECH_OPS_TD