add_circt_dialect(VFC vfc)
add_circt_doc(VFCDialect VFCDialect VFC/ -gen-dialect-doc)
add_circt_doc(VFC VFC VFC/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS VFC.td)
mlir_tablegen(VFCEnums.h.inc -gen-enum-decls)
mlir_tablegen(VFCEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRVFCEnumsIncGen)
add_dependencies(circt-headers MLIRVFCEnumsIncGen)

mlir_tablegen(VFCAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=vfc)
mlir_tablegen(VFCAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=vfc)
add_public_tablegen_target(MLIRVFCAttributesIncGen)
add_dependencies(circt-headers MLIRVFCAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS VFCOps.td)
mlir_tablegen(VFCOps.h.inc -gen-op-decls)
mlir_tablegen(VFCOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRVFCOpsIncGen)
add_dependencies(circt-headers MLIRVFCOpsIncGen)

