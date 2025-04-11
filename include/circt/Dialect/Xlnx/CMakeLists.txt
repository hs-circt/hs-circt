
# Prepare dialect variables
set(DIALECT_NAME Xlnx)
set(DIALECT_NAMESPACE xlnx)
set(DIALECT_DESCRIPTION "Xilinx dialect")
set(DIALECT_DEPENDS "circt::mlir::IR")
set(DIALECT_DIR ${DIALECT_NAME})
set(DIALECT_INC_GEN_PREFIX "${DIALECT_NAME}")
set(XLNX_INC_GEN_DEPS "")

# Add dialect
add_circt_dialect(${DIALECT_NAME} ${DIALECT_NAMESPACE})

# Add dialect documentation
add_circt_doc(${DIALECT_NAME}Dialect ${DIALECT_NAME}Dialect ${DIALECT_DIR}/ -gen-dialect-doc)
add_circt_doc(${DIALECT_NAME} ${DIALECT_NAME} ${DIALECT_DIR}/ -gen-op-doc)

# Generate dialect enums
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_LIST_DIR}/Xlnx.td)
mlir_tablegen(${DIALECT_INC_GEN_PREFIX}Enums.h.inc -gen-enum-decls)
mlir_tablegen(${DIALECT_INC_GEN_PREFIX}Enums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(${DIALECT_INC_GEN_PREFIX}EnumsIncGen)
add_dependencies(circt-headers ${DIALECT_INC_GEN_PREFIX}EnumsIncGen)
list(APPEND XLNX_INC_GEN_DEPS ${DIALECT_INC_GEN_PREFIX}EnumsIncGen)

# Generate dialect attributes
mlir_tablegen(${DIALECT_INC_GEN_PREFIX}Attributes.h.inc -gen-attrdef-decls -attrdefs-dialect=${DIALECT_NAMESPACE})
mlir_tablegen(${DIALECT_INC_GEN_PREFIX}Attributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=${DIALECT_NAMESPACE})
add_public_tablegen_target(${DIALECT_INC_GEN_PREFIX}AttributesIncGen)
add_dependencies(circt-headers ${DIALECT_INC_GEN_PREFIX}AttributesIncGen)
list(APPEND XLNX_INC_GEN_DEPS ${DIALECT_INC_GEN_PREFIX}AttributesIncGen)

# Generate dialect operations
set(LLVM_TARGET_DEFINITIONS ${DIALECT_NAME}Ops.td)
mlir_tablegen(${DIALECT_INC_GEN_PREFIX}Ops.h.inc -gen-op-decls)
mlir_tablegen(${DIALECT_INC_GEN_PREFIX}Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(${DIALECT_INC_GEN_PREFIX}OpsIncGen)
add_dependencies(circt-headers ${DIALECT_INC_GEN_PREFIX}OpsIncGen)
list(APPEND XLNX_INC_GEN_DEPS ${DIALECT_INC_GEN_PREFIX}OpsIncGen)

# Generate dialect interfaces
set(LLVM_TARGET_DEFINITIONS ${DIALECT_NAME}OpInterfaces.td)
mlir_tablegen(${DIALECT_INC_GEN_PREFIX}OpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(${DIALECT_INC_GEN_PREFIX}OpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(${DIALECT_INC_GEN_PREFIX}OpInterfacesIncGen)
add_dependencies(circt-headers ${DIALECT_INC_GEN_PREFIX}OpInterfacesIncGen)
list(APPEND XLNX_INC_GEN_DEPS ${DIALECT_INC_GEN_PREFIX}OpInterfacesIncGen)
