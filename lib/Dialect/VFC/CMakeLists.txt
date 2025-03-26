add_circt_dialect_library(CIRCTVFC
  VFCDialect.cpp
  VFCOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/VFC

  DEPENDS
  MLIRVFCEnumsIncGen
  MLIRVFCAttributesIncGen
  MLIRVFCOpsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  ) 