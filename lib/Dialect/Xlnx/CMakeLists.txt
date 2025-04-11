add_circt_dialect_library(CIRCTXlnx
  XlnxOps.cpp
  XlnxDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Xlnx

  DEPENDS
  ${XLNX_INC_GEN_DEPS}

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  ) 