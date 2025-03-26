add_circt_dialect_library(CIRCTGtech
  GtechDialect.cpp
  GtechOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Gtech

  DEPENDS
  MLIRGtechIncGen


  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR

)