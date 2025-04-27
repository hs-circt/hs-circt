add_circt_conversion_library(CoreDSPMapping
CoreDSPMapping.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTDebug
  CIRCTHW
  CIRCTTransforms
  MLIRTransforms
)