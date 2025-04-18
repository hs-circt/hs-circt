add_circt_conversion_library(CIRCTCoreMemoryMapping
  CoreMemoryMapping.cpp

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
