add_circt_conversion_library(XlnxToHW
  XlnxToHW.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core
  
  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTComb
  CIRCTSeq
  CIRCTXlnx
) 