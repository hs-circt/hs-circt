add_circt_unittest(CIRCTSynthesisTests
  CombToLutTest.cpp
)

target_link_libraries(CIRCTSynthesisTests
  PRIVATE
  CIRCTHW
  CIRCTComb
  CIRCTXlnx
  CIRCTSeq
  CIRCTHWToNetlist
  
  mockturtle
)
