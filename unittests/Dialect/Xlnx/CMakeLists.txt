add_circt_unittest(CIRCTXlnxTests
  XlnxTestUtils.cpp
  LutNBuilderTest.cpp
  Lut1to6BuilderTest.cpp
  MuxBuilderTest.cpp
  FDCEBuilderTest.cpp
  FDSEBuilderTest.cpp
  FDPEBuilderTest.cpp
)

target_link_libraries(CIRCTXlnxTests
  PUBLIC
  CIRCTXlnx
  CIRCTHW
  CIRCTSeq
  CIRCTComb
)
