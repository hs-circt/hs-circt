add_circt_unittest(CIRCTXlnxTests
  LutNBuilderTest.cpp
  Lut1to6BuilderTest.cpp
)

target_link_libraries(CIRCTXlnxTests
  PUBLIC
  CIRCTXlnx
  CIRCTHW
)
