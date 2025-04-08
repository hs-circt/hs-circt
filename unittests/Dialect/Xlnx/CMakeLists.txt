add_circt_unittest(CIRCTXlnxTests
  LutBuilderTest.cpp
)

target_link_libraries(CIRCTXlnxTests
  PUBLIC
  CIRCTXlnx
  CIRCTHW
)
