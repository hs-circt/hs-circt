add_circt_conversion_library(CIRCTHWToNetlist
  HWToNetlist.cpp

  DEPENDS
  CIRCTConversionPassIncGen
  # mockturtle

  LINK_COMPONENTS
  Core

  LINK_LIBS
  CIRCTHW
  CIRCTSupport
  MLIRLLVMCommonConversion
  MLIRTransforms

  # dummy target, just for checking
  PUBLIC
  mockturtle
)

add_mockturtle_include_directories(CIRCTHWToNetlist)

llvm_update_compile_flags(CIRCTHWToNetlist)
mlir_check_all_link_libraries(CIRCTHWToNetlist)