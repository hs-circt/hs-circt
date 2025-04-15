add_circt_conversion_library(CIRCTHWToNetlist
  HWToNetlist.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS
  CIRCTHW
  CIRCTSupport
  MLIRLLVMCommonConversion
  MLIRTransforms
)

target_link_libraries(obj.CIRCTHWToNetlist PRIVATE mockturtle)

llvm_update_compile_flags(CIRCTHWToNetlist)
mlir_check_all_link_libraries(CIRCTHWToNetlist)