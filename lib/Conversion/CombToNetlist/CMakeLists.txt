add_circt_conversion_library(CIRCTCombToNetlist
  CombToNetlist.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS
  CIRCTHW
  CIRCTXlnx
  CIRCTSupport
  MLIRLLVMCommonConversion
  MLIRTransforms
)

target_link_libraries(obj.CIRCTCombToNetlist PRIVATE mockturtle)

llvm_update_compile_flags(CIRCTCombToNetlist)
mlir_check_all_link_libraries(CIRCTCombToNetlist)