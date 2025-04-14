set(LLVM_LINK_COMPONENTS
  Support
)

add_circt_tool(circt-logic-syn
  circt-logic-syn.cpp
)

target_link_libraries(circt-logic-syn PRIVATE
  CIRCTComb
  CIRCTHW
  CIRCTSeq

  CIRCTSV
  CIRCTSupport

  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib

  PRIVATE
  mockturtle
)

# get_property(MOCKTURTLE_INC_EXPORTS GLOBAL PROPERTY MOCKTURTLE_INC_EXPORTS)
# target_include_directories(circt-logic-syn SYSTEM PRIVATE ${MOCKTURTLE_INC_EXPORTS})

add_mockturtle_include_directories(circt-logic-syn)

llvm_update_compile_flags(circt-logic-syn)
mlir_check_all_link_libraries(circt-verilog)
