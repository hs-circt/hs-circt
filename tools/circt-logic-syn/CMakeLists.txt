set(LLVM_LINK_COMPONENTS
  Support
)

add_circt_tool(circt-logic-syn
  circt-logic-syn.cpp
)

llvm_update_compile_flags(circt-logic-syn)
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
)
