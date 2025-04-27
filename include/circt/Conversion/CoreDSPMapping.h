#ifndef CIRCT_CONVERSION_COREDSPMAPPING_H
#define CIRCT_CONVERSION_COREDSPMAPPING_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace circt {

#define GEN_PASS_DECL_COREDSPMAPPING
#include "circt/Conversion/Passes.h.inc"

/// Create an Core Memory Mapping pass.
std::unique_ptr<OperationPass<ModuleOp>> createCoreDSPMappingPass();

} // namespace circt

#endif // CIRCT_CONVERSION_COREDSPMAPPING_H