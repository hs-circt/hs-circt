#ifndef CIRCT_UNITTESTS_DIALECT_XLNX_TESTUTILS_H
#define CIRCT_UNITTESTS_DIALECT_XLNX_TESTUTILS_H

#include <string>
#include <sstream>
#include <cctype>

namespace circt {
namespace xlnx_test {

// Helper function to check if a character is visible (non-whitespace, non-control)
bool isVisibleChar(char c);

// Canonize an IR string to ignore SSA value name differences and whitespace variations.
// This helps in comparing generated IR against expected IR strings.
std::string canonizeIRString(const std::string &ir);

} // namespace xlnx_test
} // namespace circt

#endif // CIRCT_UNITTESTS_DIALECT_XLNX_TESTUTILS_H 