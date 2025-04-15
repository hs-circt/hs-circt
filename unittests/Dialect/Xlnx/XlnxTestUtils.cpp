#include "XlnxTestUtils.h"

namespace circt {
namespace xlnx_test {

// Helper function to check if a character is visible (non-whitespace,
// non-control)
bool isVisibleChar(char c) {
  return !std::isspace(c) && !std::iscntrl(c) && c != '\n' && c != '\r' &&
         c != '\t';
}

// Canonize an IR string to ignore SSA value name differences and whitespace
// variations. This helps in comparing generated IR against expected IR strings.
std::string canonizeIRString(const std::string &ir) {
  std::stringstream canonization;
  bool lastWasVisible = false;
  for (char c : ir) {
    if (isVisibleChar(c)) {
      canonization << c;
      lastWasVisible = true;
    } else if (lastWasVisible) {
      canonization << ' ';
      lastWasVisible = false;
    }
  }
  // Trim leading and trailing whitespace
  std::string result = canonization.str();
  auto first = result.find_first_not_of(' ');
  if (first == std::string::npos)
    return ""; // Return empty if string is all whitespace or empty
  auto last = result.find_last_not_of(' ');
  return result.substr(first, last - first + 1);
}

} // namespace xlnx_test
} // namespace circt