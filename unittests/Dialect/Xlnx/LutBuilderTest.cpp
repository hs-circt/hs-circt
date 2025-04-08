//===- LutBuilderTest.cpp - Tests for LutN builder methods -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Xlnx/XlnxOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace circt::xlnx;
using namespace circt::hw;

namespace {

// Test Auxiliary class, used to set up the test environment
class XlnxLutTest : public testing::Test {
protected:
  void SetUp() override {
    // Register dialects
    registry.insert<hw::HWDialect, xlnx::XlnxDialect>();
    context = std::make_unique<MLIRContext>(registry);
    // Ensure dialects are loaded
    context->loadDialect<hw::HWDialect, xlnx::XlnxDialect>();
    context->allowUnregisteredDialects();
  }

  // Create a module containing a single LUT
  ModuleOp createSingleLutModule() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());
    
    // Create top-level module
    auto topModule = builder.create<ModuleOp>();
    
    // Set insertion point to top-level module
    builder.setInsertionPointToStart(topModule.getBody());
    
    // Create type
    Type i1Type = builder.getI1Type();
    
    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
      StringAttr::get(context.get(), "SingleLutModule"),
      ArrayRef<PortInfo>{}
      );
    hwModule.appendInput("a", i1Type);
    hwModule.appendInput("b", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());
    
    // Get input ports
    Value a = hwModule.getBodyBlock()->getArgument(0);
    Value b = hwModule.getBodyBlock()->getArgument(1);
    
    // Create LUT (AND gate: 0x8 corresponds to binary 1000, meaning output 1 only when both inputs are 1)
    auto lut = builder.create<XlnxLutNOp>(ValueRange{a, b}, 0x8);
    
    // Create output
    hwModule.appendOutput("out", lut.getResult());
    
    return topModule;
  }

  // Create a module containing two cascaded LUTs
  ModuleOp createSerialLutsModule() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());
    
    // Create top-level module
    auto topModule = builder.create<ModuleOp>();
    
    // Set insertion point to top-level module
    builder.setInsertionPointToStart(topModule.getBody());
    
    // Create type
    Type i1Type = builder.getI1Type();
    
    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
      StringAttr::get(context.get(), "SerialLutsModule"),
      ArrayRef<PortInfo>{}
      );
    hwModule.appendInput("a", i1Type);
    hwModule.appendInput("b", i1Type);
    hwModule.appendInput("c", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());
    
    // Get input ports
    Value a = hwModule.getBodyBlock()->getArgument(0);
    Value b = hwModule.getBodyBlock()->getArgument(1);
    Value c = hwModule.getBodyBlock()->getArgument(2);
    
    // Create first LUT (AND gate: 0x8)
    auto lut1 = builder.create<XlnxLutNOp>(ValueRange{a, b}, 0x8);
    
    // Create second LUT, connected to the output of the first LUT (OR gate: 0xE)
    auto lut2 = builder.create<XlnxLutNOp>(ValueRange{lut1.getResult(), c}, 0xE);
    
    // Create output
    hwModule.appendOutput("out", lut2.getResult());
    
    return topModule;
  }

  // Create a module containing two parallel LUTs
  ModuleOp createParallelLutsModule() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());
    
    // Create top-level module
    auto topModule = builder.create<ModuleOp>();
    
    // Set insertion point to top-level module
    builder.setInsertionPointToStart(topModule.getBody());
    
    // Create type
    Type i1Type = builder.getI1Type();
    
    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
      StringAttr::get(context.get(), "ParallelLutsModule"),
      ArrayRef<PortInfo>{}
      );
    hwModule.appendInput("a", i1Type);
    hwModule.appendInput("b", i1Type);
    hwModule.appendInput("c", i1Type);
    hwModule.appendInput("d", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());
    
    // Get input ports
    Value a = hwModule.getBodyBlock()->getArgument(0);
    Value b = hwModule.getBodyBlock()->getArgument(1);
    Value c = hwModule.getBodyBlock()->getArgument(2);
    Value d = hwModule.getBodyBlock()->getArgument(3);
    
    // Create two parallel LUTs
    auto lut1 = builder.create<XlnxLutNOp>(ValueRange{a, b}, 0x8); // AND gate
    auto lut2 = builder.create<XlnxLutNOp>(ValueRange{c, d}, 0xE); // OR gate
    
    // Create output
    hwModule.appendOutput("out1", lut1.getResult());
    hwModule.appendOutput("out2", lut2.getResult());
    
    return topModule;
  }

  // Print operation for testing verification
  std::string verifyAndPrint(Operation *op) {
    std::string result;
    llvm::raw_string_ostream os(result);
    EXPECT_FALSE(failed(verify(op)));
    op->print(os);
    return result;
  }

  bool isVisibleChar(char c) {
    return !isspace(c) && !iscntrl(c) && c != '\n' && c != '\r' && c != '\t';
  }

  // Compare the expected and actual IR without whitespace
  void compareIR(const std::string &expected, const std::string &actual) {
    // Remove all whitespace from both strings
    std::stringstream canonizationExpected, canonizationActual;
    bool lastWasVisible = false;
    for (char c : expected) {
      if (isVisibleChar(c)) {
        canonizationExpected << c;
        lastWasVisible = true;
      } else if (lastWasVisible) {
        canonizationExpected << ' ';
        lastWasVisible = false;
      }
    }
    for (char c : actual) {
      if (isVisibleChar(c)) {
        canonizationActual << c;
        lastWasVisible = true;
      } else if (lastWasVisible) {
        canonizationActual << ' ';
        lastWasVisible = false;
      }
    }
    EXPECT_EQ(canonizationExpected.str(), canonizationActual.str());
  }

  DialectRegistry registry;
  std::unique_ptr<MLIRContext> context;
};

// Test the construction of a single LUT
TEST_F(XlnxLutTest, SingleLut) {
  auto module = createSingleLutModule();
  std::string ir = verifyAndPrint(module);

  std::string expected =
  "module {\n"
  "  hw.module @SingleLutModule(in %a : i1, in %b : i1, out out : i1) {\n"
  "    %0 = xlnx.lutn(%a, %b) {INIT = 8 : ui64} : (i1, i1) -> i1\n"
  "    hw.output %0 : i1\n"
  "  }\n"
  "}\n";

  // Compare the expected and actual IR without whitespace
  compareIR(expected, ir);
}

// Test the construction of cascaded LUTs
TEST_F(XlnxLutTest, SerialLuts) {
  auto module = createSerialLutsModule();
  std::string ir = verifyAndPrint(module);
  std::string expected =
  "module {\n"
  "  hw.module @SerialLutsModule(in %a : i1, in %b : i1, in %c : i1, out out : i1) {\n"
  "    %0 = xlnx.lutn(%a, %b) {INIT = 8 : ui64} : (i1, i1) -> i1\n"
  "    %1 = xlnx.lutn(%0, %c) {INIT = 14 : ui64} : (i1, i1) -> i1\n"
  "    hw.output %1 : i1\n"
  "  }\n"
  "}\n";

  // Compare the expected and actual IR without whitespace
  compareIR(expected, ir);
}


// Test the construction of parallel LUTs
TEST_F(XlnxLutTest, ParallelLuts) {
  auto module = createParallelLutsModule();
  std::string ir = verifyAndPrint(module);
  std::string expected =
  "module {\n"
  "  hw.module @ParallelLutsModule(in %a : i1, in %b : i1, in %c : i1, in %d : i1, out out1 : i1, out out2 : i1) {\n"
  "    %0 = xlnx.lutn(%a, %b) {INIT = 8 : ui64} : (i1, i1) -> i1\n"
  "    %1 = xlnx.lutn(%c, %d) {INIT = 14 : ui64} : (i1, i1) -> i1\n"
  "    hw.output %0, %1 : i1, i1\n"
  "  }\n"
  "}\n";
  
  // Compare the expected and actual IR without whitespace
  compareIR(expected, ir);
}

} // namespace 