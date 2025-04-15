//===- Lut1to6BuilderTest.cpp - Tests for LUT1-LUT6 operations ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off
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
#include "XlnxTestUtils.h"
// clang-format on

using namespace mlir;
using namespace circt;
using namespace circt::xlnx;
using namespace circt::hw;
using namespace circt::xlnx_test;

namespace {

// Test Auxiliary class, used to set up the test environment
class XlnxSpecificLutTest : public testing::Test {
protected:
  void SetUp() override {
    // Register dialects
    registry.insert<hw::HWDialect, xlnx::XlnxDialect>();
    context = std::make_unique<MLIRContext>(registry);
    // Ensure dialects are loaded
    context->loadDialect<hw::HWDialect, xlnx::XlnxDialect>();
    context->allowUnregisteredDialects();
  }

  // Create a module containing a LUT1
  ModuleOp createLut1Module(uint64_t initValue) {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top module
    auto topModule = builder.create<ModuleOp>();
    builder.setInsertionPointToStart(topModule.getBody());

    // Create type
    Type i1Type = builder.getI1Type();

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "Lut1Module"), ArrayRef<PortInfo>{});
    hwModule.appendInput("a", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input port
    Value a = hwModule.getBodyBlock()->getArgument(0);

    // Create LUT1 (buffer: 0x2 corresponds to binary 10, which means output 1
    // when input is 1)
    auto lut = builder.create<XlnxLut1Op>(initValue, a);

    // Create output
    hwModule.appendOutput("out", lut.getResult());

    return topModule;
  }

  // Create a module containing a LUT2
  ModuleOp createLut2Module(uint64_t initValue) {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top module
    auto topModule = builder.create<ModuleOp>();
    builder.setInsertionPointToStart(topModule.getBody());

    // Create type
    Type i1Type = builder.getI1Type();

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "Lut2Module"), ArrayRef<PortInfo>{});
    hwModule.appendInput("a", i1Type);
    hwModule.appendInput("b", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input port
    Value a = hwModule.getBodyBlock()->getArgument(0);
    Value b = hwModule.getBodyBlock()->getArgument(1);

    // Create LUT2 (AND gate: 0x8 corresponds to binary 1000)
    auto lut = builder.create<XlnxLut2Op>(initValue, a, b);

    // Create output
    hwModule.appendOutput("out", lut.getResult());

    return topModule;
  }

  // Create a module containing a LUT3
  ModuleOp createLut3Module(uint64_t initValue) {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top module
    auto topModule = builder.create<ModuleOp>();
    builder.setInsertionPointToStart(topModule.getBody());

    // Create type
    Type i1Type = builder.getI1Type();

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "Lut3Module"), ArrayRef<PortInfo>{});
    hwModule.appendInput("a", i1Type);
    hwModule.appendInput("b", i1Type);
    hwModule.appendInput("c", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input port
    Value a = hwModule.getBodyBlock()->getArgument(0);
    Value b = hwModule.getBodyBlock()->getArgument(1);
    Value c = hwModule.getBodyBlock()->getArgument(2);

    // Create LUT3 (majority voter: 0xE8 corresponds to binary 11101000)
    auto lut = builder.create<XlnxLut3Op>(initValue, a, b, c);

    // Create output
    hwModule.appendOutput("out", lut.getResult());

    return topModule;
  }

  // Create a module containing a LUT4
  ModuleOp createLut4Module(uint64_t initValue) {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top module
    auto topModule = builder.create<ModuleOp>();
    builder.setInsertionPointToStart(topModule.getBody());

    // Create type
    Type i1Type = builder.getI1Type();

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "Lut4Module"), ArrayRef<PortInfo>{});
    hwModule.appendInput("a", i1Type);
    hwModule.appendInput("b", i1Type);
    hwModule.appendInput("c", i1Type);
    hwModule.appendInput("d", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input port
    Value a = hwModule.getBodyBlock()->getArgument(0);
    Value b = hwModule.getBodyBlock()->getArgument(1);
    Value c = hwModule.getBodyBlock()->getArgument(2);
    Value d = hwModule.getBodyBlock()->getArgument(3);

    // Create LUT4 (example function: 0xFFCC, complex mode)
    auto lut = builder.create<XlnxLut4Op>(initValue, a, b, c, d);

    // Create output
    hwModule.appendOutput("out", lut.getResult());

    return topModule;
  }

  // Create a module containing a LUT5
  ModuleOp createLut5Module(uint64_t initValue) {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top module
    auto topModule = builder.create<ModuleOp>();
    builder.setInsertionPointToStart(topModule.getBody());

    // Create type
    Type i1Type = builder.getI1Type();

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "Lut5Module"), ArrayRef<PortInfo>{});
    hwModule.appendInput("a", i1Type);
    hwModule.appendInput("b", i1Type);
    hwModule.appendInput("c", i1Type);
    hwModule.appendInput("d", i1Type);
    hwModule.appendInput("e", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input port
    Value a = hwModule.getBodyBlock()->getArgument(0);
    Value b = hwModule.getBodyBlock()->getArgument(1);
    Value c = hwModule.getBodyBlock()->getArgument(2);
    Value d = hwModule.getBodyBlock()->getArgument(3);
    Value e = hwModule.getBodyBlock()->getArgument(4);

    // Create LUT5 (example function: 0xAAAA5555, alternating mode)
    auto lut = builder.create<XlnxLut5Op>(initValue, a, b, c, d, e);

    // Create output
    hwModule.appendOutput("out", lut.getResult());

    return topModule;
  }

  // Create a module containing a LUT6
  ModuleOp createLut6Module(uint64_t initValue) {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top module
    auto topModule = builder.create<ModuleOp>();
    builder.setInsertionPointToStart(topModule.getBody());

    // Create type
    Type i1Type = builder.getI1Type();

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "Lut6Module"), ArrayRef<PortInfo>{});
    hwModule.appendInput("a", i1Type);
    hwModule.appendInput("b", i1Type);
    hwModule.appendInput("c", i1Type);
    hwModule.appendInput("d", i1Type);
    hwModule.appendInput("e", i1Type);
    hwModule.appendInput("f", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input port
    Value a = hwModule.getBodyBlock()->getArgument(0);
    Value b = hwModule.getBodyBlock()->getArgument(1);
    Value c = hwModule.getBodyBlock()->getArgument(2);
    Value d = hwModule.getBodyBlock()->getArgument(3);
    Value e = hwModule.getBodyBlock()->getArgument(4);
    Value f = hwModule.getBodyBlock()->getArgument(5);

    // Create LUT6 (example function: 0x8000000000000000, only output 1 when all
    // inputs are 1)
    auto lut = builder.create<XlnxLut6Op>(initValue, a, b, c, d, e, f);

    // Create output
    hwModule.appendOutput("out", lut.getResult());

    return topModule;
  }

  // Create a module containing cascaded LUTs
  // Equivalent LUT expression: LUT3(LUT2(LUT1(a, 0x2), b, 0x8), c, d, 0xFE)
  ModuleOp createCascadedLutsModule() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top module
    auto topModule = builder.create<ModuleOp>();
    builder.setInsertionPointToStart(topModule.getBody());

    // Create type
    Type i1Type = builder.getI1Type();

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "CascadedLutsModule"),
        ArrayRef<PortInfo>{});
    hwModule.appendInput("a", i1Type);
    hwModule.appendInput("b", i1Type);
    hwModule.appendInput("c", i1Type);
    hwModule.appendInput("d", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input port
    Value a = hwModule.getBodyBlock()->getArgument(0);
    Value b = hwModule.getBodyBlock()->getArgument(1);
    Value c = hwModule.getBodyBlock()->getArgument(2);
    Value d = hwModule.getBodyBlock()->getArgument(3);

    // Cascade using different types of LUTs
    // LUT1 as buffer
    auto lut1 = builder.create<XlnxLut1Op>(0x2, a);

    // LUT2 as AND gate
    auto lut2 = builder.create<XlnxLut2Op>(0x8, lut1.getResult(), b);

    // LUT3 as OR gate
    auto lut3 = builder.create<XlnxLut3Op>(0xFE, lut2.getResult(), c, d);

    // Create output
    hwModule.appendOutput("out", lut3.getResult());

    return topModule;
  }

  // Create a module containing common logic gates
  // The order is: buffer, NOT gate, AND gate, OR gate, XOR gate
  ModuleOp createCommonGatesModule() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top module
    auto topModule = builder.create<ModuleOp>();
    builder.setInsertionPointToStart(topModule.getBody());

    // Create type
    Type i1Type = builder.getI1Type();

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "CommonGatesModule"),
        ArrayRef<PortInfo>{});
    hwModule.appendInput("a", i1Type);
    hwModule.appendInput("b", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input port
    Value a = hwModule.getBodyBlock()->getArgument(0);
    Value b = hwModule.getBodyBlock()->getArgument(1);

    // Create various logic gates
    // Buffer - LUT1 (buffer, INIT=0x2, binary 10)
    auto bufferLut = builder.create<XlnxLut1Op>(0x2, a);

    // NOT - LUT1 (NOT gate, INIT=0x1, binary 01)
    auto notLut = builder.create<XlnxLut1Op>(0x1, a);

    // AND - LUT2 (AND gate, INIT=0x8, binary 1000)
    auto andLut = builder.create<XlnxLut2Op>(0x8, a, b);

    // OR - LUT2 (OR gate, INIT=0xE, binary 1110)
    auto orLut = builder.create<XlnxLut2Op>(0xE, a, b);

    // XOR - LUT2 (XOR gate, INIT=0x6, binary 0110)
    auto xorLut = builder.create<XlnxLut2Op>(0x6, a, b);

    // Create output
    hwModule.appendOutput("buffer_out", bufferLut.getResult());
    hwModule.appendOutput("not_out", notLut.getResult());
    hwModule.appendOutput("and_out", andLut.getResult());
    hwModule.appendOutput("or_out", orLut.getResult());
    hwModule.appendOutput("xor_out", xorLut.getResult());

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

  DialectRegistry registry;
  std::unique_ptr<MLIRContext> context;
};

// Test LUT1 operation
TEST_F(XlnxSpecificLutTest, Lut1Test) {
  auto module = createLut1Module(0x2);
  std::string ir = verifyAndPrint(module);
  std::string expected =
      "module {\n"
      "  hw.module @Lut1Module(in %a : i1, out out : i1) {\n"
      "    %0 = xlnx.lut1(I0 : %a) {INIT = 2 : ui2} : i1 -> i1\n"
      "    hw.output %0 : i1\n"
      "  }\n"
      "}\n";
  EXPECT_EQ(canonizeIRString(expected), canonizeIRString(ir));
}

// Test LUT2 operation
TEST_F(XlnxSpecificLutTest, Lut2Test) {
  auto module = createLut2Module(0x8);
  std::string ir = verifyAndPrint(module);
  std::string expected =
      "module {\n"
      "  hw.module @Lut2Module(in %a : i1, in %b : i1, out out : i1) {\n"
      "    %0 = xlnx.lut2(I0 : %a, I1 : %b) {INIT = 8 : ui4} : i1, i1 -> i1\n"
      "    hw.output %0 : i1\n"
      "  }\n"
      "}\n";
  EXPECT_EQ(canonizeIRString(expected), canonizeIRString(ir));
}

// Test LUT3 operation
TEST_F(XlnxSpecificLutTest, Lut3Test) {
  auto module = createLut3Module(0xE8);
  std::string ir = verifyAndPrint(module);
  std::string expected =
      // clang-format off
      "module {\n"
      "  hw.module @Lut3Module(in %a : i1, in %b : i1, in %c : i1, out out : i1) {\n"
      "    %0 = xlnx.lut3(I0 : %a, I1 : %b, I2 : %c) {INIT = 232 : ui8} : i1, i1, i1 -> i1\n"
      "    hw.output %0 : i1\n"
      "  }\n"
      "}\n"
      // clang-format on
      ;
  EXPECT_EQ(canonizeIRString(expected), canonizeIRString(ir));
}

// Test LUT4 operation
TEST_F(XlnxSpecificLutTest, Lut4Test) {
  auto module = createLut4Module(0xFFCC);
  std::string ir = verifyAndPrint(module);
  std::string expected =
      // clang-format off
      "module {\n"
      "  hw.module @Lut4Module(in %a : i1, in %b : i1, in %c : i1, in %d : i1, out out : i1) {\n"
      "    %0 = xlnx.lut4(I0 : %a, I1 : %b, I2 : %c, I3 : %d) {INIT = 65484 : ui16} : i1, i1, i1, i1 -> i1\n"
      "    hw.output %0 : i1\n"
      "  }\n"
      "}\n"
      // clang-format on
      ;
  EXPECT_EQ(canonizeIRString(expected), canonizeIRString(ir));
}

// Test LUT5 operation
TEST_F(XlnxSpecificLutTest, Lut5Test) {
  auto module = createLut5Module(2863311530);
  std::string ir = verifyAndPrint(module);
  std::string expected =
      // clang-format off
      "module {\n"
      "  hw.module @Lut5Module(in %a : i1, in %b : i1, in %c : i1, in %d : i1, in %e : i1, out out : i1) {\n"
      "    %0 = xlnx.lut5(I0 : %a, I1 : %b, I2 : %c, I3 : %d, I4 : %e) {INIT = 2863311530 : ui32} : i1, i1, i1, i1, i1 -> i1\n"
      "    hw.output %0 : i1\n"
      "  }\n"
      "}\n"
      // clang-format on
      ;
  EXPECT_EQ(canonizeIRString(expected), canonizeIRString(ir));
}

// Test LUT6 operation
TEST_F(XlnxSpecificLutTest, Lut6Test) {
  auto module = createLut6Module(0x8000000000000000);
  std::string ir = verifyAndPrint(module);
  std::string expected =
      // clang-format off
      "module {\n"
      "  hw.module @Lut6Module(in %a : i1, in %b : i1, in %c : i1, in %d : i1, in %e : i1, in %f : i1, out out : i1) {\n"
      "    %0 = xlnx.lut6(I0 : %a, I1 : %b, I2 : %c, I3 : %d, I4 : %e, I5 : %f) {INIT = 9223372036854775808 : ui64} : i1, i1, i1, i1, i1, i1 -> i1\n"
      "    hw.output %0 : i1\n"
      "  }\n"
      "}\n"
      // clang-format on
      ;
  EXPECT_EQ(canonizeIRString(expected), canonizeIRString(ir));
}

// Test cascaded LUT operation
TEST_F(XlnxSpecificLutTest, CascadedLutsTest) {
  auto module = createCascadedLutsModule();
  std::string ir = verifyAndPrint(module);
  std::string expected =
      // clang-format off
      "module {\n"
      "  hw.module @CascadedLutsModule(in %a : i1, in %b : i1, in %c : i1, in %d : i1, out out : i1) {\n"
      "    %0 = xlnx.lut1(I0 : %a) {INIT = 2 : ui2} : i1 -> i1\n"
      "    %1 = xlnx.lut2(I0 : %0, I1 : %b) {INIT = 8 : ui4} : i1, i1 -> i1\n"
      "    %2 = xlnx.lut3(I0 : %1, I1 : %c, I2 : %d) {INIT = 254 : ui8} : i1, i1, i1 -> i1\n"
      "    hw.output %2 : i1\n"
      "  }\n"
      "}\n"
      // clang-format on
      ;
  EXPECT_EQ(canonizeIRString(expected), canonizeIRString(ir));
}

// Test common logic gates
TEST_F(XlnxSpecificLutTest, CommonGatesTest) {
  auto module = createCommonGatesModule();
  std::string ir = verifyAndPrint(module);
  std::string expected =
      // clang-format off
      "module {\n"
      "  hw.module @CommonGatesModule(in %a : i1, in %b : i1, out buffer_out : i1, out not_out : i1, out and_out : i1, out or_out : i1, out xor_out : i1) {\n"
      "    %0 = xlnx.lut1(I0 : %a) {INIT = 2 : ui2} : i1 -> i1\n"
      "    %1 = xlnx.lut1(I0 : %a) {INIT = 1 : ui2} : i1 -> i1\n"
      "    %2 = xlnx.lut2(I0 : %a, I1 : %b) {INIT = 8 : ui4} : i1, i1 -> i1\n"
      "    %3 = xlnx.lut2(I0 : %a, I1 : %b) {INIT = 14 : ui4} : i1, i1 -> i1\n"
      "    %4 = xlnx.lut2(I0 : %a, I1 : %b) {INIT = 6 : ui4} : i1, i1 -> i1\n"
      "    hw.output %0, %1, %2, %3, %4 : i1, i1, i1, i1, i1\n"
      "  }\n"
      "}\n"
      // clang-format on
      ;
  EXPECT_EQ(canonizeIRString(expected), canonizeIRString(ir));
}

} // namespace
