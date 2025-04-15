//===- MuxBuilderTest.cpp - Tests for MUX operations ----------------------===//
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
class XlnxMuxTest : public testing::Test {
protected:
  void SetUp() override {
    // Register dialects
    registry.insert<hw::HWDialect, xlnx::XlnxDialect>();
    context = std::make_unique<MLIRContext>(registry);
    // Ensure dialects are loaded
    context->loadDialect<hw::HWDialect, xlnx::XlnxDialect>();
    context->allowUnregisteredDialects();
  }

  // Create a module containing a MUXF7
  ModuleOp createMuxF7Module() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top module
    auto topModule = builder.create<ModuleOp>();
    builder.setInsertionPointToStart(topModule.getBody());

    // Create type
    Type i1Type = builder.getI1Type();

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "MuxF7Module"), ArrayRef<PortInfo>{});
    hwModule.appendInput("select", i1Type);
    hwModule.appendInput("input0", i1Type);
    hwModule.appendInput("input1", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input ports
    Value select = hwModule.getBodyBlock()->getArgument(0);
    Value input0 = hwModule.getBodyBlock()->getArgument(1);
    Value input1 = hwModule.getBodyBlock()->getArgument(2);

    // Create MUXF7
    auto mux = builder.create<XlnxMuxF7Op>(select, input0, input1);

    // Create output
    hwModule.appendOutput("out", mux.getResult());

    return topModule;
  }

  // Create a module containing a MUXF8
  ModuleOp createMuxF8Module() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top module
    auto topModule = builder.create<ModuleOp>();
    builder.setInsertionPointToStart(topModule.getBody());

    // Create type
    Type i1Type = builder.getI1Type();

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "MuxF8Module"), ArrayRef<PortInfo>{});
    hwModule.appendInput("select", i1Type);
    hwModule.appendInput("input0", i1Type);
    hwModule.appendInput("input1", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input ports
    Value select = hwModule.getBodyBlock()->getArgument(0);
    Value input0 = hwModule.getBodyBlock()->getArgument(1);
    Value input1 = hwModule.getBodyBlock()->getArgument(2);

    // Create MUXF8
    auto mux = builder.create<XlnxMuxF8Op>(select, input0, input1);

    // Create output
    hwModule.appendOutput("out", mux.getResult());

    return topModule;
  }

  // Create a module containing a MUXF9
  ModuleOp createMuxF9Module() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top module
    auto topModule = builder.create<ModuleOp>();
    builder.setInsertionPointToStart(topModule.getBody());

    // Create type
    Type i1Type = builder.getI1Type();

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "MuxF9Module"), ArrayRef<PortInfo>{});
    hwModule.appendInput("select", i1Type);
    hwModule.appendInput("input0", i1Type);
    hwModule.appendInput("input1", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input ports
    Value select = hwModule.getBodyBlock()->getArgument(0);
    Value input0 = hwModule.getBodyBlock()->getArgument(1);
    Value input1 = hwModule.getBodyBlock()->getArgument(2);

    // Create MUXF9
    auto mux = builder.create<XlnxMuxF9Op>(select, input0, input1);

    // Create output
    hwModule.appendOutput("out", mux.getResult());

    return topModule;
  }

  // Create a module containing cascaded MUXes
  ModuleOp createCascadedMuxModule() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top module
    auto topModule = builder.create<ModuleOp>();
    builder.setInsertionPointToStart(topModule.getBody());

    // Create type
    Type i1Type = builder.getI1Type();

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "CascadedMuxModule"),
        ArrayRef<PortInfo>{});
    hwModule.appendInput("select1", i1Type);
    hwModule.appendInput("select2", i1Type);
    hwModule.appendInput("select3", i1Type);
    hwModule.appendInput("a", i1Type);
    hwModule.appendInput("b", i1Type);
    hwModule.appendInput("c", i1Type);
    hwModule.appendInput("d", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input ports
    Value select1 = hwModule.getBodyBlock()->getArgument(0);
    Value select2 = hwModule.getBodyBlock()->getArgument(1);
    Value select3 = hwModule.getBodyBlock()->getArgument(2);
    Value a = hwModule.getBodyBlock()->getArgument(3);
    Value b = hwModule.getBodyBlock()->getArgument(4);
    Value c = hwModule.getBodyBlock()->getArgument(5);
    Value d = hwModule.getBodyBlock()->getArgument(6);

    // Create first level MUXes (MUXF7)
    auto mux1 = builder.create<XlnxMuxF7Op>(select1, a, b);
    auto mux2 = builder.create<XlnxMuxF7Op>(select1, c, d);

    // Create second level MUX (MUXF8)
    auto mux3 = builder.create<XlnxMuxF8Op>(select2, mux1, mux2);

    // Create output
    hwModule.appendOutput("out", mux3.getResult());

    return topModule;
  }

  // Create a complex MUX tree module
  ModuleOp createComplexMuxTreeModule() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top module
    auto topModule = builder.create<ModuleOp>();
    builder.setInsertionPointToStart(topModule.getBody());

    // Create type
    Type i1Type = builder.getI1Type();

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "ComplexMuxTreeModule"),
        ArrayRef<PortInfo>{});
    hwModule.appendInput("select1", i1Type);
    hwModule.appendInput("select2", i1Type);
    hwModule.appendInput("select3", i1Type);
    hwModule.appendInput("a", i1Type);
    hwModule.appendInput("b", i1Type);
    hwModule.appendInput("c", i1Type);
    hwModule.appendInput("d", i1Type);
    hwModule.appendInput("e", i1Type);
    hwModule.appendInput("f", i1Type);
    hwModule.appendInput("g", i1Type);
    hwModule.appendInput("h", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input ports
    Value select1 = hwModule.getBodyBlock()->getArgument(0);
    Value select2 = hwModule.getBodyBlock()->getArgument(1);
    Value select3 = hwModule.getBodyBlock()->getArgument(2);
    Value a = hwModule.getBodyBlock()->getArgument(3);
    Value b = hwModule.getBodyBlock()->getArgument(4);
    Value c = hwModule.getBodyBlock()->getArgument(5);
    Value d = hwModule.getBodyBlock()->getArgument(6);
    Value e = hwModule.getBodyBlock()->getArgument(7);
    Value f = hwModule.getBodyBlock()->getArgument(8);
    Value g = hwModule.getBodyBlock()->getArgument(9);
    Value h = hwModule.getBodyBlock()->getArgument(10);

    // First level - 4 MUXF7
    auto mux1 = builder.create<XlnxMuxF7Op>(select1, a, b);
    auto mux2 = builder.create<XlnxMuxF7Op>(select1, c, d);
    auto mux3 = builder.create<XlnxMuxF7Op>(select1, e, f);
    auto mux4 = builder.create<XlnxMuxF7Op>(select1, g, h);

    // Second level - 2 MUXF8
    auto mux5 = builder.create<XlnxMuxF8Op>(select2, mux1, mux2);
    auto mux6 = builder.create<XlnxMuxF8Op>(select2, mux3, mux4);

    // Third level - 1 MUXF9
    auto mux7 = builder.create<XlnxMuxF9Op>(select3, mux5, mux6);

    // Create output
    hwModule.appendOutput("out", mux7.getResult());

    return topModule;
  }

  // Create a module with LUTs and MUX
  ModuleOp createLutsWithMuxModule() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top module
    auto topModule = builder.create<ModuleOp>();
    builder.setInsertionPointToStart(topModule.getBody());

    // Create type
    Type i1Type = builder.getI1Type();

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "LutsWithMuxModule"),
        ArrayRef<PortInfo>{});
    hwModule.appendInput("select", i1Type);
    hwModule.appendInput("a", i1Type);
    hwModule.appendInput("b", i1Type);
    hwModule.appendInput("c", i1Type);
    hwModule.appendInput("d", i1Type);
    hwModule.appendInput("e", i1Type);
    hwModule.appendInput("f", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input ports
    Value select = hwModule.getBodyBlock()->getArgument(0);
    Value a = hwModule.getBodyBlock()->getArgument(1);
    Value b = hwModule.getBodyBlock()->getArgument(2);
    Value c = hwModule.getBodyBlock()->getArgument(3);
    Value d = hwModule.getBodyBlock()->getArgument(4);
    Value e = hwModule.getBodyBlock()->getArgument(5);
    Value f = hwModule.getBodyBlock()->getArgument(6);

    // Create two LUT6 instances
    auto lut1 = builder.create<XlnxLut6Op>(8589934590, a, b, c, d, e, f);
    auto lut2 = builder.create<XlnxLut6Op>(4294967295, a, b, c, d, e, f);

    // Create a MUXF7 to select between the two LUTs
    auto mux = builder.create<XlnxMuxF7Op>(select, lut1, lut2);

    // Create output
    hwModule.appendOutput("out", mux.getResult());

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

// Test MUXF7 operation
TEST_F(XlnxMuxTest, MuxF7Test) {
  auto module = createMuxF7Module();
  std::string ir = verifyAndPrint(module);
  std::string expected =
      // clang-format off
      "module {\n"
      "  hw.module @MuxF7Module(in %select : i1, in %input0 : i1, in %input1 : i1, out out : i1) {\n"
      "    %0 = xlnx.muxf7(S : %select, I0 : %input0, I1 : %input1) : i1, i1, i1 -> i1\n"
      "    hw.output %0 : i1\n"
      "  }\n"
      "}\n"
      // clang-format on
      ;
  EXPECT_EQ(canonizeIRString(expected), canonizeIRString(ir));
}

// Test MUXF8 operation
TEST_F(XlnxMuxTest, MuxF8Test) {
  auto module = createMuxF8Module();
  std::string ir = verifyAndPrint(module);
  std::string expected =
      // clang-format off
      "module {\n"
      "  hw.module @MuxF8Module(in %select : i1, in %input0 : i1, in %input1 : i1, out out : i1) {\n"
      "    %0 = xlnx.muxf8(S : %select, I0 : %input0, I1 : %input1) : i1, i1, i1 -> i1\n"
      "    hw.output %0 : i1\n"
      "  }\n"
      "}\n"
      // clang-format on
      ;
  EXPECT_EQ(canonizeIRString(expected), canonizeIRString(ir));
}

// Test MUXF9 operation
TEST_F(XlnxMuxTest, MuxF9Test) {
  auto module = createMuxF9Module();
  std::string ir = verifyAndPrint(module);
  std::string expected =
      // clang-format off
      "module {\n"
      "  hw.module @MuxF9Module(in %select : i1, in %input0 : i1, in %input1 : i1, out out : i1) {\n"
      "    %0 = xlnx.muxf9(S : %select, I0 : %input0, I1 : %input1) : i1, i1, i1 -> i1\n"
      "    hw.output %0 : i1\n"
      "  }\n"
      "}\n"
      // clang-format on
      ;
  EXPECT_EQ(canonizeIRString(expected), canonizeIRString(ir));
}

// Test cascaded MUX operations
TEST_F(XlnxMuxTest, CascadedMuxTest) {
  auto module = createCascadedMuxModule();
  std::string ir = verifyAndPrint(module);
  std::string expected =
      // clang-format off
      "module {\n"
      "  hw.module @CascadedMuxModule(in %select1 : i1, in %select2 : i1, in %select3 : i1, in %a : i1, in %b : i1, in %c : i1, in %d : i1, out out : i1) {\n"
      "    %0 = xlnx.muxf7(S : %select1, I0 : %a, I1 : %b) : i1, i1, i1 -> i1\n"
      "    %1 = xlnx.muxf7(S : %select1, I0 : %c, I1 : %d) : i1, i1, i1 -> i1\n"
      "    %2 = xlnx.muxf8(S : %select2, I0 : %0, I1 : %1) : i1, i1, i1 -> i1\n"
      "    hw.output %2 : i1\n"
      "  }\n"
      "}\n";
  // clang-format on
  EXPECT_EQ(canonizeIRString(expected), canonizeIRString(ir));
}

// Test complex MUX tree
TEST_F(XlnxMuxTest, ComplexMuxTreeTest) {
  auto module = createComplexMuxTreeModule();
  std::string ir = verifyAndPrint(module);
  std::string expected =
      // clang-format off
      "module {\n"
      "  hw.module @ComplexMuxTreeModule(in %select1 : i1, in %select2 : i1, in %select3 : i1, in %a : i1, in %b : i1, in %c : i1, in %d : i1, in %e : i1, in %f : i1, in %g : i1, in %h : i1, out out : i1) {\n"
      "    %0 = xlnx.muxf7(S : %select1, I0 : %a, I1 : %b) : i1, i1, i1 -> i1\n"
      "    %1 = xlnx.muxf7(S : %select1, I0 : %c, I1 : %d) : i1, i1, i1 -> i1\n"
      "    %2 = xlnx.muxf7(S : %select1, I0 : %e, I1 : %f) : i1, i1, i1 -> i1\n"
      "    %3 = xlnx.muxf7(S : %select1, I0 : %g, I1 : %h) : i1, i1, i1 -> i1\n"
      "    %4 = xlnx.muxf8(S : %select2, I0 : %0, I1 : %1) : i1, i1, i1 -> i1\n"
      "    %5 = xlnx.muxf8(S : %select2, I0 : %2, I1 : %3) : i1, i1, i1 -> i1\n"
      "    %6 = xlnx.muxf9(S : %select3, I0 : %4, I1 : %5) : i1, i1, i1 -> i1\n"
      "    hw.output %6 : i1\n"
      "  }\n"
      "}\n"
      // clang-format on
      ;
  EXPECT_EQ(canonizeIRString(expected), canonizeIRString(ir));
}

// Test LUTs with MUX
TEST_F(XlnxMuxTest, LutsWithMuxTest) {
  auto module = createLutsWithMuxModule();
  std::string ir = verifyAndPrint(module);
  std::string expected =
      // clang-format off
      "module {\n"
      "  hw.module @LutsWithMuxModule(in %select : i1, in %a : i1, in %b : i1, in %c : i1, in %d : i1, in %e : i1, in %f : i1, out out : i1) {\n"
      "    %0 = xlnx.lut6(I0 : %a, I1 : %b, I2 : %c, I3 : %d, I4 : %e, I5 : %f) {INIT = 8589934590 : ui64} : i1, i1, i1, i1, i1, i1 -> i1\n"
      "    %1 = xlnx.lut6(I0 : %a, I1 : %b, I2 : %c, I3 : %d, I4 : %e, I5 : %f) {INIT = 4294967295 : ui64} : i1, i1, i1, i1, i1, i1 -> i1\n"
      "    %2 = xlnx.muxf7(S : %select, I0 : %0, I1 : %1) : i1, i1, i1 -> i1\n"
      "    hw.output %2 : i1\n"
      "  }\n"
      "}\n"
      // clang-format on
      ;
  EXPECT_EQ(canonizeIRString(expected), canonizeIRString(ir));
}

} // namespace