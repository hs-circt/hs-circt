//===- FDPEBuilderTest.cpp - Tests for FDPE operations --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off
#include "circt/Dialect/Xlnx/XlnxOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Dialect/Comb/CombOps.h"
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
using namespace circt::seq;
using namespace circt::comb;
using namespace circt::xlnx_test;

namespace {

// Test Auxiliary class, used to set up the test environment
class XlnxFDPETest : public testing::Test {
protected:
  void SetUp() override {
    // Register dialects
    registry.insert<hw::HWDialect, xlnx::XlnxDialect, seq::SeqDialect,
                    comb::CombDialect>();
    context = std::make_unique<MLIRContext>(registry);
    // Ensure dialects are loaded
    context->loadDialect<hw::HWDialect, xlnx::XlnxDialect, seq::SeqDialect,
                         comb::CombDialect>();
    context->allowUnregisteredDialects();
  }

  // Verilog equivalent:
  // module BasicFDPEModule(input clock, input ce, input pre, input d, output
  // q);
  //   FDPE fdpe_inst (
  //     .C(clock),
  //     .CE(ce),
  //     .PRE(pre),
  //     .D(d),
  //     .Q(q)
  //   );
  // endmodule
  ModuleOp createBasicFDPEModule() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top-level module
    auto topModule = builder.create<ModuleOp>();

    // Set insertion point to top-level module
    builder.setInsertionPointToStart(topModule.getBody());

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "BasicFDPEModule"),
        ArrayRef<PortInfo>{});

    // Create types
    Type i1Type = builder.getI1Type();
    Type clockType = ClockType::get(context.get());

    // Append input ports
    hwModule.appendInput("clock", clockType);
    hwModule.appendInput("ce", i1Type);
    hwModule.appendInput("pre", i1Type);
    hwModule.appendInput("d", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input ports
    Value clock = hwModule.getBodyBlock()->getArgument(0);
    Value ce = hwModule.getBodyBlock()->getArgument(1);
    Value pre = hwModule.getBodyBlock()->getArgument(2);
    Value d = hwModule.getBodyBlock()->getArgument(3);

    // Create FDPE
    auto fdpe = builder.create<XlnxFDPEOp>(clock, ce, pre, d);

    // Create output
    hwModule.appendOutput("q", fdpe.getResult());

    return topModule;
  }

  // Verilog equivalent:
  // module FDPEWithAttributesModule(input clock, input ce, input pre, input d,
  // output q);
  //   FDPE #(
  //     .INIT(1'b1),         // Non-default INIT
  //     .IS_C_INVERTED(1'b1),
  //     .IS_CE_INVERTED(1'b1),
  //     .IS_PRE_INVERTED(1'b1),
  //     .IS_D_INVERTED(1'b1)
  //   ) fdpe_inst (
  //     .C(clock),
  //     .CE(ce),
  //     .PRE(pre),
  //     .D(d),
  //     .Q(q)
  //   );
  // endmodule
  ModuleOp createFDPEWithAttributesModule() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top-level module
    auto topModule = builder.create<ModuleOp>();

    // Set insertion point to top-level module
    builder.setInsertionPointToStart(topModule.getBody());

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "FDPEWithAttributesModule"),
        ArrayRef<PortInfo>{});

    // Create types
    Type i1Type = builder.getI1Type();
    Type clockType = ClockType::get(context.get());

    // Append input ports
    hwModule.appendInput("clock", clockType);
    hwModule.appendInput("ce", i1Type);
    hwModule.appendInput("pre", i1Type);
    hwModule.appendInput("d", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input ports
    Value clock = hwModule.getBodyBlock()->getArgument(0);
    Value ce = hwModule.getBodyBlock()->getArgument(1);
    Value pre = hwModule.getBodyBlock()->getArgument(2);
    Value d = hwModule.getBodyBlock()->getArgument(3);

    // Collect operands
    SmallVector<Value> operands = {clock, ce, pre, d};

    // Create attributes array for non-default values
    SmallVector<NamedAttribute> attributes;
    attributes.push_back(builder.getNamedAttr("init", builder.getIntegerAttr(builder.getIntegerType(1, false), 1)));
    attributes.push_back(builder.getNamedAttr("is_c_inverted", builder.getIntegerAttr(builder.getIntegerType(1, false), 1)));
    attributes.push_back(builder.getNamedAttr("is_d_inverted", builder.getIntegerAttr(builder.getIntegerType(1, false), 1)));
    attributes.push_back(builder.getNamedAttr("is_ce_inverted", builder.getBoolAttr(true))); // Use BoolAttr as it's the defined type
    attributes.push_back(builder.getNamedAttr("is_pre_inverted", builder.getIntegerAttr(builder.getIntegerType(1, false), 1)));

    // Create FDPE using ValueRange and NamedAttribute array
    auto fdpe = builder.create<XlnxFDPEOp>(operands, attributes);

    // Create output
    hwModule.appendOutput("q", fdpe.getResult());

    return topModule;
  }

  // Verilog equivalent:
  // module FDPECounterModule(input clock, input ce, input pre, output [3:0]
  // count);
  //   reg [3:0] count_reg;
  //   wire [3:0] next_count = count_reg + 4'b1;
  //
  //   FDPE #(.INIT(1'b0)) count_ff0 (.C(clock), .CE(ce), .PRE(pre),
  //   .D(next_count[0]), .Q(count_reg[0])); FDPE #(.INIT(1'b0)) count_ff1
  //   (.C(clock), .CE(ce), .PRE(pre), .D(next_count[1]), .Q(count_reg[1]));
  //   FDPE #(.INIT(1'b0)) count_ff2 (.C(clock), .CE(ce), .PRE(pre),
  //   .D(next_count[2]), .Q(count_reg[2])); FDPE #(.INIT(1'b0)) count_ff3
  //   (.C(clock), .CE(ce), .PRE(pre), .D(next_count[3]), .Q(count_reg[3]));
  //
  //   assign count = count_reg;
  // endmodule
  ModuleOp createFDPECounterModule() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top-level module
    auto topModule = builder.create<ModuleOp>();

    // Set insertion point to top-level module
    builder.setInsertionPointToStart(topModule.getBody());

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "FDPECounterModule"),
        ArrayRef<PortInfo>{});

    // Create types
    Type i1Type = builder.getI1Type();
    Type i4Type = builder.getIntegerType(4);
    Type clockType = ClockType::get(context.get());

    // Append input ports
    hwModule.appendInput("clock", clockType);
    hwModule.appendInput("ce", i1Type);
    hwModule.appendInput("pre", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input ports
    Value clock = hwModule.getBodyBlock()->getArgument(0);
    Value ce = hwModule.getBodyBlock()->getArgument(1);
    Value pre = hwModule.getBodyBlock()->getArgument(2);

    // Constants
    Value c1_i4 = builder.create<hw::ConstantOp>(i4Type, 1);
    Value placeholderD = builder.create<hw::ConstantOp>(i1Type, 0);

    // Create 4 FDPE flip-flops for a 4-bit counter
    SmallVector<XlnxFDPEOp, 4> counterOps; // Changed op type
    SmallVector<Value, 4> counterResults;

    // Create FDPEs with placeholder D inputs first
    // INIT defaults to 1'b0 for FDPE
    for (int i = 0; i < 4; i++) {
      auto fdpeOp = builder.create<XlnxFDPEOp>(clock, ce, pre, placeholderD); // Changed op type
      counterOps.push_back(fdpeOp);
      counterResults.push_back(fdpeOp.getResult());
    }

    // Form the current counter value from FDPE outputs
    Value counterVal = builder.create<comb::ConcatOp>(counterResults);

    // Compute next counter value (current + 1)
    Value nextVal = builder.create<comb::AddOp>(counterVal, c1_i4);

    // Extract bits for next state
    SmallVector<Value, 4> nextBits;
    for (int i = 0; i < 4; i++) {
      nextBits.push_back(builder.create<comb::ExtractOp>(nextVal, i, 1));
    }

    // Update the D inputs of the FDPE ops using the computed nextBits
    for (int i = 0; i < 4; i++) {
      counterOps[i].getOperation()->setOperand(3, nextBits[i]);
    }

    // Create output
    hwModule.appendOutput("count", counterVal);

    return topModule;
  }

  // Verilog equivalent:
  // module FDPEToggleModule(input clock, input ce, input pre, output q);
  //   reg q_reg;
  //   wire next_q = ~q_reg;
  //
  //   FDPE #(.INIT(1'b0)) q_ff (.C(clock), .CE(ce), .PRE(pre), .D(next_q),
  //   .Q(q_reg));
  //
  //   assign q = q_reg;
  // endmodule
  ModuleOp createFDPEToggleModule() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top-level module
    auto topModule = builder.create<ModuleOp>();

    // Set insertion point to top-level module
    builder.setInsertionPointToStart(topModule.getBody());

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "FDPEToggleModule"),
        ArrayRef<PortInfo>{});

    // Create types
    Type i1Type = builder.getI1Type();
    Type clockType = ClockType::get(context.get());

    // Append input ports
    hwModule.appendInput("clock", clockType);
    hwModule.appendInput("ce", i1Type);
    hwModule.appendInput("pre", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input ports
    Value clock = hwModule.getBodyBlock()->getArgument(0);
    Value ce = hwModule.getBodyBlock()->getArgument(1);
    Value pre = hwModule.getBodyBlock()->getArgument(2);

    // Constant
    Value c1_i1 = builder.create<hw::ConstantOp>(i1Type, 1);
    Value placeholderD = builder.create<hw::ConstantOp>(i1Type, 0);

    // Create FDPE flip-flop for toggle logic
    // INIT defaults to 1'b0 for FDPE
    auto fdpeOp = builder.create<XlnxFDPEOp>(clock, ce, pre, placeholderD); // Changed op type

    // Compute next state (invert current state)
    Value currentQ = fdpeOp.getResult();
    Value nextQ = builder.create<comb::XorOp>(currentQ, c1_i1);

    // Update the D input of the FDPE op
    fdpeOp.getOperation()->setOperand(3, nextQ);

    // Create output
    hwModule.appendOutput("q", currentQ);

    return topModule;
  }

  // Helper function to verify and print the module
  std::string verifyAndPrint(Operation *op) {
    std::string result;
    llvm::raw_string_ostream os(result);
    if (failed(verify(op)))
      return "Verification failed";
    op->print(os);
    return result;
  }

  DialectRegistry registry;
  std::unique_ptr<MLIRContext> context;
};

// Test case for basic FDPE operation
TEST_F(XlnxFDPETest, BasicFDPE) {
  auto topModule = createBasicFDPEModule();
  ASSERT_TRUE(topModule);
  std::string expectedIR = R"(module {
  hw.module @BasicFDPEModule(in %clock : !seq.clock, in %ce : i1, in %pre : i1, in %d : i1, out q : i1) {
    %0 = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %d) : !seq.clock, i1, i1, i1 -> i1
    hw.output %0 : i1
  }
})";
  EXPECT_EQ(canonizeIRString(verifyAndPrint(topModule)), canonizeIRString(expectedIR));
}

// Test case for FDPE operation with attributes
TEST_F(XlnxFDPETest, FDPEWithAttributes) {
  auto topModule = createFDPEWithAttributesModule();
  ASSERT_TRUE(topModule);
  // FDPE: init=1, is_c_inverted=1, is_d_inverted=1, is_ce_inverted=1, is_pre_inverted=1
  // Default: init=0, is_c_inverted=0, is_d_inverted=0, is_ce_inverted=0, is_pre_inverted=0
  // Attribute order: init, is_c_inverted, is_d_inverted, is_ce_inverted, is_pre_inverted
  std::string expectedIR = R"(module {
  hw.module @FDPEWithAttributesModule(in %clock : !seq.clock, in %ce : i1, in %pre : i1, in %d : i1, out q : i1) {
    %0 = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %d) {init = 1 : ui1, is_c_inverted = 1 : ui1, is_ce_inverted = true, is_d_inverted = 1 : ui1, is_pre_inverted = 1 : ui1} : !seq.clock, i1, i1, i1 -> i1
    hw.output %0 : i1
  }
})";
  EXPECT_EQ(canonizeIRString(verifyAndPrint(topModule)), canonizeIRString(expectedIR));
}

// Test case for FDPE counter
TEST_F(XlnxFDPETest, FDPECounter) {
  auto topModule = createFDPECounterModule();
  ASSERT_TRUE(topModule);
  std::string expectedIR = R"(module {
  hw.module @FDPECounterModule(in %clock : !seq.clock, in %ce : i1, in %pre : i1, out count : i4) {
    %c1_i4 = hw.constant 1 : i4
    %false = hw.constant false
    %0 = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %6) : !seq.clock, i1, i1, i1 -> i1
    %1 = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %7) : !seq.clock, i1, i1, i1 -> i1
    %2 = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %8) : !seq.clock, i1, i1, i1 -> i1
    %3 = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %9) : !seq.clock, i1, i1, i1 -> i1
    %4 = comb.concat %0, %1, %2, %3 : i1, i1, i1, i1
    %5 = comb.add %4, %c1_i4 : i4
    %6 = comb.extract %5 from 0 : (i4) -> i1
    %7 = comb.extract %5 from 1 : (i4) -> i1
    %8 = comb.extract %5 from 2 : (i4) -> i1
    %9 = comb.extract %5 from 3 : (i4) -> i1
    hw.output %4 : i4
  }
})";
  EXPECT_EQ(canonizeIRString(verifyAndPrint(topModule)), canonizeIRString(expectedIR));
}

// Test case for FDPE toggle logic
TEST_F(XlnxFDPETest, FDPEToggle) {
  auto topModule = createFDPEToggleModule();
  ASSERT_TRUE(topModule);
  std::string expectedIR = R"(module {
  hw.module @FDPEToggleModule(in %clock : !seq.clock, in %ce : i1, in %pre : i1, out q : i1) {
    %true = hw.constant true
    %false = hw.constant false
    %0 = xlnx.fdpe(C : %clock, CE : %ce, PRE : %pre, D : %1) : !seq.clock, i1, i1, i1 -> i1
    %1 = comb.xor %0, %true : i1
    hw.output %0 : i1
  }
})";
  EXPECT_EQ(canonizeIRString(verifyAndPrint(topModule)), canonizeIRString(expectedIR));
}

} // namespace
