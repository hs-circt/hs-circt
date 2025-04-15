//===- FDCEBuilderTest.cpp - Tests for FDCE operations ---------------------===//
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
class XlnxFDCETest : public testing::Test {
protected:
  void SetUp() override {
    // Register dialects
    registry.insert<hw::HWDialect, xlnx::XlnxDialect, seq::SeqDialect, comb::CombDialect>();
    context = std::make_unique<MLIRContext>(registry);
    // Ensure dialects are loaded
    context->loadDialect<hw::HWDialect, xlnx::XlnxDialect, seq::SeqDialect, comb::CombDialect>();
    context->allowUnregisteredDialects();
  }

  // Verilog equivalent:
  // module BasicFDCEModule(input clock, input ce, input clr, input d, output q);
  //   FDCE fdce_inst (
  //     .C(clock),
  //     .CE(ce),
  //     .CLR(clr),
  //     .D(d),
  //     .Q(q)
  //   );
  // endmodule
  ModuleOp createBasicFDCEModule() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top-level module
    auto topModule = builder.create<ModuleOp>();

    // Set insertion point to top-level module
    builder.setInsertionPointToStart(topModule.getBody());

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "BasicFDCEModule"),
        ArrayRef<PortInfo>{});

    // Create types
    Type i1Type = builder.getI1Type();
    Type clockType = ClockType::get(context.get());

    // Append input ports
    hwModule.appendInput("clock", clockType);
    hwModule.appendInput("ce", i1Type);
    hwModule.appendInput("clr", i1Type);
    hwModule.appendInput("d", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input ports
    Value clock = hwModule.getBodyBlock()->getArgument(0);
    Value ce = hwModule.getBodyBlock()->getArgument(1);
    Value clr = hwModule.getBodyBlock()->getArgument(2);
    Value d = hwModule.getBodyBlock()->getArgument(3);

    // Create FDCE
    auto fdce = builder.create<XlnxFDCEOp>(clock, ce, clr, d);

    // Create output
    hwModule.appendOutput("q", fdce.getResult());

    return topModule;
  }

  // Verilog equivalent:
  // module FDCEWithAttributesModule(input clock, input ce, input clr, input d, output q);
  //   FDCE #(
  //     .INIT(1'b1),
  //     .IS_C_INVERTED(1'b1),
  //     .IS_CLR_INVERTED(1'b1),
  //     .IS_D_INVERTED(1'b1)
  //   ) fdce_inst (
  //     .C(clock),
  //     .CE(ce),
  //     .CLR(clr),
  //     .D(d),
  //     .Q(q)
  //   );
  // endmodule
  ModuleOp createFDCEWithAttributesModule() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top-level module
    auto topModule = builder.create<ModuleOp>();

    // Set insertion point to top-level module
    builder.setInsertionPointToStart(topModule.getBody());

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "FDCEWithAttributesModule"),
        ArrayRef<PortInfo>{});

    // Create types
    Type i1Type = builder.getI1Type();
    Type clockType = ClockType::get(context.get());

    // Append input ports
    hwModule.appendInput("clock", clockType);
    hwModule.appendInput("ce", i1Type);
    hwModule.appendInput("clr", i1Type);
    hwModule.appendInput("d", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input ports
    Value clock = hwModule.getBodyBlock()->getArgument(0);
    Value ce = hwModule.getBodyBlock()->getArgument(1);
    Value clr = hwModule.getBodyBlock()->getArgument(2);
    Value d = hwModule.getBodyBlock()->getArgument(3);

    // Create FDCE with attributes
    auto fdce = builder.create<XlnxFDCEOp>(
        clock, ce, clr, d,
        builder.getIntegerAttr(builder.getIntegerType(1, false), 1), // INIT
        builder.getIntegerAttr(builder.getIntegerType(1, false), 1), // IS_C_INVERTED
        builder.getIntegerAttr(builder.getIntegerType(1, false), 1), // IS_CLR_INVERTED
        builder.getIntegerAttr(builder.getIntegerType(1, false), 1)  // IS_D_INVERTED
    );

    // Create output
    hwModule.appendOutput("q", fdce.getResult());

    return topModule;
  }

  // Verilog equivalent:
  // module FDCECounterModule(input clock, input ce, input clr, output [3:0] count);
  //   reg [3:0] count_reg;
  //   wire [3:0] next_count = count_reg + 4'b1;
  //
  //   FDCE #(.INIT(1'b0)) count_ff0 (.C(clock), .CE(ce), .CLR(clr), .D(next_count[0]), .Q(count_reg[0]));
  //   FDCE #(.INIT(1'b0)) count_ff1 (.C(clock), .CE(ce), .CLR(clr), .D(next_count[1]), .Q(count_reg[1]));
  //   FDCE #(.INIT(1'b0)) count_ff2 (.C(clock), .CE(ce), .CLR(clr), .D(next_count[2]), .Q(count_reg[2]));
  //   FDCE #(.INIT(1'b0)) count_ff3 (.C(clock), .CE(ce), .CLR(clr), .D(next_count[3]), .Q(count_reg[3]));
  //
  //   assign count = count_reg;
  // endmodule
  ModuleOp createFDCECounterModule() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top-level module
    auto topModule = builder.create<ModuleOp>();

    // Set insertion point to top-level module
    builder.setInsertionPointToStart(topModule.getBody());

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "FDCECounterModule"),
        ArrayRef<PortInfo>{});

    // Create types
    Type i1Type = builder.getI1Type();
    Type i4Type = builder.getIntegerType(4);
    Type clockType = ClockType::get(context.get());

    // Append input ports
    hwModule.appendInput("clock", clockType);
    hwModule.appendInput("ce", i1Type);
    hwModule.appendInput("clr", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input ports
    Value clock = hwModule.getBodyBlock()->getArgument(0);
    Value ce = hwModule.getBodyBlock()->getArgument(1);
    Value clr = hwModule.getBodyBlock()->getArgument(2);

    // Constants
    Value c1_i4 = builder.create<hw::ConstantOp>(i4Type, 1);
    Value placeholderD = builder.create<hw::ConstantOp>(i1Type, 0);

    // Create 4 FDCE flip-flops for a 4-bit counter
    SmallVector<XlnxFDCEOp, 4> counterOps;
    SmallVector<Value, 4> counterResults;

    // Create FDCEs with placeholder D inputs first
    for (int i = 0; i < 4; i++) {
      auto fdceOp = builder.create<XlnxFDCEOp>(
          clock, ce, clr, placeholderD,
          builder.getIntegerAttr(builder.getIntegerType(1, false), 0));
      counterOps.push_back(fdceOp);
      counterResults.push_back(fdceOp.getResult());
    }

    // Form the current counter value from FDCE outputs
    Value counterVal = builder.create<comb::ConcatOp>(counterResults);

    // Compute next counter value (current + 1)
    Value nextVal = builder.create<comb::AddOp>(counterVal, c1_i4);

    // Extract bits for next state
    SmallVector<Value, 4> nextBits;
    for (int i = 0; i < 4; i++) {
      nextBits.push_back(builder.create<comb::ExtractOp>(nextVal, i, 1));
    }

    // Update the D inputs of the FDCE ops using the computed nextBits
    for (int i = 0; i < 4; i++) {
      counterOps[i].getOperation()->setOperand(3, nextBits[i]);
    }

    // Create output using the computed counter value
    hwModule.appendOutput("count", counterVal);

    return topModule;
  }

  // Verilog equivalent:
  // module FDCEToggleModule(input clock, input ce, input clr, input toggle, output q);
  //   reg state_reg;
  //   wire next_state;
  //   wire should_toggle = toggle & ce;
  //   wire toggled_state = state_reg ^ 1'b1;
  //
  //   assign next_state = should_toggle ? toggled_state : state_reg;
  //
  //   FDCE #(.INIT(1'b0)) toggle_ff (
  //     .C(clock),
  //     .CE(ce),
  //     .CLR(clr),
  //     .D(next_state),
  //     .Q(state_reg)
  //   );
  //
  //   assign q = state_reg;
  // endmodule
  ModuleOp createFDCEToggleModule() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top-level module
    auto topModule = builder.create<ModuleOp>();

    // Set insertion point to top-level module
    builder.setInsertionPointToStart(topModule.getBody());

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "FDCEToggleModule"),
        ArrayRef<PortInfo>{});

    // Create types
    Type i1Type = builder.getI1Type();
    Type clockType = ClockType::get(context.get());

    // Append input ports
    hwModule.appendInput("clock", clockType);
    hwModule.appendInput("ce", i1Type);
    hwModule.appendInput("clr", i1Type);
    hwModule.appendInput("toggle", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input ports
    Value clock = hwModule.getBodyBlock()->getArgument(0);
    Value ce = hwModule.getBodyBlock()->getArgument(1);
    Value clr = hwModule.getBodyBlock()->getArgument(2);
    Value toggle = hwModule.getBodyBlock()->getArgument(3);

    // Constants
    Value c1_i1 = builder.create<hw::ConstantOp>(i1Type, 1);
    Value placeholderD_toggle = builder.create<hw::ConstantOp>(i1Type, 0);

    // Create toggle flip-flop state register
    auto state = builder.create<XlnxFDCEOp>(clock, ce, clr,
                                           placeholderD_toggle,
                                           builder.getIntegerAttr(builder.getIntegerType(1, false), 0));

    // Calculate the should_toggle condition
    auto should_toggle = builder.create<comb::AndOp>(toggle, ce);

    // Calculate the toggled state
    auto toggled_state = builder.create<comb::XorOp>(state.getResult(), c1_i1);

    // Calculate the next state based on the toggle condition
    auto next_state = builder.create<comb::MuxOp>(should_toggle, toggled_state, state.getResult());

    // Connect the next_state to the FDCE D input
    state.getOperation()->setOperand(3, next_state);

    // Create output
    hwModule.appendOutput("q", state.getResult());

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

// Test the basic construction of FDCE
TEST_F(XlnxFDCETest, BasicFDCE) {
  auto module = createBasicFDCEModule();
  std::string generatedIR = verifyAndPrint(module);

  std::string expectedIR = R"(module {
  hw.module @BasicFDCEModule(in %clock : !seq.clock, in %ce : i1, in %clr : i1, in %d : i1, out q : i1) {
    %0 = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %d) : !seq.clock, i1, i1, i1 -> i1
    hw.output %0 : i1
  }
})";

  // Canonize both strings to ignore SSA value name differences and whitespace variations
  std::string canonGenerated = canonizeIRString(generatedIR);
  std::string canonExpected = canonizeIRString(expectedIR);

  EXPECT_EQ(canonGenerated, canonExpected);
}

// Test the construction of FDCE with attributes
TEST_F(XlnxFDCETest, FDCEWithAttributes) {
  auto module = createFDCEWithAttributesModule();
  std::string generatedIR = verifyAndPrint(module);

  std::string expectedIR = R"(module {
  hw.module @FDCEWithAttributesModule(in %clock : !seq.clock, in %ce : i1, in %clr : i1, in %d : i1, out q : i1) {
    %0 = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %d) {INIT = 1 : ui1, IS_C_INVERTED = 1 : ui1, IS_CLR_INVERTED = 1 : ui1, IS_D_INVERTED = 1 : ui1} : !seq.clock, i1, i1, i1 -> i1
    hw.output %0 : i1
  }
})";
  
  // Canonize both strings to ignore SSA value name differences and whitespace variations
  std::string canonGenerated = canonizeIRString(generatedIR);
  std::string canonExpected = canonizeIRString(expectedIR);

  EXPECT_EQ(canonGenerated, canonExpected);
}

// Test the construction of a 4-bit counter using FDCE
TEST_F(XlnxFDCETest, FDCECounter) {
  auto module = createFDCECounterModule();
  std::string generatedIR = verifyAndPrint(module);

  std::string expectedIR = R"(module {
  hw.module @FDCECounterModule(in %clock : !seq.clock, in %ce : i1, in %clr : i1, out %count : i4) {
    %c1_i4 = hw.constant 1 : i4
    %c0_i1 = hw.constant 0 : i1
    %fdce0 = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %d0) {INIT = 0 : i1} : !seq.clock, i1, i1, i1 -> i1
    %fdce1 = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %d1) {INIT = 0 : i1} : !seq.clock, i1, i1, i1 -> i1
    %fdce2 = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %d2) {INIT = 0 : i1} : !seq.clock, i1, i1, i1 -> i1
    %fdce3 = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %d3) {INIT = 0 : i1} : !seq.clock, i1, i1, i1 -> i1
    %counterVal = comb.concat %fdce0, %fdce1, %fdce2, %fdce3 : i1, i1, i1, i1 -> i4
    %nextVal = comb.add %counterVal, %c1_i4 : i4
    %d0 = comb.extract %nextVal from 0 : (i4) -> i1
    %d1 = comb.extract %nextVal from 1 : (i4) -> i1
    %d2 = comb.extract %nextVal from 2 : (i4) -> i1
    %d3 = comb.extract %nextVal from 3 : (i4) -> i1
    hw.output %counterVal : i4
  }
})";

  // Canonize both strings to ignore SSA value name differences and whitespace variations
  std::string canonGenerated = canonizeIRString(generatedIR);
  std::string canonExpected = canonizeIRString(expectedIR);

  EXPECT_EQ(canonGenerated, canonExpected);
}

// Test the construction of a toggle flip-flop using FDCE
TEST_F(XlnxFDCETest, FDCEToggle) {
  auto module = createFDCEToggleModule();
  std::string generatedIR = verifyAndPrint(module);

  std::string expectedIR = R"(module {
  hw.module @FDCEToggleModule(in %clock : !seq.clock, in %ce : i1, in %clr : i1, in %toggle : i1, out q : i1) {
    %true = hw.constant true
    %false = hw.constant false
    %0 = xlnx.fdce(C : %clock, CE : %ce, CLR : %clr, D : %3) : !seq.clock, i1, i1, i1 -> i1
    %1 = comb.and %toggle, %ce : i1
    %2 = comb.xor %0, %true : i1
    %3 = comb.mux %1, %2, %0 : i1
    hw.output %0 : i1
  }
})";

  // Canonize both strings to ignore SSA value name differences and whitespace variations
  std::string canonGenerated = canonizeIRString(generatedIR);
  std::string canonExpected = canonizeIRString(expectedIR);

  EXPECT_EQ(canonGenerated, canonExpected);
}
} // end namespace 