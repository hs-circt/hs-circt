//===- FDREBuilderTest.cpp - Tests for FDRE operations --------------------===//
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
class XlnxFDRETest : public testing::Test {
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
  // module BasicFDREModule(input clock, input ce, input r, input d, output q);
  //   FDRE fdre_inst (
  //     .C(clock),
  //     .CE(ce),
  //     .R(r), // Changed from S
  //     .D(d),
  //     .Q(q)
  //   );
  // endmodule
  ModuleOp createBasicFDREModule() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top-level module
    auto topModule = builder.create<ModuleOp>();

    // Set insertion point to top-level module
    builder.setInsertionPointToStart(topModule.getBody());

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "BasicFDREModule"),
        ArrayRef<PortInfo>{});

    // Create types
    Type i1Type = builder.getI1Type();
    Type clockType = ClockType::get(context.get());

    // Append input ports
    hwModule.appendInput("clock", clockType);
    hwModule.appendInput("ce", i1Type);
    hwModule.appendInput("r", i1Type); // Changed from s
    hwModule.appendInput("d", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input ports
    Value clock = hwModule.getBodyBlock()->getArgument(0);
    Value ce = hwModule.getBodyBlock()->getArgument(1);
    Value r = hwModule.getBodyBlock()->getArgument(2); // Changed from s
    Value d = hwModule.getBodyBlock()->getArgument(3);

    // Create FDRE
    auto fdre = builder.create<XlnxFDREOp>(clock, ce, r, d); // Changed op type

    // Create output
    hwModule.appendOutput("q", fdre.getResult());

    return topModule;
  }

  // Verilog equivalent:
  // module FDREWithAttributesModule(input clock, input ce, input r, input d,
  // output q);
  //   FDRE #(
  //     .INIT(1'b1),         // Non-default INIT
  //     .IS_C_INVERTED(1'b1),
  //     .IS_R_INVERTED(1'b1), // Changed from IS_S_INVERTED
  //     .IS_D_INVERTED(1'b1)
  //   ) fdre_inst (
  //     .C(clock),
  //     .CE(ce),
  //     .R(r),             // Changed from S
  //     .D(d),
  //     .Q(q)
  //   );
  // endmodule
  ModuleOp createFDREWithAttributesModule() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top-level module
    auto topModule = builder.create<ModuleOp>();

    // Set insertion point to top-level module
    builder.setInsertionPointToStart(topModule.getBody());

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "FDREWithAttributesModule"),
        ArrayRef<PortInfo>{});

    // Create types
    Type i1Type = builder.getI1Type();
    Type clockType = ClockType::get(context.get());

    // Append input ports
    hwModule.appendInput("clock", clockType);
    hwModule.appendInput("ce", i1Type);
    hwModule.appendInput("r", i1Type); // Changed from s
    hwModule.appendInput("d", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input ports
    Value clock = hwModule.getBodyBlock()->getArgument(0);
    Value ce = hwModule.getBodyBlock()->getArgument(1);
    Value r = hwModule.getBodyBlock()->getArgument(2); // Changed from s
    Value d = hwModule.getBodyBlock()->getArgument(3);

    // Create FDRE with attributes
    auto fdre = builder.create<XlnxFDREOp>(
        clock, ce, r, d, // Changed op type and r input
        builder.getIntegerAttr(builder.getIntegerType(1, false), 1), // INIT = 1 (non-default)
        builder.getIntegerAttr(builder.getIntegerType(1, false), 1), // IS_C_INVERTED = 1
        builder.getIntegerAttr(builder.getIntegerType(1, false), 1), // IS_D_INVERTED = 1
        builder.getIntegerAttr(builder.getIntegerType(1, false), 1) // IS_R_INVERTED = 1
    );

    // Create output
    hwModule.appendOutput("q", fdre.getResult());

    return topModule;
  }

  // Verilog equivalent:
  // module FDRECounterModule(input clock, input ce, input r, output [3:0]
  // count);
  //   reg [3:0] count_reg;
  //   wire [3:0] next_count = count_reg + 4'b1;
  //
  //   FDRE #(.INIT(1'b0)) count_ff0 (.C(clock), .CE(ce), .R(r), // Changed from S
  //   .D(next_count[0]), .Q(count_reg[0])); FDRE #(.INIT(1'b0)) count_ff1
  //   (.C(clock), .CE(ce), .R(r), .D(next_count[1]), .Q(count_reg[1])); // Changed from S
  //   FDRE #(.INIT(1'b0)) count_ff2 (.C(clock), .CE(ce), .R(r), // Changed from S
  //   .D(next_count[2]), .Q(count_reg[2])); FDRE #(.INIT(1'b0)) count_ff3
  //   (.C(clock), .CE(ce), .R(r), .D(next_count[3]), .Q(count_reg[3])); // Changed from S
  //
  //   assign count = count_reg;
  // endmodule
  ModuleOp createFDRECounterModule() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top-level module
    auto topModule = builder.create<ModuleOp>();

    // Set insertion point to top-level module
    builder.setInsertionPointToStart(topModule.getBody());

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "FDRECounterModule"),
        ArrayRef<PortInfo>{});

    // Create types
    Type i1Type = builder.getI1Type();
    Type i4Type = builder.getIntegerType(4);
    Type clockType = ClockType::get(context.get());

    // Append input ports
    hwModule.appendInput("clock", clockType);
    hwModule.appendInput("ce", i1Type);
    hwModule.appendInput("r", i1Type); // Changed from s

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input ports
    Value clock = hwModule.getBodyBlock()->getArgument(0);
    Value ce = hwModule.getBodyBlock()->getArgument(1);
    Value r = hwModule.getBodyBlock()->getArgument(2); // Changed from s

    // Constants
    Value c1_i4 = builder.create<hw::ConstantOp>(i4Type, 1);
    Value placeholderD = builder.create<hw::ConstantOp>(i1Type, 0);

    // Create 4 FDRE flip-flops for a 4-bit counter
    SmallVector<XlnxFDREOp, 4> counterOps; // Changed op type
    SmallVector<Value, 4> counterResults;

    // Create FDREs with placeholder D inputs first
    // Default INIT for FDRE is 0
    for (int i = 0; i < 4; i++) {
      auto fdreOp = builder.create<XlnxFDREOp>(clock, ce, r, placeholderD); // Changed op type and r input
      counterOps.push_back(fdreOp);
      counterResults.push_back(fdreOp.getResult());
    }

    // Form the current counter value from FDRE outputs
    Value counterVal = builder.create<comb::ConcatOp>(counterResults);

    // Compute next counter value (current + 1)
    Value nextVal = builder.create<comb::AddOp>(counterVal, c1_i4);

    // Extract bits for next state
    SmallVector<Value, 4> nextBits;
    for (int i = 0; i < 4; i++) {
      nextBits.push_back(builder.create<comb::ExtractOp>(nextVal, i, 1));
    }

    // Update the D inputs of the FDRE ops using the computed nextBits
    for (int i = 0; i < 4; i++) {
      counterOps[i].getOperation()->setOperand(3, nextBits[i]);
    }

    // Create output using the computed counter value
    hwModule.appendOutput("count", counterVal);

    return topModule;
  }

  // Verilog equivalent:
  // module FDREToggleModule(input clock, input ce, input r, input toggle,
  // output q);
  //   reg state_reg;
  //   wire next_state;
  //   wire should_toggle = toggle & ce;
  //   wire toggled_state = state_reg ^ 1'b1;
  //
  //   assign next_state = should_toggle ? toggled_state : state_reg;
  //
  //   FDRE #(.INIT(1'b0)) toggle_ff ( // Default INIT for FDRE is 0
  //     .C(clock),
  //     .CE(ce),
  //     .R(r),          // Changed from S
  //     .D(next_state),
  //     .Q(state_reg)
  //   );
  //
  //   assign q = state_reg;
  // endmodule
  ModuleOp createFDREToggleModule() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    // Create top-level module
    auto topModule = builder.create<ModuleOp>();

    // Set insertion point to top-level module
    builder.setInsertionPointToStart(topModule.getBody());

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "FDREToggleModule"),
        ArrayRef<PortInfo>{});

    // Create types
    Type i1Type = builder.getI1Type();
    Type clockType = ClockType::get(context.get());

    // Append input ports
    hwModule.appendInput("clock", clockType);
    hwModule.appendInput("ce", i1Type);
    hwModule.appendInput("r", i1Type); // Changed from s
    hwModule.appendInput("toggle", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input ports
    Value clock = hwModule.getBodyBlock()->getArgument(0);
    Value ce = hwModule.getBodyBlock()->getArgument(1);
    Value r = hwModule.getBodyBlock()->getArgument(2); // Changed from s
    Value toggle = hwModule.getBodyBlock()->getArgument(3);

    // Constants
    Value c1_i1 = builder.create<hw::ConstantOp>(i1Type, 1);
    Value placeholderD_toggle = builder.create<hw::ConstantOp>(i1Type, 0);

    // Create toggle flip-flop state register using FDRE
    // Default INIT for FDRE is 0
    auto state = builder.create<XlnxFDREOp>(clock, ce, r, placeholderD_toggle); // Changed op type and r input

    // Calculate the should_toggle condition
    auto should_toggle = builder.create<comb::AndOp>(toggle, ce);

    // Calculate the toggled state
    auto toggled_state = builder.create<comb::XorOp>(state.getResult(), c1_i1);

    // Calculate the next state based on the toggle condition
    auto next_state = builder.create<comb::MuxOp>(should_toggle, toggled_state,
                                                  state.getResult());

    // Connect the next_state to the FDRE D input
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

// Test the basic construction of FDRE
TEST_F(XlnxFDRETest, BasicFDRE) {
  auto module = createBasicFDREModule();
  std::string generatedIR = verifyAndPrint(module);

  std::string expectedIR = R"(module {
  hw.module @BasicFDREModule(in %clock : !seq.clock, in %ce : i1, in %r : i1, in %d : i1, out q : i1) {
    %0 = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %d) : !seq.clock, i1, i1, i1 -> i1
    hw.output %0 : i1
  }
})";

  // Canonize both strings to ignore SSA value name differences and whitespace
  // variations
  std::string canonGenerated = canonizeIRString(generatedIR);
  std::string canonExpected = canonizeIRString(expectedIR);

  EXPECT_EQ(canonGenerated, canonExpected);
}

// Test the construction of FDRE with attributes
TEST_F(XlnxFDRETest, FDREWithAttributes) {
  auto module = createFDREWithAttributesModule();
  std::string generatedIR = verifyAndPrint(module);

  // Default: INIT=0, IS_C_INVERTED=0, IS_D_INVERTED=0, IS_R_INVERTED=0
  // Set:     INIT=1, IS_C_INVERTED=1, IS_D_INVERTED=1, IS_R_INVERTED=1
  std::string expectedIR = R"(module {
  hw.module @FDREWithAttributesModule(in %clock : !seq.clock, in %ce : i1, in %r : i1, in %d : i1, out q : i1) {
    %0 = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %d) {INIT = 1 : ui1, IS_C_INVERTED = 1 : ui1, IS_D_INVERTED = 1 : ui1, IS_R_INVERTED = 1 : ui1} : !seq.clock, i1, i1, i1 -> i1
    hw.output %0 : i1
  }
})";

  // Canonize both strings to ignore SSA value name differences and whitespace
  // variations
  std::string canonGenerated = canonizeIRString(generatedIR);
  std::string canonExpected = canonizeIRString(expectedIR);

  EXPECT_EQ(canonGenerated, canonExpected);
}

// Test the construction of a 4-bit counter using FDRE
TEST_F(XlnxFDRETest, FDRECounter) {
  auto module = createFDRECounterModule();
  std::string generatedIR = verifyAndPrint(module);

  std::string expectedIR = R"(module {
  hw.module @FDRECounterModule(in %clock : !seq.clock, in %ce : i1, in %r : i1, out count : i4) {
    %c1_i4 = hw.constant 1 : i4
    %false = hw.constant false
    %0 = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %6) : !seq.clock, i1, i1, i1 -> i1
    %1 = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %7) : !seq.clock, i1, i1, i1 -> i1
    %2 = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %8) : !seq.clock, i1, i1, i1 -> i1
    %3 = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %9) : !seq.clock, i1, i1, i1 -> i1
    %4 = comb.concat %0, %1, %2, %3 : i1, i1, i1, i1
    %5 = comb.add %4, %c1_i4 : i4
    %6 = comb.extract %5 from 0 : (i4) -> i1
    %7 = comb.extract %5 from 1 : (i4) -> i1
    %8 = comb.extract %5 from 2 : (i4) -> i1
    %9 = comb.extract %5 from 3 : (i4) -> i1
    hw.output %4 : i4
  }
})";

  // Canonize both strings to ignore SSA value name differences and whitespace
  // variations
  std::string canonGenerated = canonizeIRString(generatedIR);
  std::string canonExpected = canonizeIRString(expectedIR);

  EXPECT_EQ(canonGenerated, canonExpected);
}

// Test the construction of a toggle flip-flop using FDRE
TEST_F(XlnxFDRETest, FDREToggle) {
  auto module = createFDREToggleModule();
  std::string generatedIR = verifyAndPrint(module);

  std::string expectedIR = R"(module {
  hw.module @FDREToggleModule(in %clock : !seq.clock, in %ce : i1, in %r : i1, in %toggle : i1, out q : i1) {
    %true = hw.constant true
    %false = hw.constant false
    %0 = xlnx.fdre(C : %clock, CE : %ce, R : %r, D : %3) : !seq.clock, i1, i1, i1 -> i1
    %1 = comb.and %toggle, %ce : i1
    %2 = comb.xor %0, %true : i1
    %3 = comb.mux %1, %2, %0 : i1
    hw.output %0 : i1
  }
})";

  // Canonize both strings to ignore SSA value name differences and whitespace
  // variations
  std::string canonGenerated = canonizeIRString(generatedIR);
  std::string canonExpected = canonizeIRString(expectedIR);

  EXPECT_EQ(canonGenerated, canonExpected);
}
} // end namespace
