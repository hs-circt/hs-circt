//===- CombToLutTest.cpp - Comb to Lut unit tests ===------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for the `FVInt` class.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HWToNetlist.h"
#include "circt/Synthesis/CombToNetlist.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Xlnx/XlnxOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "gtest/gtest.h"

#include <kitty/kitty.hpp>
#include <mockturtle/io/write_dot.hpp>

using namespace mlir;
using namespace circt;
using namespace circt::hw;
using namespace circt::synthesis;

namespace {
class CombToLutTest : public testing::Test {
private:
  DialectRegistry registry;
  std::unique_ptr<MLIRContext> context;

protected:
  void SetUp() override {
    // Register dialects
    registry.insert<hw::HWDialect, comb::CombDialect, seq::SeqDialect,
                    xlnx::XlnxDialect>();
    context = std::make_unique<MLIRContext>(registry);
    // Ensure dialects are loaded
    context->loadDialect<hw::HWDialect, comb::CombDialect, seq::SeqDialect,
                         xlnx::XlnxDialect>();
    context->allowUnregisteredDialects();
  }

  ModuleOp createTinyModule1() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    auto topModule = builder.create<ModuleOp>();
    builder.setInsertionPointToStart(topModule.getBody());

    // Create type
    Type i1Type = builder.getI1Type();

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "TestModule"), ArrayRef<PortInfo>{});
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

    auto and1 = builder.create<comb::AndOp>(a, b);
    auto and2 = builder.create<comb::AndOp>(c, d);
    auto or1 = builder.create<comb::OrOp>(and1, and2);

    // Create output
    hwModule.appendOutput("out", or1.getResult());

    return topModule;
  }

  ModuleOp createTinyModule2() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    auto topModule = builder.create<ModuleOp>();
    builder.setInsertionPointToStart(topModule.getBody());

    // Create type
    Type i1Type = builder.getI1Type();

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "TestModule"), ArrayRef<PortInfo>{});
    hwModule.appendInput("a", i1Type);
    hwModule.appendInput("b", i1Type);
    hwModule.appendInput("c", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input port
    Value a = hwModule.getBodyBlock()->getArgument(0);
    Value b = hwModule.getBodyBlock()->getArgument(1);
    Value c = hwModule.getBodyBlock()->getArgument(2);

    auto and1 = builder.create<comb::AndOp>(a, b);
    auto and2 = builder.create<comb::AndOp>(b, c);
    auto or1 = builder.create<comb::OrOp>(and1, and2);

    // Create output
    hwModule.appendOutput("out", or1.getResult());

    return topModule;
  }

  ModuleOp createModuleWithReg() {
    ImplicitLocOpBuilder builder(UnknownLoc::get(context.get()), context.get());

    auto topModule = builder.create<ModuleOp>();
    builder.setInsertionPointToStart(topModule.getBody());

    // Create type
    Type i1Type = builder.getI1Type();

    // Create hardware module
    auto hwModule = builder.create<hw::HWModuleOp>(
        StringAttr::get(context.get(), "TestModule"), ArrayRef<PortInfo>{});
    hwModule.appendInput("a", i1Type);
    hwModule.appendInput("b", i1Type);
    hwModule.appendInput("c", i1Type);
    hwModule.appendInput("clk", seq::ClockType::get(builder.getContext()));
    hwModule.appendInput("rst", i1Type);

    // Create module body
    builder.setInsertionPointToStart(hwModule.getBodyBlock());

    // Get input port
    Value a = hwModule.getBodyBlock()->getArgument(0);
    Value b = hwModule.getBodyBlock()->getArgument(1);
    Value c = hwModule.getBodyBlock()->getArgument(2);
    Value clk = hwModule.getBodyBlock()->getArgument(3);

    auto regA = builder.create<seq::CompRegOp>(a, clk);
    auto regB = builder.create<seq::CompRegOp>(b, clk);

    auto and1 = builder.create<comb::AndOp>(regA, regB);

    auto regC = builder.create<seq::CompRegOp>(and1, clk);

    auto and2 = builder.create<comb::AndOp>(regC, c);

    // Create output
    hwModule.appendOutput("out", and2.getResult());

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
};

TEST_F(CombToLutTest, Tiny1) {
  auto module = createTinyModule1();
  std::string ir = verifyAndPrint(module);

  llvm::outs() << "UNITTEST: Tiny1\n" << ir << "\n";

  auto hwModule = module.lookupSymbol<hw::HWModuleOp>("TestModule");
  auto blocks = collectCombinationalBlocks(hwModule);
  int blockCnt = 0;
  for (const auto &it : blocks) {
    if (!it->isLeader())
      continue;

    ++blockCnt;

    auto opRange = blocks.members(*it);
    auto combBlock = CombinationalBlock(opRange.begin(), opRange.end());
    aig_network aig = convertBlockToAIG(combBlock);

    mapper::NaiveLutMapper naiveMapper;
    naiveMapper.setAIG(&aig);
    EXPECT_FALSE(failed(naiveMapper.map()));

    kitty::dynamic_truth_table tt(4);
    kitty::create_from_hex_string(tt, "f888");

    auto lut = naiveMapper.getResult();
    EXPECT_EQ(tt, lut.node_function(lut.po_at(0)));

    llvm::outs() << "Mapping success\n";

    OpBuilder builder(module);
    EXPECT_FALSE(failed(convertLUTToHW(builder, lut, combBlock)));
  }

  EXPECT_EQ(blockCnt, 1);

  llvm::outs() << "UNITTEST Tiny: Result IR\n"
               << verifyAndPrint(module) << "\n";
}

TEST_F(CombToLutTest, Tiny2) {
  auto module = createTinyModule2();
  std::string ir = verifyAndPrint(module);

  llvm::outs() << "UNITTEST: Tiny2\n" << ir << "\n";

  auto hwModule = module.lookupSymbol<hw::HWModuleOp>("TestModule");
  auto blocks = collectCombinationalBlocks(hwModule);
  int blockCnt = 0;
  for (const auto &it : blocks) {
    if (!it->isLeader())
      continue;

    ++blockCnt;

    OpBuilder builder(module);
    CombBlockConverter lutConv(hwModule, builder);
    mapper::NaiveLutMapper m;
    EXPECT_FALSE(failed(lutConv.combToLut(m)));

    // {
    //   kitty::static_truth_table<3> a, b, c, res;
    //   kitty::create_nth_var(a, 0);
    //   kitty::create_nth_var(b, 1);
    //   kitty::create_nth_var(c, 2);
    //   res = (a & b) | (b & c);
    //   llvm::outs() << "UNITTEST: Truth table\n" << kitty::to_hex(res) <<
    //   "\n";
    // }
    auto res = hwModule.getBodyBlock()->getTerminator()->getOperands()[0];
    auto lutOp = res.getDefiningOp<xlnx::XlnxLutNOp>();

    EXPECT_EQ(lutOp.getINIT(), 0xc8ul);
  }

  EXPECT_EQ(blockCnt, 1);

  llvm::outs() << "UNITTEST Tiny2: Result IR\n"
               << verifyAndPrint(module) << "\n";
}

TEST_F(CombToLutTest, Reg1) {
  auto module = createModuleWithReg();
  std::string ir = verifyAndPrint(module);

  llvm::outs() << "UNITTEST: Reg1\n" << ir << "\n";

  auto hwModule = module.lookupSymbol<hw::HWModuleOp>("TestModule");
  auto blocks = collectCombinationalBlocks(hwModule);
  int blockCnt = 0;
  for (const auto &it : blocks) {
    if (!it->isLeader())
      continue;

    ++blockCnt;

    OpBuilder builder(module);
    CombBlockConverter lutConv(hwModule, builder);
    mapper::NaiveLutMapper m;
    EXPECT_FALSE(failed(lutConv.combToLut(m)));
  }

  EXPECT_EQ(blockCnt, 2);

  llvm::outs() << "UNITTEST Reg1: Result IR\n"
               << verifyAndPrint(module) << "\n";
}
} // namespace