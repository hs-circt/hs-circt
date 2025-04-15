//===- HWToSV.cpp - HW To SV Conversion Pass ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main HW to SV Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HWToNetlist.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

// mockturtle includes
#include "mockturtle/algorithms/aig_balancing.hpp"
#include "mockturtle/algorithms/emap.hpp"
#include "mockturtle/algorithms/lut_mapper.hpp"
#include "mockturtle/generators/arithmetic.hpp"
#include "mockturtle/io/aiger_reader.hpp"
#include "mockturtle/io/genlib_reader.hpp"
#include "mockturtle/io/write_blif.hpp"
#include "mockturtle/io/write_verilog.hpp"
#include "mockturtle/networks/aig.hpp"
#include "mockturtle/networks/block.hpp"
#include "mockturtle/utils/name_utils.hpp"
#include "mockturtle/utils/tech_library.hpp"
#include "mockturtle/views/cell_view.hpp"
#include "mockturtle/views/depth_view.hpp"
#include "mockturtle/views/names_view.hpp"

namespace circt {
#define GEN_PASS_DEF_LOWERHWTONETLIST
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;

namespace {
struct HWToNetlistPass
    : public circt::impl::LowerHWToNetlistBase<HWToNetlistPass> {
  void runOnOperation() override;
};

void tryMockturtle() {
  using namespace mockturtle;

  aig_network aig;

  std::vector<aig_network::signal> a(2), b(2);
  std::generate(a.begin(), a.end(), [&aig]() { return aig.create_pi(); });
  std::generate(b.begin(), b.end(), [&aig]() { return aig.create_pi(); });
  auto carry = aig.create_pi();

  carry_ripple_adder_inplace(aig, a, b, carry);

  std::for_each(a.begin(), a.end(), [&](auto f) { aig.create_po(f); });
  aig.create_po(carry);

  const auto simm = simulate<kitty::static_truth_table<5u>>(aig);

  klut_network klut = lut_map(aig);
  write_blif(klut, "klut.blif");

  fmt::print("[i] Lut mapper success\n");
}

struct CombinationalBlock {
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
  DenseSet<Operation *> ops;
};

class CombLutMapper {
public:
  CombLutMapper(hw::HWModuleOp module) : module(module) {}
  LogicalResult map(ConversionTarget &target) {
    tryMockturtle();
    return success();
  }

private:
  SmallVector<CombinationalBlock> collectCombinationalBlocks() {}
  LogicalResult mapCombinationalBlock(const CombinationalBlock &block) {}

  hw::HWModuleOp module;
};

} // namespace

//===----------------------------------------------------------------------===//
// Conversion Infrastructure
//===----------------------------------------------------------------------===//
static void populateLegality(ConversionTarget &target) {
  target.addIllegalDialect<comb::CombDialect>();
  target.addLegalDialect<hw::HWDialect>();

  target.addLegalOp<comb::TruthTableOp>();
}

void HWToNetlistPass::runOnOperation() {
  MLIRContext &context = getContext();
  hw::HWModuleOp module = getOperation();

  ConversionTarget target(context);

  populateLegality(target);

  if (failed(CombLutMapper(module).map(target)))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// HW to SV Conversion Pass
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<hw::HWModuleOp>>
circt::createLowerHWToNetlistPass() {
  return std::make_unique<HWToNetlistPass>();
}
