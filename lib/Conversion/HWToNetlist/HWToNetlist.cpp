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
#include "circt/Dialect/Xlnx/XlnxOps.h"
#include "circt/Synthesis/CombToNetlist.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/TypeSwitch.h"

// mockturtle includes
#include "mockturtle/algorithms/aig_balancing.hpp"
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
using namespace circt::hw;
using namespace circt::synthesis;

namespace {
bool isCombOperation(Operation *op) {
  using namespace comb;
  // if (isa<hw::ConstantOp>(op))
  //   return true;

  if (!isa<comb::CombDialect>(op->getDialect()))
    return false;

  return TypeSwitch<Operation *, bool>(op)
      .Case<ConcatOp, ExtractOp, ReplicateOp, ParityOp>(
          [&](auto op) { return false; })
      .Default([&](Operation *op) { return true; });
}

struct AIGConverter {
  using aig_network = mockturtle::aig_network;
  using signal = mockturtle::aig_network::signal;

  AIGConverter(const CombinationalBlock &combBlock);

  void createPIs();
  void createPOs();

  aig_network &getAIG() { return aig; }
  signal buildTree(Operation *op);
  signal toAIGSignal(Operation *op);

private:
  mockturtle::aig_network aig;
  DenseMap<Value, mockturtle::aig_network::signal> valueToSignal;
  const CombinationalBlock &combBlock;
};

AIGConverter::AIGConverter(const CombinationalBlock &combBlock)
    : combBlock(combBlock) {
  createPIs();
  // TODO: traverse comb region in topological order
  for (Operation *op : combBlock.getOps()) {
    Value result = op->getResult(0);
    auto intType = dyn_cast<IntegerType>(result.getType());
    assert(intType.getWidth() == 1 && "Only support single-bit output");

    valueToSignal[result] = toAIGSignal(op);
  }

  createPOs();
}

void AIGConverter::createPIs() {
  for (auto [i, input] : enumerate(combBlock.getInputs())) {
    // llvm::outs() << "PI: " << input << "\n";
    valueToSignal[input] = aig.create_pi();
  }
}

void AIGConverter::createPOs() {
  for (auto op : combBlock.getOutputs()) {
    // llvm::outs() << "PO: " << op << "\n";
    aig.create_po(valueToSignal.at(op));
  }
}

mockturtle::aig_network::signal AIGConverter::toAIGSignal(Operation *op) {
  using namespace comb;
  assert(op->getNumResults() == 1 &&
         "For now, AIGConverter only support operation with one result");

  return buildTree(op);
}

// allow operation with inputs more then 2
mockturtle::aig_network::signal AIGConverter::buildTree(Operation *op) {
  // llvm::outs() << "buildTree: " << *op << "\n";
  std::deque<signal> que;
  for (Value operand : op->getOperands()) {
    // llvm::outs() << "operand: " << operand << "\n";
    if (auto constOp = operand.getDefiningOp<hw::ConstantOp>()) {
      bool val = constOp.getValue().isOne();
      que.push_back(aig.get_constant(val));
    } else
      que.push_back(valueToSignal.at(operand));
  }

  // For now, assume there is no complex operator in the combinational blocks
  while (que.size() > 1) {
    auto lhs = que.front();
    que.pop_front();
    auto rhs = que.front();
    que.pop_front();

    auto out = TypeSwitch<Operation *, signal>(op)
                   .Case<comb::AndOp>(
                       [&](comb::AndOp op) { return aig.create_and(lhs, rhs); })
                   .Case<comb::OrOp>(
                       [&](comb::OrOp op) { return aig.create_or(lhs, rhs); })
                   .Case<comb::XorOp>(
                       [&](comb::XorOp op) { return aig.create_xor(lhs, rhs); })
                   .Case<comb::MuxOp>([&](comb::MuxOp op) {
                     auto ctrl = lhs;
                     lhs = rhs;
                     rhs = que.front();
                     que.pop_front();

                     return aig.create_ite(ctrl, lhs, rhs);
                   })
                   .Default([&](Operation *op) {
                     llvm::errs()
                         << "Unsupported operation: " << op->getName() << "\n";
                     assert(false);
                     return signal{0};
                   });

    que.push_back(out);
  }

  return que.front();
}

class LUTNetworkToHW {
  using Node = klut_network::node;

public:
  LUTNetworkToHW(OpBuilder &builder, const klut_network &klut,
                 const CombinationalBlock &combBlock);

  LogicalResult convert();

private:
  Operation *createLutOp(Node node);
  void reconnectPOs();

private:
  OpBuilder &builder;
  const klut_network &klut;
  const CombinationalBlock &combBlock;
  SmallVector<Operation *> newOps;
  DenseSet<Operation *> erasedOps;
};

LUTNetworkToHW::LUTNetworkToHW(OpBuilder &builder, const klut_network &klut,
                               const CombinationalBlock &combBlock)
    : builder(builder), klut(klut), combBlock(combBlock),
      newOps(klut.size(), nullptr) {
  auto *opBlk = combBlock.getOps().front()->getBlock();
  builder.setInsertionPointToStart(opBlk);
}

LogicalResult LUTNetworkToHW::convert() {
  // for (auto [i, input] : enumerate(combBlock.getInputs()))
  //   llvm::outs() << "Input: " << i << " " << input << "\n";

  klut.foreach_node([&](auto node, auto index) {
    if (klut.is_pi(node) || klut.is_constant(node))
      return;

    auto op = createLutOp(node);

    assert(!newOps[node]);
    newOps[node] = op;
  });

  reconnectPOs();

  // Delete all mapped comb operations
  for (auto *op : combBlock.getOps()) {
    if (!erasedOps.count(op))
      op->erase();
  }

  return success();
}

Operation *LUTNetworkToHW::createLutOp(Node node) {
  assert(node > 2 && !klut.is_pi(node));

  SmallVector<Value, 6> inputs;
  klut.foreach_fanin(node, [&](auto fanin) {
    if (klut.is_pi(fanin)) {
      // pi starts from index 2
      // llvm::outs() << "Fanin: " << klut.pi_at(fanin - 2)
      //              << ", PI: " << fanin - 2 << "\n";
      inputs.push_back(combBlock.getInputs()[fanin - 2]);
    } else if (klut.is_constant(fanin)) {
      auto constOp =
          builder.create<hw::ConstantOp>(combBlock.getOps().front()->getLoc(),
                                         builder.getIntegerType(1), fanin);
      inputs.push_back(constOp);
    } else {
      Value value = newOps[fanin]->getResult(0);
      assert(value);
      inputs.push_back(value);
    }
  });

  assert(klut.node_function(node).num_vars() <= 6);
  uint64_t tt = 0;
  for (auto val : klut.node_function(node)) {
    tt = val;
    break;
  }

  return builder.create<xlnx::XlnxLutNOp>(combBlock.getOps().front()->getLoc(),
                                          tt, ValueRange{inputs});
}

void LUTNetworkToHW::reconnectPOs() {
  IRRewriter rewriter(builder);
  // set POs into new operations
  for (auto [idx, poValue] : enumerate(combBlock.getOutputs())) {
    Operation *oldOp = poValue.getDefiningOp();
    auto newOpIdx = klut.po_at(idx);
    // With the help of rewriter, dangling old operations will be pruned
    // automatically
    rewriter.replaceOp(oldOp, newOps[newOpIdx]);
    erasedOps.insert(oldOp);
  }
}
} // namespace

namespace circt {
namespace synthesis {
mockturtle::aig_network convertBlockToAIG(const CombinationalBlock &combBlock) {
  AIGConverter aigConverter(combBlock);
  return aigConverter.getAIG();
}

LogicalResult convertLUTToHW(OpBuilder &builder, const klut_network &klut,
                             const CombinationalBlock &combBlock) {
  LUTNetworkToHW converter(builder, klut, combBlock);
  return converter.convert();
}

llvm::EquivalenceClasses<Operation *>
collectCombinationalBlocks(hw::HWModuleOp module) {
  llvm::EquivalenceClasses<Operation *> combBlks;

  // Only walk operations in the current block
  for (auto &block : module.getBody()) {
    for (Operation &operation : block) {
      if (!isCombOperation(&operation))
        continue;

      combBlks.insert(&operation);

      // collect logic blocks by operation's connections
      for (const auto &operand : operation.getOperands()) {
        auto *op = operand.getDefiningOp();
        if (!op) {
          // operation is a block's argument
          // collect all user operations into one logic block
          for (auto *user : operation.getUsers()) {
            if (isCombOperation(user))
              combBlks.unionSets(&operation, user);
          }
        } else if (isCombOperation(op)) {
          combBlks.unionSets(&operation, op);
        }
      }
    }
  }

  return combBlks;
}

/// Only block's arguments and non-comb operations' results are PI
void CombinationalBlock::updatePI(Operation *op) {
  for (Value operand : op->getOperands()) {
    auto *op = operand.getDefiningOp();
    if (!op || (!isCombOperation(op) && !isa<hw::ConstantOp>(op)))
      inputs.push_back(operand);
  }
}

/// Only operations without comb operations as users are outputs
void CombinationalBlock::updatePO(Operation *op) {
  for (auto *user : op->getUsers()) {
    if (isCombOperation(user))
      return;
  }
  assert(op->getNumResults() == 1 && "Only support single-bit output");
  outputs.push_back(op->getResult(0));
}

namespace mapper {
LogicalResult NaiveLutMapper::map() {
  using namespace mockturtle;
  lut_map_params ps;
  ps.cut_enumeration_ps.cut_size = 6u;
  ps.cut_enumeration_ps.cut_limit = 8u;
  ps.recompute_cuts = true;
  ps.area_oriented_mapping = false;
  ps.cut_expansion = true;
  lut_map_stats st;
  result = lut_map(*aig, ps, &st);
  return success();
}
} // namespace mapper

LogicalResult CombBlockConverter::combToLut(mapper::MapperAlgoBase &mapper) {
  auto blocks = collectCombinationalBlocks(module);
  for (const auto &it : blocks) {
    if (!it->isLeader())
      continue;
    auto opRange = blocks.members(*it);
    auto combBlock = CombinationalBlock(opRange.begin(), opRange.end());
    aig_network aig = convertBlockToAIG(combBlock);

    mapper.setAIG(&aig);
    // TODO: AIG optimization
    // mapper.optimize();

    if (failed(mapper.map()))
      return failure();

    if (failed(convertLUTToHW(builder, mapper.getResult(), combBlock)))
      return failure();
  }

  return success();
}
} // namespace synthesis
} // namespace circt

using namespace circt::synthesis;

namespace {

struct HWToNetlistPass
    : public circt::impl::LowerHWToNetlistBase<HWToNetlistPass> {
  void runOnOperation() override;
};

} // namespace

static void populateLegality(ConversionTarget &target) {
  target.addIllegalDialect<comb::CombDialect>();
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<xlnx::XlnxDialect>();
  // TODO: sequantial mapping
  target.addLegalDialect<seq::SeqDialect>();

  // target.addLegalOp<comb::TruthTableOp>();
}

void HWToNetlistPass::runOnOperation() {
  MLIRContext &context = getContext();
  hw::HWModuleOp module = getOperation();
  OpBuilder builder(module);

  ConversionTarget target(context);

  populateLegality(target);

  CombBlockConverter lutConv(module, builder);
  mapper::NaiveLutMapper m;
  if (failed(lutConv.combToLut(m)))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// HW to SV Conversion Pass
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<hw::HWModuleOp>>
circt::createLowerHWToNetlistPass() {
  return std::make_unique<HWToNetlistPass>();
}
