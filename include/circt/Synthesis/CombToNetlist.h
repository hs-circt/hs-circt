//===- CombToNetlist.h - Comb to Netlist ----------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a library of Comb to Netlist.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SYNTHESIS_COMBTONETLIST_H
#define CIRCT_SYNTHESIS_COMBTONETLIST_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/EquivalenceClasses.h"

#include "mlir/Analysis/TopologicalSortUtils.h"

#include "mockturtle/networks/aig.hpp"
#include "mockturtle/networks/klut.hpp"

namespace circt {
namespace synthesis {
using aig_network = mockturtle::aig_network;
using klut_network = mockturtle::klut_network;

class CombinationalBlock;

// Utility functions
llvm::EquivalenceClasses<Operation *>
collectCombinationalBlocks(hw::HWModuleOp module);

aig_network convertBlockToAIG(const CombinationalBlock &combBlock);

LogicalResult convertLUTToHW(OpBuilder &builder, const klut_network &klut,
                             const CombinationalBlock &combBlock);

class CombinationalBlock {
public:
  template <typename R>
  CombinationalBlock(R &&begin, R &&end) : ops(begin, end) {
    ops = mlir::topologicalSort(ops);

    for (auto *op : ops) {
      updatePI(op);
      updatePO(op);
      // ops.insert(op);
    }
  }

  /// Only block's arguments and constOp's results and are PI
  void updatePI(Operation *op);

  /// Only operations without comb operations as users are outputs
  void updatePO(Operation *op);

  ArrayRef<Value> getInputs() const { return inputs; }
  ArrayRef<Value> getOutputs() const { return outputs; }
  const SetVector<Operation *> &getOps() const { return ops; }

private:
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
  // store in reverse topological order
  SetVector<Operation *> ops;
};

namespace optimizer {
// struct OptimizerBase {
//   virtual bool operator()(aig_network &aig) = 0;
// };

struct TrivialAIGOptimizer {
  // TrivialAIGOptimizer(const AIGConverter &aig) : aig(aig) {}
  bool operator()(aig_network &aig) { return true; }
};

} // namespace optimizer

namespace mapper {
struct MapperAlgoBase {
  MapperAlgoBase() = default;
  virtual ~MapperAlgoBase() = default;

  void setAIG(const aig_network *aig) { this->aig = aig; }
  const klut_network &getResult() const { return result; }

  virtual LogicalResult map() = 0;

protected:
  // const CombinationalBlock &combBlock;
  const aig_network *aig{};
  klut_network result{};
};

class NaiveLutMapper : public MapperAlgoBase {
public:
  LogicalResult map() override;
};
} // namespace mapper

class CombBlockConverter {
public:
  CombBlockConverter(hw::HWModuleOp module, OpBuilder &builder)
      : module(module), builder(builder) {}

  LogicalResult combToLut(mapper::MapperAlgoBase &mapper);

protected:
  hw::HWModuleOp module;
  OpBuilder &builder;
};

} // namespace synthesis
} // namespace circt

#endif // CIRCT_SYNTHESIS_COMBTONETLIST_H
