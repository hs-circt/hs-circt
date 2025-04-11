//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_XLNX_XLNXOPS_H
#define CIRCT_DIALECT_XLNX_XLNXOPS_H

// clang-format off
#include "circt/Dialect/Xlnx/XlnxDialect.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/ValueRange.h"
#include "circt/Dialect/Xlnx/XlnxTypes.h"

#define GET_OP_CLASSES
#include "circt/Dialect/Xlnx/XlnxOpInterfaces.h.inc"
#include "circt/Dialect/Xlnx/Xlnx.h.inc"
// clang-format on


#include <optional>

namespace circt {
namespace xlnx {
namespace xlnx_prims_helper {
namespace application {
template <class Action>
struct Applicator {
  template <class Opt>
  static void apply(const Action &action, Opt &O) {
    action.apply(O);
  }
};

template <typename Obj>
void ApplyAction(Obj &O) {
  (void)O;
}

template <typename Obj, typename Action>
void ApplyAction(Obj &O, const Action &action) {
  Applicator<Action>::apply(action, O);
}

template <typename Obj, typename Action, typename... Actions>
void ApplyAction(Obj &O, const Action &action, const Actions &...actions) {
  ApplyAction(O, action);
  ApplyAction(O, actions...);
}
} // namespace application

namespace model {
struct Lut1 {
  template <typename... Actions>
  explicit Lut1(Actions &&...actions) {
    application::ApplyAction(*this, actions...);
  }

  template <typename OpBuilder>
  auto done(OpBuilder &opBuilder) {
    return opBuilder.template create<circt::xlnx::XlnxLut1Op>(i0, init);
  }

  std::optional<::mlir::Value> i0 = std::nullopt;

  std::optional<uint64_t> init = std::nullopt;
};
} // namespace model
} // namespace xlnx_prims_helper

#ifdef LUT1_NAMED_ASSOCIATED_BUILD
template <typename... Actions>
void XlnxLut1Op::build(::mlir::OpBuilder &odsBuilder,
                       ::mlir::OperationState &odsState, Actions &&...actions) {
  xlnx_prims_helper::model::Lut1 lut1(actions...);
  // Check if I0 and INIT are assigned
  if (!lut1.i0.has_value()) {
    assert(false && "I0 is not assigned");
  }
  if (!lut1.init.has_value()) {
    assert(false && "INIT is not assigned");
  }
  XlnxLut1Op::build(odsBuilder, odsState, lut1.i0.value(), lut1.init.value());
}
#endif
} // namespace xlnx
} // namespace circt

#define LUT_INPUT(IDX)                                                         \
  struct I##IDX {                                                              \
    explicit I##IDX(::mlir::Value connectWire) : connectWire(connectWire) {}   \
    template <typename PrimOpBuilder>                                          \
    void apply(PrimOpBuilder &opBuilder) const {                               \
      opBuilder.i##IDX = connectWire;                                          \
    }                                                                          \
    ::mlir::Value connectWire = nullptr;                                       \
  };

LUT_INPUT(0)
LUT_INPUT(1)
LUT_INPUT(2)
LUT_INPUT(3)
LUT_INPUT(4)
LUT_INPUT(5)

#undef LUT_INPUT

struct INIT {
  explicit INIT(uint64_t init) : init(init) {}

  template <typename PrimOpBuilder>
  void apply(PrimOpBuilder &opBuilder) const {
    opBuilder.init = init;
  }

  uint64_t init = 0;
};

#endif // CIRCT_DIALECT_XLNX_XLNXOPS_H
