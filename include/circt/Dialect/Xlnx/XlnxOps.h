//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_XLNX_XLNXOPS_H
#define CIRCT_DIALECT_XLNX_XLNXOPS_H

#include "circt/Dialect/Xlnx/XlnxDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/ValueRange.h"

#define GET_OP_CLASSES
#include "circt/Dialect/Xlnx/Xlnx.h.inc"

namespace circt {
    namespace xlnx {
        namespace xlnx_prims_helper {
            template <class Action>
            struct Applicator {
                template <class Opt> static void apply(const Action &action, Opt &O) {
                    action.apply(O);
                }
            };

            template <typename Obj> void ApplyAction(Obj &O) {
                (void) O;
            }

            template <typename Obj, typename Action> void ApplyAction(Obj &O, const Action &action) {
                Applicator<Action>::apply(action, O);
            }

            template <typename Obj, typename Action, typename ... Actions>
            void ApplyAction(Obj &O, const Action &action, const Actions &... actions) {
                ApplyAction(O, action);
                ApplyAction(O, actions...);
            }
            
            struct Lut1 {
                template <typename ... Actions>
                explicit Lut1(Actions &&... actions) {
                    ApplyAction(*this, actions...);
                }

                template <typename OpBuilder>
                auto done(OpBuilder &opBuilder) {
                    return opBuilder.template create<circt::xlnx::XlnxLut1Op>(i0, init);
                }

                ::mlir::Value i0 = nullptr;
                uint64_t init = 0;
            };
        }
        template <typename ... Actions>
        void XlnxLut1Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Actions &&... actions) {
            xlnx_prims_helper::Lut1 lut1(actions...);
            XlnxLut1Op::build(odsBuilder, odsState, lut1.i0, lut1.init);
        }
    }
}

struct I0 {
    explicit I0(::mlir::Value connectWire) : connectWire(connectWire) {}

    template <typename PrimOpBuilder>
    void apply(PrimOpBuilder &opBuilder) const {
        opBuilder.i0 = connectWire;
    }

    ::mlir::Value connectWire = nullptr;
};

struct INIT {
    explicit INIT(uint64_t init) : init(init) {}

    template <typename PrimOpBuilder>
    void apply(PrimOpBuilder &opBuilder) const {
        opBuilder.init = init;
    }

    uint64_t init = 0;
};

#endif // CIRCT_DIALECT_XLNX_XLNXOPS_H
