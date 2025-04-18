//===- CoreToXlnx.cpp - Core to Xlnx Conversion Pass ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Core to Xlnx conversion pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CoreToXlnx.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Xlnx/XlnxDialect.h"
#include "circt/Dialect/Xlnx/XlnxOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTCORETOXLNX
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;
using namespace seq;
using namespace xlnx;

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// Operation Conversion
//===----------------------------------------------------------------------===//

enum class ResetType { SyncSet, SyncReset, Unknown };

template <typename T>
struct ClockEnable {
  static Value get();
};

template <>
struct ClockEnable<CompRegOp> {
  template <typename Op, typename OpAdaptor, typename Rewriter>
  static Value get(Op op, OpAdaptor adaptor, Rewriter &rewriter) {
    return rewriter.template create<hw::ConstantOp>(op.getLoc(),
                                                    rewriter.getI1Type(), 1);
  }
};

template <>
struct ClockEnable<CompRegClockEnabledOp> {
  template <typename Op, typename OpAdaptor, typename Rewriter>
  static Value get(Op op, OpAdaptor adaptor, Rewriter &rewriter) {
    return adaptor.getClockEnable();
  }
};

template <typename Op>
struct CompRegRewriterHelper : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<Op>::OpAdaptor;

  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value inputDataSignal = adaptor.getInput();
    Value clockSignal = adaptor.getClk();
    Value clockEnable = ClockEnable<Op>::get(op, adaptor, rewriter);
    Value resetSignal = adaptor.getReset();
    Value resetValue = adaptor.getResetValue();

    // Check if we have a reset signal and value
    bool hasReset = false;
    if (resetSignal != nullptr && resetValue != nullptr)
      hasReset = true;
    else if (resetSignal == nullptr && resetValue == nullptr)
      hasReset = false;
    else {
      op.emitError() << "Reset signal and reset value must both be provided or "
                        "both be omitted.";
      return failure();
    }

    // Determine if we should use FDSE (set) or FDRE (reset) based on reset
    // value
    ResetType resetType = ResetType::Unknown;

    // Only if hasReset is true, we need to determine if we should use FDSE or
    // FDRE If hasReset is false, we can directly use FDRE with a constant 0
    // reset signal and skip the FDSE/FDRE determination
    if (hasReset) {
      // Check if reset value is a constant
      if (auto constOp = resetValue.getDefiningOp<hw::ConstantOp>()) {
        // Use FDSE if reset value exists and is constant 1
        if (constOp.getValue().isOne())
          resetType = ResetType::SyncSet;
        else if (constOp.getValue().isZero())
          resetType = ResetType::SyncReset;
        else {
          op.emitError() << "Reset value is not a constant. Cannot determine "
                            "whether to use FDSE or FDRE.";
          return failure();
        }
      } else {
        // Reset value is not a constant, emit a warning and return failure
        op.emitError() << "Reset value is not a constant. Cannot determine "
                          "whether to use FDSE or FDRE.";
        return failure();
      }
    } else {
      // No reset signal or value, use FDRE with a constant 0 reset signal
      resetType = ResetType::SyncReset;
    }

    // Check if the operation has an initial value
    if (op.getInitialValue()) {
      op.emitWarning() << "Initial value is not fully supported in the current "
                          "implementation. The conversion may not correctly "
                          "handle all initial value cases.";
    }

    // Create the appropriate Xilinx flip-flop primitive
    Value result;
    if (resetType == ResetType::SyncSet) {
      // Use FDSE (synchronous set) when reset value is 1
      auto fdseOp = rewriter.create<xlnx::XlnxFDSEOp>(
          loc, inputDataSignal.getType(), /*C=*/clockSignal, /*CE=*/clockEnable,
          /*S=*/resetSignal, /*D=*/inputDataSignal);
      result = fdseOp.getResult();
    } else if (resetType == ResetType::SyncReset) {
      // Use FDRE (synchronous reset) when reset value is 0 or not specified
      // If no reset is provided, create a constant 0 for the reset signal
      if (!hasReset) {
        resetSignal =
            rewriter.create<hw::ConstantOp>(loc, rewriter.getI1Type(), 0);
      }

      auto fdreOp = rewriter.create<xlnx::XlnxFDREOp>(
          loc, inputDataSignal.getType(), /*C=*/clockSignal, /*CE=*/clockEnable,
          /*R=*/resetSignal, /*D=*/inputDataSignal);
      result = fdreOp.getResult();
    } else {
      op.emitError() << "Reset type is unknown. Cannot determine if we should "
                        "use FDSE or FDRE.";
      return failure();
    }

    // Replace the original operation with the new Xilinx flip-flop operation
    rewriter.replaceOp(op, result);

    return success();
  }
};

/// Converts `seq.compreg` to `xlnx.fdse` or `xlnx.fdre` based on reset value.
/// If reset value is 1, use `xlnx.fdse` (synchronous set).
/// If reset value is 0 or not specified, use `xlnx.fdre` (synchronous reset).
struct CompRegLowering : public CompRegRewriterHelper<CompRegOp> {
  using CompRegRewriterHelper<CompRegOp>::CompRegRewriterHelper;
};

/// Converts `seq.compreg.ce` to `xlnx.fdse` or `xlnx.fdre` based on reset
/// value. If reset value is 1, use `xlnx.fdse` (synchronous set). If reset
/// value is 0 or not specified, use `xlnx.fdre` (synchronous reset).
struct CompRegCELowering : public CompRegRewriterHelper<CompRegClockEnabledOp> {
  using CompRegRewriterHelper<CompRegClockEnabledOp>::CompRegRewriterHelper;
};

} // namespace

//===----------------------------------------------------------------------===//
// CoreToXlnxPass
//===----------------------------------------------------------------------===//

namespace {
struct CoreToXlnxPass
    : public circt::impl::ConvertCoreToXlnxBase<CoreToXlnxPass> {
  void runOnOperation() override;
};
} // namespace

static void populateLegality(ConversionTarget &target) {
  // Mark seq.compreg and seq.compreg.ce as illegal
  target.addIllegalOp<seq::CompRegOp, seq::CompRegClockEnabledOp>();

  // Mark Xlnx dialect as legal
  target.addLegalDialect<XlnxDialect,          // Xlnx ops are legal
                         mlir::BuiltinDialect, // Needed for func, module etc.
                         HWDialect>();

  // Mark specific core ops required by Xlnx ops as legal
  target.addLegalOp<hw::ConstantOp>();
  target.addLegalOp<hw::InstanceOp>();

  // Mark seq.initial and seq.yield as legal since we need them for initial
  // values
  target.addLegalOp<seq::InitialOp, seq::YieldOp>();
}

static void populateOpConversion(RewritePatternSet &patterns) {
  patterns.add<CompRegLowering, CompRegCELowering>(patterns.getContext());
}

void CoreToXlnxPass::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();

  IRRewriter rewriter(module);

  ConversionTarget target(context);
  RewritePatternSet patterns(&context);

  populateLegality(target);
  populateOpConversion(patterns);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

/// Creates the CoreToXlnx pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertCoreToXlnxPass() {
  return std::make_unique<CoreToXlnxPass>();
}