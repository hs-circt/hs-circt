//===- CoreToXlnx.cpp - Core to Xlnx Conversion Pass ----------------------===//
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
#include "circt/Dialect/Comb/CombOps.h"
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
using namespace comb;
//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// Operation Conversion
//===----------------------------------------------------------------------===//

enum class ResetType { SyncSet, SyncReset, Unknown };

template <typename OpLowering, typename Op>
struct CompRegRewriterHelper : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<Op>::OpAdaptor;

  // Only if hasReset is true, we need to determine if we should use FDSE or
  // FDRE If hasReset is false, we can directly use FDRE with a constant 0
  // reset signal and skip the FDSE/FDRE determination
  ResetType getResetType(Op op, Value resetValue) const {
    // Check if reset value is a constant
    if (auto constOp = resetValue.getDefiningOp<hw::ConstantOp>()) {
      IntegerAttr attr = constOp.getValueAttr();
      // Check if the constant is i1 type.
      if (!attr || !attr.getType().isInteger(1)) {
        op.emitError() << "Reset value constant is not an i1 type:";
        constOp->dump();
        return ResetType::Unknown;
      }
      // Use FDSE if reset value exists and is constant 1
      if (attr.getValue().isOne()) {
        return ResetType::SyncSet;
      } else if (attr.getValue().isZero()) {
        return ResetType::SyncReset;
      } else {
        // Should not happen for i1
        op.emitError() << "Invalid i1 constant value for reset:";
        constOp->dump();
        return ResetType::Unknown;
      }
    } else {
      // Reset value is not a constant, emit an error
      op->emitError() << "Reset value is not a constant. Cannot determine "
                         "whether to use FDSE or FDRE:";
      resetValue.dump(); // Dump the Value
      return ResetType::Unknown;
    }
  }

  struct MultiBitContext {
    Op op; // Original operation
    uint32_t numBits;
    uint32_t selectedIndex;
    Value inputDataSignal;
    Value resetValue;
    std::vector<Value> outputDataSignals;
    ConversionPatternRewriter &rewriter;

    MultiBitContext(Op op, Value inputDataSignal, Value resetValue,
                    ConversionPatternRewriter &rewriter)
        : op(op), inputDataSignal(inputDataSignal), resetValue(resetValue),
          rewriter(rewriter) {
      numBits = inputDataSignal.getType().getIntOrFloatBitWidth();
      outputDataSignals.reserve(numBits);
      selectedIndex = 0;
    }

    void next() { selectedIndex++; }

    bool isDone() const { return selectedIndex >= numBits; }

    Value get1BitInputDataSignal() const {
      if (numBits > 1) {
        return rewriter.create<comb::ExtractOp>(op->getLoc(), inputDataSignal,
                                                selectedIndex,
                                                /*bitWidth=*/1);
      } else {
        return inputDataSignal;
      }
    }

    Value get1BitResetValue() const {
      Location loc = op->getLoc();
      if (numBits == 1) {
        // If the original resetValue is already 1-bit, return it directly.
        // Ensure it's actually i1 if hasReset is true.
        if (resetValue && !resetValue.getType().isInteger(1)) {
          op->emitError() << "Single-bit reset value is not i1 type.";
          return nullptr;
        }
        return resetValue;
      }

      // Get the defining ConstantOp for the multi-bit reset value.
      auto *definingOp = resetValue.getDefiningOp();
      auto constantOp = dyn_cast_or_null<hw::ConstantOp>(definingOp);

      if (!constantOp) {
        // This should ideally be checked before entering the loop or
        // construction. If resetValue is not a constant, we cannot extract
        // bits.
        op->emitError()
            << "Multi-bit reset value must be a compile-time constant.";
        return nullptr; // Indicate failure
      }

      IntegerAttr attr = constantOp.getValueAttr();
      if (!attr) {
        op->emitError() << "Constant reset value is missing IntegerAttr.";
        return nullptr;
      }

      const APInt &multiBitValue = attr.getValue();
      // Extract the bit at selectedIndex.
      APInt singleBitValue = multiBitValue.lshr(selectedIndex).trunc(1);

      // Create a new hw.ConstantOp for this single bit value.
      // Insert the new constant right before the current operation 'op'.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(op);
      return rewriter.create<hw::ConstantOp>(loc, singleBitValue);
    }

    void addOutputDataSignal(Value outputDataSignal) {
      outputDataSignals.push_back(outputDataSignal);
    }

    Value getOutputDataSignal() const {
      if (numBits > 1) {
        return rewriter.create<comb::ConcatOp>(op->getLoc(), outputDataSignals);
      } else {
        // Should have exactly one signal if numBits is 1
        assert(outputDataSignals.size() == 1 &&
               "Expected one output signal for single bit register");
        return outputDataSignals.front();
      }
    }
  };

  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value inputDataSignal = adaptor.getInput();
    Value clockSignal = adaptor.getClk();
    Value clockEnable = OpLowering::getClockEnable(op, adaptor, rewriter);
    Value resetSignal = adaptor.getReset();
    Value resetValue = adaptor.getResetValue();

    // Check if the operation has an initial value
    if (op.getInitialValue()) {
      op.emitWarning() << "Initial value is not fully supported in the current "
                          "implementation. The conversion may not correctly "
                          "handle all initial value cases.";
    }

    // Check if we have a reset signal and value
    bool hasReset = false;
    if (resetSignal != nullptr && resetValue != nullptr) {
      hasReset = true;
      // Check if resetSignal is multiple bits
      if (resetSignal.getType().getIntOrFloatBitWidth() > 1) {
        op.emitError() << "Reset signal must be a single bit.";
        return failure();
      }
      // Check if resetValue type matches input data type width
      if (resetValue.getType() != inputDataSignal.getType()) {
        op.emitError() << "Reset value type (" << resetValue.getType()
                       << ") must match input data type ("
                       << inputDataSignal.getType() << ").";
        return failure();
      }
      // Check if resetValue is a constant if it's multi-bit. Single-bit reset
      // can be dynamic.
      unsigned resetWidth = resetValue.getType().getIntOrFloatBitWidth();
      if (resetWidth > 1 && !resetValue.getDefiningOp<hw::ConstantOp>()) {
        op.emitError()
            << "Multi-bit reset value must be a compile-time constant.";
        return failure();
      }

    } else if (resetSignal == nullptr && resetValue == nullptr) {
      hasReset = false;
    } else {
      op.emitError() << "Reset signal and reset value must both be provided or "
                        "both be omitted.";
      return failure();
    }

    // If no reset is provided, create a constant 0 for the reset signal
    // and a compatible zero constant for resetValue.
    if (!hasReset) {
      resetSignal =
          rewriter.create<hw::ConstantOp>(loc, rewriter.getI1Type(), 0);
      // Create a zero constant of the same type as the input data
      APInt zeroVal(inputDataSignal.getType().getIntOrFloatBitWidth(), 0);
      resetValue = rewriter.create<hw::ConstantOp>(
          loc, inputDataSignal.getType(),
          rewriter.getIntegerAttr(inputDataSignal.getType(), zeroVal));
    }

    MultiBitContext multiBitCtx{op, inputDataSignal, resetValue, rewriter};

    do {
      Value singleBitInputDataSignal = multiBitCtx.get1BitInputDataSignal();
      Value singleBitResetValue =
          multiBitCtx.get1BitResetValue(); // Returns Value
      if (!singleBitResetValue && hasReset) {
        op->emitError() << "Failed to extract single bit reset value.";
        return failure();
      }

      // Determine if we should use FDSE (set) or FDRE (reset) based on reset
      // value
      ResetType resetType = ResetType::Unknown;
      if (hasReset) {
        // Pass the single-bit *Value* to getResetType
        resetType = getResetType(op, singleBitResetValue);
      } else {
        // If no reset was originally provided, we synthesized a 0 reset, so use
        // SyncReset
        resetType = ResetType::SyncReset;
      }

      if (resetType == ResetType::Unknown) {
        op.emitError()
            << "Reset type is unknown. Cannot determine if we should "
               "use FDSE or FDRE.";
        return failure();
      }

      // Create the appropriate Xilinx flip-flop primitive
      Value singleBitResult;
      // Ensure the input type is i1 for FDSE/FDRE
      if (!singleBitInputDataSignal.getType().isInteger(1)) {
        op->emitError() << "Input to FDSE/FDRE must be i1, but got "
                        << singleBitInputDataSignal.getType();
        return failure();
      }

      if (resetType == ResetType::SyncSet) {
        // Use FDSE (synchronous set) when reset value is 1
        auto fdseOp = rewriter.create<xlnx::XlnxFDSEOp>(
            loc, singleBitInputDataSignal.getType(), /*C=*/clockSignal,
            /*CE=*/clockEnable,
            /*S=*/resetSignal, /*D=*/singleBitInputDataSignal);
        singleBitResult = fdseOp.getResult();
      } else if (resetType == ResetType::SyncReset) {
        auto fdreOp = rewriter.create<xlnx::XlnxFDREOp>(
            loc, singleBitInputDataSignal.getType(), /*C=*/clockSignal,
            /*CE=*/clockEnable,
            /*R=*/resetSignal, /*D=*/singleBitInputDataSignal);
        singleBitResult = fdreOp.getResult();
      } else {
        op.emitError()
            << "Reset type is unknown. Cannot determine if we should "
               "use FDSE or FDRE.";
        return failure();
      }
      multiBitCtx.addOutputDataSignal(singleBitResult);
      multiBitCtx.next();
    } while (!multiBitCtx.isDone());

    // Replace the original operation with the new Xilinx flip-flop operation
    rewriter.replaceOp(op, multiBitCtx.getOutputDataSignal());

    return success();
  }
};

/// Converts `seq.compreg` to `xlnx.fdse` or `xlnx.fdre` based on reset value.
/// If reset value is 1, use `xlnx.fdse` (synchronous set).
/// If reset value is 0 or not specified, use `xlnx.fdre` (synchronous reset).
struct CompRegLowering
    : public CompRegRewriterHelper<CompRegLowering, CompRegOp> {
  using CompRegRewriterHelper<CompRegLowering,
                              CompRegOp>::CompRegRewriterHelper;

  static Value
  getClockEnable(CompRegOp op,
                 CompRegRewriterHelper<CompRegLowering, CompRegOp>::OpAdaptor adaptor,
                 ConversionPatternRewriter &rewriter) {
    // Find the parent HWModuleOp.
    auto hwModule = op->template getParentOfType<hw::HWModuleOp>();
    Type i1Type = rewriter.getI1Type();
    IntegerAttr trueAttr = rewriter.getIntegerAttr(i1Type, 1);

    if (!hwModule) {
      op.emitError("CompRegOp must be inside an hw.module");
      // Cannot directly signal failure from this helper function.
      // Fallback to creating a local constant, though this likely indicates
      // an unexpected IR structure upstream.
      return rewriter.template create<hw::ConstantOp>(op.getLoc(), trueAttr);
    }

    Block *moduleBody = hwModule.getBodyBlock();
    if (!moduleBody) {
      // Should not happen for a valid HWModuleOp with a body.
      op.emitError("Parent hw.module has no body block");
      // Fallback similar to the !hwModule case.
      return rewriter.template create<hw::ConstantOp>(op.getLoc(), trueAttr);
    }

    // Search the HW module's top-level operations for an existing `hw.constant
    // true : i1`.
    Value existingConstantValue;
    for (Operation &topOp : moduleBody->getOperations()) {
      if (auto constantOp = dyn_cast<hw::ConstantOp>(topOp)) {
        if (constantOp.getType() == i1Type && constantOp.getValue().isOne()) {
          existingConstantValue = constantOp.getResult();
          break; // Found the constant, no need to search further.
        }
      }
      // Note: We search the entire top level first to ensure we find *any*
      // suitable constant, rather than optimizing for dominance prematurely.
    }

    // Return the existing constant if found.
    if (existingConstantValue)
      return existingConstantValue;

    // Otherwise, create a new constant at the beginning of the HW module body.
    OpBuilder::InsertionGuard guard(
        rewriter); // Saves/restores insertion point.
    rewriter.setInsertionPointToStart(moduleBody);
    // Use the HW module's location for the new constant operation.
    return rewriter.template create<hw::ConstantOp>(hwModule.getLoc(),
                                                    trueAttr);
  }
};

/// Converts `seq.compreg.ce` to `xlnx.fdse` or `xlnx.fdre` based on reset
/// value. If reset value is 1, use `xlnx.fdse` (synchronous set). If reset
/// value is 0 or not specified, use `xlnx.fdre` (synchronous reset).
struct CompRegCELowering
    : public CompRegRewriterHelper<CompRegCELowering, CompRegClockEnabledOp> {
  using CompRegRewriterHelper<CompRegCELowering,
                              CompRegClockEnabledOp>::CompRegRewriterHelper;

  static Value getClockEnable(
      CompRegClockEnabledOp op,
      CompRegRewriterHelper<CompRegCELowering, CompRegClockEnabledOp>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) {
    return adaptor.getClockEnable();
  }
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
  target.addLegalOp<comb::ExtractOp>();
  target.addLegalOp<comb::ConcatOp>();

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