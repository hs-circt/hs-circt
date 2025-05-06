//===- SeqToXlnx.cpp - Core to Xlnx Conversion Pass ----------------------===//
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

#include "circt/Conversion/SeqToXlnx.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Xlnx/XlnxDialect.h"
#include "circt/Dialect/Xlnx/XlnxOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTSEQTOXLNX
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

/**
 * @brief Helper class for rewriting `seq.compreg` and `seq.compreg.ce` operations.
 *
 * This template class provides the common logic for converting `seq::CompRegOp` or
 * `seq::CompRegClockEnabledOp` into Xilinx flip-flop primitives (`xlnx::XlnxFDSEOp`
 * or `xlnx::XlnxFDREOp`). It handles the conversion for both single-bit and
 * multi-bit registers and determines whether to use FDSE (synchronous set) or
 * FDRE (synchronous reset) based on the reset value.
 *
 * @tparam OpLowering The specific operation conversion class (e.g., `CompRegLowering`
 *                    or `CompRegCELowering`) that provides details specific to the
 *                    operation, such as how to get the clock enable signal.
 * @tparam Op The original MLIR operation type being converted (e.g., `seq::CompRegOp`
 *            or `seq::CompRegClockEnabledOp`).
 */
template <typename OpLowering, typename Op>
struct CompRegRewriterHelper : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<Op>::OpAdaptor;

  /**
   * @brief Determines the reset type (synchronous set or synchronous reset) based on a single bit of the reset value.
   *
   * This function only needs to be called when a reset signal is present (i.e., `hasReset` is true)
   * to determine whether FDSE or FDRE should be used.
   * If no reset signal exists, FDRE can be used directly with the reset signal tied to constant 0,
   * skipping this determination.
   *
   * @param op The original operation, used for emitting errors.
   * @param resetBitValue The single-bit `APInt` of the reset value. Must be of i1 type.
   * @return ResetType indicating synchronous set (`SyncSet`), synchronous reset (`SyncReset`), or unknown (`Unknown`).
   *         Returns `Unknown` and emits an error if `resetBitValue` is not a valid i1 value.
   */
  ResetType getResetType(Op op, const APInt &resetBitValue) const {
    // Check if the constant is i1 type.
    if (resetBitValue.getBitWidth() != 1) {
      op.emitError() << "Reset value bit is not an i1 type, but is "
                     << resetBitValue.getBitWidth() << " bits wide.";
      // Dump the APInt value for debugging if needed
      llvm::errs() << "Invalid APInt for reset: ";
      resetBitValue.print(llvm::errs(), /*isSigned=*/false);
      llvm::errs() << "\n";
      return ResetType::Unknown;
    }
    // Use FDSE if reset value exists and is constant 1
    if (resetBitValue.isOne()) {
      return ResetType::SyncSet;
    } else if (resetBitValue.isZero()) {
      return ResetType::SyncReset;
    } else {
      // Should not happen for i1
      op.emitError() << "Invalid i1 value for reset bit.";
      // Dump the APInt value for debugging if needed
      llvm::errs() << "Invalid APInt for reset: ";
      resetBitValue.print(llvm::errs(), /*isSigned=*/false);
      llvm::errs() << "\n";
      return ResetType::Unknown;
    }
  }

  /**
   * @brief Context structure for handling multi-bit register conversions.
   *
   * This struct encapsulates the state and helper functions needed when converting
   * a multi-bit register bit-by-bit.
   */
  struct MultiBitContext {
    Op op; ///< The original MLIR operation.
    uint32_t numBits; ///< The number of bits in the register.
    uint32_t selectedIndex; ///< The index of the bit currently being processed.
    Value inputDataSignal; ///< The original multi-bit input data signal.
    Value resetValue; ///< The original reset value (potentially a multi-bit constant).
    std::vector<Value> outputDataSignals; ///< Stores the output signals from each single-bit flip-flop.
    ConversionPatternRewriter &rewriter; ///< The rewriter used for creating new operations.

    /**
     * @brief Constructs a MultiBitContext.
     *
     * @param op The original MLIR operation.
     * @param inputDataSignal The original multi-bit input data signal.
     * @param resetValue The original reset value (potentially a multi-bit constant).
     * @param rewriter The rewriter used for creating new operations.
     */
    MultiBitContext(Op op, Value inputDataSignal, Value resetValue,
                    ConversionPatternRewriter &rewriter)
        : op(op), inputDataSignal(inputDataSignal), resetValue(resetValue),
          rewriter(rewriter) {
      numBits = inputDataSignal.getType().getIntOrFloatBitWidth();
      outputDataSignals.reserve(numBits);
      selectedIndex = 0;
    }

    /** @brief Moves to the next bit. */
    void next() { selectedIndex++; }

    /** @brief Checks if all bits have been processed. */
    bool isDone() const { return selectedIndex >= numBits; }

    /**
     * @brief Gets the single-bit input data signal for the current index.
     *
     * If the register is multi-bit, creates a `comb::ExtractOp` to extract the signal for the current bit.
     * If the register is single-bit, returns the original input signal directly.
     *
     * @return The single-bit input data signal (i1) for the current bit.
     */
    Value get1BitInputDataSignal() const {
      if (numBits > 1) {
        return rewriter.create<comb::ExtractOp>(op->getLoc(), inputDataSignal,
                                                selectedIndex,
                                                /*bitWidth=*/1);
      } else {
        return inputDataSignal;
      }
    }

    /**
     * @brief Gets the single-bit reset value for the current index.
     *
     * - If the register is single-bit:
     *   - If `resetValue` is an `hw::ConstantOp`, extracts its `APInt` value.
     *   - If `resetValue` is not a constant, or its type is not i1, emits an error and returns `std::nullopt`.
     * - If the register is multi-bit:
     *   - `resetValue` MUST be an `hw::ConstantOp`.
     *   - Extracts the bit at the current index from the multi-bit constant value and returns it as an `APInt`.
     *   - If `resetValue` is not a constant or lacks an `IntegerAttr`, emits an error and returns `std::nullopt`.
     *
     * @return An `std::optional<APInt>` containing the reset value for the current bit. Returns `std::nullopt` if it cannot be determined or an error occurs.
     */
    std::optional<APInt> get1BitResetValue() const {
      Location loc = op->getLoc();
      if (numBits == 1) {
        // Handle single-bit reset value
        if (!resetValue) {
          // This case should technically be handled by the logic
          // synthesizing a zero reset if hasReset is false initially.
          // If we reach here with hasReset=true and !resetValue, it's an error.
          op->emitError() << "Missing reset value for single-bit register "
                             "when reset is expected.";
          return std::nullopt;
        }
        if (!resetValue.getType().isInteger(1)) {
          op->emitError() << "Single-bit reset value is not i1 type.";
          return std::nullopt;
        }
        // If it's a constant, extract the APInt
        if (auto constantOp = resetValue.getDefiningOp<hw::ConstantOp>()) {
          if (auto attr = constantOp.getValueAttr()) {
            return attr.getValue();
          } else {
            op->emitError() << "Single-bit constant reset value is missing "
                               "IntegerAttr.";
            return std::nullopt;
          }
        } else {
          // If the single-bit reset is not a constant, we cannot determine
          // FDSE/FDRE at compile time based on its value. This logic
          // currently requires a constant reset value.
          op->emitError() << "Single-bit reset value must be a compile-time "
                             "constant to determine FDSE/FDRE.";
          resetValue.dump(); // Dump the Value
          return std::nullopt;
        }
      }

      // Handle multi-bit reset value (must be constant)
      auto *definingOp = resetValue.getDefiningOp();
      auto constantOp = dyn_cast_or_null<hw::ConstantOp>(definingOp);

      if (!constantOp) {
        op->emitError()
            << "Multi-bit reset value must be a compile-time constant.";
        return std::nullopt;
      }

      IntegerAttr attr = constantOp.getValueAttr();
      if (!attr) {
        op->emitError() << "Constant reset value is missing IntegerAttr.";
        return std::nullopt;
      }

      const APInt &multiBitValue = attr.getValue();
      // Extract the bit at selectedIndex.
      APInt singleBitValue = multiBitValue.lshr(selectedIndex).trunc(1);

      return singleBitValue; // Return the extracted APInt bit
    }

    /**
     * @brief Adds the output signal of a single-bit flip-flop to the results vector.
     * @param outputDataSignal The output `Value` of the single-bit flip-flop.
     */
    void addOutputDataSignal(Value outputDataSignal) {
      outputDataSignals.push_back(outputDataSignal);
    }

    /**
     * @brief Gets the final (potentially multi-bit) output data signal.
     *
     * If the register is multi-bit, creates a `comb::ConcatOp` to concatenate all single-bit output signals.
     * If the register is single-bit, returns the single output signal directly.
     *
     * @return The final output data signal.
     */
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

  /**
   * @brief Matches a `seq.compreg` or `seq.compreg.ce` operation and rewrites it into Xilinx flip-flops.
   *
   * This function implements the core conversion logic:
   * 1. Gets the operand adaptor.
   * 2. Gets the clock, clock enable, reset signal, and reset value.
   * 3. Handles the case where no reset signal is provided (synthesizes a synchronous reset to 0).
   * 4. Validates the reset signal and reset value (e.g., multi-bit reset value must be constant).
   * 5. Creates a `MultiBitContext` to handle potential bit-wise conversion.
   * 6. Iterates through each bit:
   *    a. Extracts the single-bit input data.
   *    b. Extracts the single-bit reset value (if present).
   *    c. Determines the reset type (FDSE or FDRE).
   *    d. Creates the corresponding `xlnx::XlnxFDSEOp` or `xlnx::XlnxFDREOp`.
   *    e. Adds the single-bit output to the `MultiBitContext`.
   * 7. Gets the final (potentially multi-bit) output signal from the `MultiBitContext`.
   * 8. Replaces the original operation.
   *
   * @param op The original operation to match and rewrite.
   * @param adaptor The operand adaptor, providing access to the operands.
   * @param rewriter The conversion pattern rewriter used for IR modifications.
   * @return `success()` if the match and rewrite were successful, `failure()` otherwise.
   *
   * @example Convert `seq.compreg` to `xlnx.fdre` (no reset -> default reset to 0)
   * ```mlir
   * %data_in = hw.constant 1 : i8
   * %clk = // ... clock signal
   * %out = seq.compreg %data_in, %clk : i8
   * ```
   * Converts to:
   * ```mlir
   * %c0_i1 = hw.constant 0 : i1
   * %c1_i1 = hw.constant 1 : i1 // Clock enable (constant true)
   * %c0_i8 = hw.constant 0 : i8 // Synthesized reset value
   * %extracted_bits = // ... 8 x comb.extract ops from %data_in
   * %fdre_outs = // ... 8 x xlnx.fdre %clk, %c1_i1, %c0_i1, %extracted_bits[...]
   * %out = comb.concat %fdre_outs[...] : (i1, i1, i1, i1, i1, i1, i1, i1) -> i8
   * ```
   *
   * @example Convert `seq.compreg` to `xlnx.fdse` (reset value is all ones)
   * ```mlir
   * %data_in = hw.constant 0 : i4
   * %clk = // ... clock signal
   * %rst = // ... reset signal (i1)
   * %rst_val = hw.constant 15 : i4 // All ones (binary 1111)
   * %out = seq.compreg %data_in, %clk, %rst, %rst_val : i4
   * ```
   * Converts to:
   * ```mlir
   * %c1_i1 = hw.constant 1 : i1 // Clock enable
   * %extracted_bits = // ... 4 x comb.extract ops from %data_in
   * %fdse_outs = // ... 4 x xlnx.fdse %clk, %c1_i1, %rst, %extracted_bits[...]
   * %out = comb.concat %fdse_outs[...] : (i1, i1, i1, i1) -> i4
   * ```
   */
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

      // Determine if we should use FDSE (set) or FDRE (reset) based on reset
      // value
      ResetType resetType = ResetType::Unknown;
      if (hasReset) {
        std::optional<APInt> singleBitResetValueOpt =
            multiBitCtx.get1BitResetValue();
        if (!singleBitResetValueOpt) {
          op->emitError() << "Failed to determine single bit reset value.";
          return failure();
        }
        // Pass the single-bit APInt to getResetType
        resetType = getResetType(op, *singleBitResetValueOpt);
      } else {
        // If no reset was originally provided, we synthesized a 0 reset, so use
        // SyncReset
        resetType = ResetType::SyncReset;
      }

      if (resetType == ResetType::Unknown) {
        op.emitError() << "Reset type is unknown. Cannot determine if we "
                          "should use FDSE or FDRE.";
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
        op.emitError() << "Reset type is unknown. Cannot determine if we "
                          "should use FDSE or FDRE.";
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

/**
 * @brief Conversion pattern for lowering `seq.compreg` to `xlnx.fdse` or `xlnx.fdre`.
 *
 * This pattern utilizes `CompRegRewriterHelper` to handle the common logic for
 * converting register-like operations. It specifically targets the `seq.compreg`
 * operation. Since `seq.compreg` does not have an explicit clock enable input,
 * this pattern provides a `getClockEnable` implementation that always returns a
 * constant `true` value, effectively treating the register as always enabled.
 * The conversion yields either `xlnx.fdse` (synchronous set) or `xlnx.fdre`
 * (synchronous reset) based on the reset value of the `seq.compreg`.
 */
struct CompRegLowering
    : public CompRegRewriterHelper<CompRegLowering, CompRegOp> {
  using CompRegRewriterHelper<CompRegLowering,
                              CompRegOp>::CompRegRewriterHelper;

  /**
   * @brief Finds or creates a constant value within a module.
   *
   * This helper function searches for or creates a constant with a specific type and value
   * within the given `hw.module`. It first searches the top-level operations of `hwModule`
   * for an existing `hw::ConstantOp` with the specified type and value.
   * If a matching constant is found, its result `Value` is returned.
   * If not found, a new `hw::ConstantOp` is created at the beginning of the module body,
   * and its result `Value` is returned. This approach helps reuse constants, avoid redundancy,
   * and keep the generated IR clean.
   *
   * @param op The current operation, used for error reporting and getting location info (if `hwModule` is invalid or has no body).
   * @param hwModule The parent `hw.module` to search or create the constant within.
   * @param type The desired MLIR type of the constant.
   * @param valueAttr The `IntegerAttr` value of the constant.
   * @param rewriter The `ConversionPatternRewriter` used to create the new constant op if needed.
   * @return The `Value` of the found or newly created constant. In error cases (e.g., invalid `hwModule`),
   *         it might fall back to creating a local constant at `op`'s location, but this usually indicates an
   *         unexpected IR structure upstream.
   */
  static Value
  getOrCreateConstantInModule(Operation *op, hw::HWModuleOp hwModule, Type type,
                              IntegerAttr valueAttr,
                              ConversionPatternRewriter &rewriter) {
    if (!hwModule) {
      op->emitError("Operation must be inside an hw.module");
      // Cannot directly signal failure from this helper function.
      // Fallback to creating a local constant, though this likely indicates
      // an unexpected IR structure upstream.
      return rewriter.create<hw::ConstantOp>(op->getLoc(), valueAttr);
    }

    Block *moduleBody = hwModule.getBodyBlock();
    if (!moduleBody) {
      // Should not happen for a valid HWModuleOp with a body.
      op->emitError("Parent hw.module has no body block");
      // Fallback similar to the !hwModule case.
      return rewriter.create<hw::ConstantOp>(op->getLoc(), valueAttr);
    }

    // Search the HW module's top-level operations for an existing constant
    // with the specified type and value.
    Value existingConstantValue;
    for (Operation &topOp : moduleBody->getOperations()) {
      if (auto constantOp = dyn_cast<hw::ConstantOp>(topOp)) {
        if (constantOp.getType() == type &&
            constantOp.getValueAttr() == valueAttr) {
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
    return rewriter.create<hw::ConstantOp>(hwModule.getLoc(), valueAttr);
  }

  /**
   * @brief Gets the clock enable signal for `seq.compreg`.
   *
   * `seq.compreg` does not have an explicit clock enable input, so it is always enabled.
   * This function returns a constant `Value` representing `true` (i1 value 1).
   * It uses `getOrCreateConstantInModule` to reuse or create this constant.
   *
   * @param op The original `CompRegOp` operation.
   * @param adaptor The operand adaptor (unused).
   * @param rewriter The rewriter used to create the constant if needed.
   * @return A constant i1 `Value` representing `true` (1).
   */
  static Value getClockEnable(
      CompRegOp op,
      CompRegRewriterHelper<CompRegLowering, CompRegOp>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) {
    // Find the parent HWModuleOp.
    auto hwModule = op->template getParentOfType<hw::HWModuleOp>();
    Type i1Type = rewriter.getI1Type();
    IntegerAttr trueAttr = rewriter.getIntegerAttr(i1Type, 1);

    // Use the extracted helper function to get or create a constant true value
    return getOrCreateConstantInModule(op, hwModule, i1Type, trueAttr,
                                       rewriter);
  }
};

/// @brief Conversion pattern for lowering `seq.compreg.ce` to `xlnx.fdse` or `xlnx.fdre`.
/// Uses `CompRegRewriterHelper` for the conversion logic.
/// Provides the `getClockEnable` implementation for `seq.compreg.ce` (returns its clock enable input).
struct CompRegCELowering
    : public CompRegRewriterHelper<CompRegCELowering, CompRegClockEnabledOp> {
  using CompRegRewriterHelper<CompRegCELowering,
                              CompRegClockEnabledOp>::CompRegRewriterHelper;

  /**
   * @brief Gets the clock enable signal for `seq.compreg.ce`.
   *
   * `seq.compreg.ce` has an explicit clock enable input. This function returns that input directly.
   *
   * @param op The original `CompRegClockEnabledOp` operation (unused).
   * @param adaptor The operand adaptor, used to access the clock enable input.
   * @param rewriter The rewriter (unused).
   * @return The clock enable input `Value` of the `seq.compreg.ce`.
   */
  static Value getClockEnable(
      CompRegClockEnabledOp op,
      CompRegRewriterHelper<CompRegCELowering, CompRegClockEnabledOp>::OpAdaptor
          adaptor,
      ConversionPatternRewriter &rewriter) {
    return adaptor.getClockEnable();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// SeqToXlnxPass
//===----------------------------------------------------------------------===//

namespace {
struct SeqToXlnxPass
    : public circt::impl::ConvertSeqToXlnxBase<SeqToXlnxPass> {
  void runOnOperation() override;
};
} // namespace

/**
 * @brief Configures the legality rules for the SeqToXlnx conversion.
 *
 * This function defines which operations are legal after the conversion and which must be converted:
 * - Marks `seq::CompRegOp` and `seq::CompRegClockEnabledOp` as illegal.
 * - Marks `XlnxDialect`, `mlir::BuiltinDialect`, `HWDialect` as legal.
 * - Marks necessary `hw::ConstantOp`, `hw::InstanceOp`, `comb::ExtractOp`, `comb::ConcatOp` as legal.
 * - Marks `seq::InitialOp` and `seq::YieldOp`, needed for initial values, as legal.
 *
 * @param target The conversion target to configure.
 */
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

/**
 * @brief Populates the pattern set with operation conversion patterns.
 *
 * Populates the given pattern set with the operation conversion patterns required for the SeqToXlnx conversion.
 * Specifically, it adds the patterns responsible for the actual operation conversion
 * (`CompRegLowering` and `CompRegCELowering`) to the `patterns` set. These patterns
 * will be used during `applyPartialConversion` to match and rewrite the corresponding
 * `seq::CompRegOp` and `seq::CompRegClockEnabledOp` operations.
 *
 * @param patterns The rewrite pattern set to populate with conversion patterns.
 */
static void populateOpConversion(RewritePatternSet &patterns) {
  patterns.add<CompRegLowering, CompRegCELowering>(patterns.getContext());
}

/**
 * @brief Runs the conversion pass on an operation (typically the top-level module).
 *
 * This function sets up the conversion target and rewrite patterns, then applies partial conversion.
 * 1. Gets the MLIR context and the top-level module operation.
 * 2. Creates a ConversionTarget.
 * 3. Creates a RewritePatternSet.
 * 4. Calls `populateLegality` to configure which operations are legal or illegal.
 * 5. Calls `populateOpConversion` to add conversion patterns.
 * 6. Applies the conversion using `applyPartialConversion`.
 * 7. Signals pass failure if the conversion fails.
 */
void SeqToXlnxPass::runOnOperation() {
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

/**
 * @brief Creates an instance of the SeqToXlnx conversion pass.
 *
 * This is the factory function used for registering and creating the pass.
 * @return A unique pointer (`std::unique_ptr`) to the newly created `SeqToXlnxPass` instance.
 */
std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertSeqToXlnxPass() {
  return std::make_unique<SeqToXlnxPass>();
}