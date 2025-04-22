//===- XlnxToHW.cpp - Convert Xlnx to HW ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Xlnx to HW conversion pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/XlnxToHW.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Xlnx/XlnxDialect.h"
#include "circt/Dialect/Xlnx/XlnxOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTXLNXTOHW
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;
using namespace xlnx;

namespace {
struct XlnxOpToHWInstLowering : public OpInterfaceConversionPattern<xlnx::HWInstable> {
  XlnxOpToHWInstLowering(MLIRContext *context)
      : OpInterfaceConversionPattern<xlnx::HWInstable>(context, /*benefit=*/1) {
  }

  LogicalResult
  matchAndRewrite(xlnx::HWInstable op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // 首先检查操作是否来自 Xlnx 方言
    if (op->getDialect()->getNamespace() != "xlnx") {
      return failure();
    }

    // 为了调试，打印匹配到的操作
    llvm::errs() << "Matched operation: " << op->getName() << "\n";

    // TODO: 稍后实现转换逻辑

    return success();
  }
};

struct XlnxToHWPass : public circt::impl::ConvertXlnxToHWBase<XlnxToHWPass> {
  void populateLegality(ConversionTarget &target) {
    target.addIllegalDialect<xlnx::XlnxDialect>();

    // 将 HW 和 Builtin 方言标记为合法目标
    target.addLegalDialect<hw::HWDialect, mlir::BuiltinDialect>();
  }

  void populateOpConversion(RewritePatternSet &patterns) {
    // 添加 XlnxOpToHWInstLowering 模式来转换所有实现了 HWInstable 接口的 Xlnx 方言操作
    patterns.add<XlnxOpToHWInstLowering>(patterns.getContext());
  }

  void runOnOperation() override {
    MLIRContext &context = getContext();
    ModuleOp module = getOperation();

    IRRewriter rewriter(module);

    ConversionTarget target(context);
    RewritePatternSet patterns(&context);

    // Configure legality and add conversion patterns
    populateLegality(target);
    populateOpConversion(patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

// Function to create the pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertXlnxToHWPass() {
  return std::make_unique<XlnxToHWPass>();
}