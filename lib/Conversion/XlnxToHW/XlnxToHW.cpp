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
#include "llvm/Support/FormatVariadic.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTXLNXTOHW
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;
using namespace xlnx;

namespace {
struct XlnxOpToHWInstLowering
    : public OpInterfaceConversionPattern<xlnx::HWInstable> {
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;

  XlnxOpToHWInstLowering(MLIRContext *context)
      : OpInterfaceConversionPattern<xlnx::HWInstable>(context, /*benefit=*/1) {
  }

private:
  /**
   * @brief Processes port information for an xlnx::HWInstable operation.
   *
   * Iterates over the operation's operands and results. Based on the
   * operation's `portDict` attribute, it maps them to corresponding port
   * names, types, and directions. Fills the `modulePorts` vector (for module
   * signature) and `instanceInputs` vector (for instance inputs).
   *
   * @param op The xlnx::HWInstable operation to process.
   * @param rewriter The conversion pattern rewriter (currently unused but
   *                 kept for future extensions).
   * @param modulePorts [Output] Vector to be filled with module port
   *                    information.
   * @param instanceInputs [Output] Vector to be filled with instance input
   *                       values.
   * @return Returns success() if all ports were processed successfully;
   *         otherwise, returns failure() and emits an error.
   *
   * @par Example
   * Assume an `xlnx::HWInstable` operation `op` with the following features:
   * - Operands: `%in1 : i32`, `%in2 : i1`
   * - Results: `%out1 : i1`
   * - `portDict`: `{"A": %in1, "B": %in2, "Y": %out1}`
   *
   * After calling `processPorts(op, ..., modulePorts, instanceInputs)`:
   * - `modulePorts` will contain:
   *   - `{ name: "A", type: i32, dir: Input, argNum: 0 }`
   *   - `{ name: "B", type: i1,  dir: Input, argNum: 1 }`
   *   - `{ name: "Y", type: i1,  dir: Output, argNum: 0 }`
   * - `instanceInputs` will contain: `[%in1, %in2]`
   */
  LogicalResult processPorts(xlnx::HWInstable op,
                             ConversionPatternRewriter &rewriter,
                             SmallVectorImpl<hw::PortInfo> &modulePorts,
                             SmallVectorImpl<Value> &instanceInputs) const {
    MLIRContext *context = op->getContext();
    auto portDict = op.getPortDict();
    DenseMap<Value, StringRef> valueToPortName;
    for (auto const &[name, val] : portDict) {
      valueToPortName[val] = name;
    }

    // Process input ports
    for (Value operand : op->getOperands()) {
      auto it = valueToPortName.find(operand);
      if (it != valueToPortName.end()) {
        modulePorts.push_back(hw::PortInfo{
            ModulePort{
                StringAttr::get(context, it->second), // name
                operand.getType(),                    // type
                hw::ModulePort::Direction::Input      // dir
            },
            instanceInputs.size() // argNum
        });
        instanceInputs.push_back(operand);
        valueToPortName.erase(it); // Mark as processed
      } else {
        return op.emitError("Operand not found in port dictionary");
      }
    }

    // Process output ports
    size_t resultIndex = 0;
    for (Value result : op->getResults()) {
      auto it = valueToPortName.find(result);
      if (it != valueToPortName.end()) {
        modulePorts.push_back(hw::PortInfo{
            ModulePort{
                StringAttr::get(context, it->second), // name
                result.getType(),                     // type
                hw::ModulePort::Direction::Output     // dir
            },
            resultIndex++ // argNum
        });
        valueToPortName.erase(it); // Mark as processed
      } else {
        return op.emitError("Result not found in port dictionary");
      }
    }

    // Check if all ports were processed
    if (!valueToPortName.empty()) {
      return op.emitError("Not all ports in port dictionary were matched to "
                          "operands or results");
    }
    return success();
  }

  /**
   * @brief Gets or creates an hw::HWModuleExternOp matching the given gate
   * type and port signature.
   *
   * First searches the parent module for an existing `hw::HWModuleExternOp`
   * with the same `gateType`. If found, validates that its port signature
   * matches `modulePorts`. If not found or the signature doesn't match (a
   * detailed error will be emitted using the `getDirectionString` helper
   * function), a new `hw::HWModuleExternOp` is created. The newly created
   * module will contain attributes from the original `op` (excluding
   * name-related attributes).
   *
   * @param op The original xlnx::HWInstable operation (for context, location,
   *           and attributes).
   * @param rewriter The conversion pattern rewriter, used to create the new
   *                 external module.
   * @param gateType The name of the external module to find or create.
   * @param modulePorts The expected module port signature.
   * @return A FailureOr containing the found or created hw::HWModuleExternOp;
   *         returns failure if the signature mismatches.
   *
   * @par Example
   * Assume `gateType = "AND_GATE"` and `modulePorts` defines a signature
   * for a two-input, single-output AND gate.
   *
   * 1. **If `hw.module.extern @AND_GATE` already exists in the parent module
   *    with a matching signature:**
   *    The function will return `Success(existingModule)` pointing to the
   *    existing module.
   *
   * 2. **If `hw.module.extern @AND_GATE` does not exist in the parent
   * module:** The function will create a new `hw.module.extern @AND_GATE` at
   * the beginning of the parent module with the signature defined by
   * `modulePorts` and copy relevant attributes from `op`. It will then return
   * `Success(newModule)` pointing to the new module.
   *
   * 3. **If `hw.module.extern @AND_GATE` exists in the parent module but the
   *    signature mismatches:**
   *    The function will emit an error and return `Failure`.
   */
  FailureOr<hw::HWModuleExternOp> getOrCreateExternModule(
      xlnx::HWInstable op, ConversionPatternRewriter &rewriter,
      StringRef gateType, ArrayRef<hw::PortInfo> modulePorts) const {
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    MLIRContext *context = op->getContext();
    Location loc = op->getLoc();
    hw::HWModuleExternOp externModule;

    // Search for existing extern module
    for (auto mod : parentModule.getOps<hw::HWModuleExternOp>()) {
      if (mod.getName() == gateType) {
        // Validate signature of existing module
        auto existingPorts = mod.getPortList();
        if (existingPorts.size() != modulePorts.size()) {
          op.emitError("Found existing extern module '")
              << gateType << "' but port count mismatch (expected "
              << modulePorts.size() << ", found " << existingPorts.size()
              << ")";
          return failure();
        }

        bool signatureMatch = true;
        for (size_t i = 0; i < modulePorts.size(); ++i) {
          if (modulePorts[i].name != existingPorts[i].name ||
              modulePorts[i].type != existingPorts[i].type ||
              modulePorts[i].dir != existingPorts[i].dir) {
            signatureMatch = false;
            op.emitError("Found existing extern module '")
                << gateType << "' but port signature mismatch at index " << i
                << ": expected ('" << modulePorts[i].name.getValue() << "', "
                << modulePorts[i].type << ", "
                << getDirectionString(modulePorts[i].dir) << "), found ('"
                << existingPorts[i].name.getValue() << "', "
                << existingPorts[i].type << ", "
                << getDirectionString(existingPorts[i].dir) << ")";
            break; // Report first mismatch
          }
        }

        if (!signatureMatch) {
          return failure();
        }

        externModule = mod; // Found matching module
        break;
      }
    }

    // Create new extern module if not found
    if (!externModule) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(parentModule.getBody());
      externModule = rewriter.create<hw::HWModuleExternOp>(
          loc, StringAttr::get(context, gateType), modulePorts);
      externModule->setAttr("parameters", getParameters(op, rewriter, true));
    }
    return externModule;
  }

  /**
   * @brief Converts an hw::ModulePort::Direction enum to its string
   * representation.
   * @param dir The direction enum value to convert.
   * @return The string representation of the direction ("Input", "Output",
   * "InOut"), or "Unknown".
   */
  StringRef getDirectionString(hw::ModulePort::Direction dir) const {
    switch (dir) {
    case hw::ModulePort::Direction::Input:
      return "Input";
    case hw::ModulePort::Direction::Output:
      return "Output";
    case hw::ModulePort::Direction::InOut:
      return "InOut";
    }
    return "Unknown"; // Should not happen
  }

  /**
   * @brief Gets the parameter list for an xlnx::HWInstable operation.
   *
   * Iterates over the operation's attribute dictionary (either the default or
   * regular dictionary based on the `inDefault` parameter), converts each
   * attribute to an hw::ParamDeclAttr, and adds it to the parameter list.
   * Returns an ArrayAttr containing all parameters.
   *
   * @param op The xlnx::HWInstable operation to get parameters from.
   * @param rewriter The conversion pattern rewriter, used to create the
   *                 ArrayAttr.
   * @param inDefault If true, uses the operation's default attribute dictionary
   *                  (`getDefaultAttrDict`); otherwise, uses the regular
   *                  attribute dictionary (`getAttrDict`).
   * @return An ArrayAttr containing all parameters.
   *
   * @par Example
   * Assume `op` is an xlnx::HWInstable operation with the following attributes:
   * - Regular attributes: `{"param1": 42, "param2": true}`
   * - Default attributes: `{"default_param": "hello"}`
   * 
   * After calling `getParameters(op, rewriter, false)` (or without the third
   * argument):
   * - The returned ArrayAttr will contain two hw::ParamDeclAttr from the regular
   * attributes:
   *   - `hw::ParamDeclAttr::get(StringAttr::get(context, "param1"), i32, IntegerAttr::get(context, 42))`
   *   - `hw::ParamDeclAttr::get(StringAttr::get(context, "param2"), i1, IntegerAttr::get(context, true))`
   * 
   * After calling `getParameters(op, rewriter, true)`:
   * - The returned ArrayAttr will contain one hw::ParamDeclAttr from the default
   * attributes:
   *   - `hw::ParamDeclAttr::get(StringAttr::get(context, "default_param"), string, StringAttr::get(context, "hello"))`
   */
  ArrayAttr getParameters(xlnx::HWInstable op,
                          ConversionPatternRewriter &rewriter, bool inDefault = false) const {
    SmallVector<Attribute> parameters;
    auto context = rewriter.getContext();
    auto dict = inDefault ? op.getDefaultAttrDict() : op.getAttrDict();
    for (auto const &[name, typedAttr] : dict) {
      parameters.push_back(hw::ParamDeclAttr::get(name, typedAttr));
    }
    return ArrayAttr::get(context, parameters);
  }

public:
  /**
   * @brief Converts an xlnx::HWInstable operation to an hw::InstanceOp.
   *
   * This method performs the following steps:
   * 1. Checks if the operation is from the xlnx dialect.
   * 2. Processes the operation's port information, including input and output
   *    ports.
   * 3. Gets or creates an hw::HWModuleExternOp matching the operation's gate
   *    type and port signature.
   * 4. Creates an hw::InstanceOp to instantiate the external module.
   * 5. Sets the regular attributes from the original operation (obtained via
   *    `getParameters(op, rewriter, false)`) as the `parameters` attribute
   *    of the instance.
   * 6. Replaces the original operation's results with the results of the newly
   *    created instance operation.
   */
  LogicalResult
  matchAndRewrite(xlnx::HWInstable op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Check if the operation is from the Xlnx dialect
    if (op->getDialect()->getNamespace() != "xlnx") {
      return failure();
    }

    Location loc = op->getLoc();
    // Store the gate type in a std::string to avoid dangling reference
    std::string gateType = op.getGateType();
    SmallVector<hw::PortInfo> modulePorts;
    SmallVector<Value> instanceInputs;

    // 1. Process Ports
    if (failed(processPorts(op, rewriter, modulePorts, instanceInputs))) {
      return failure(); // Errors emitted within processPorts
    }

    // 2. Get or Create HWModuleExternOp
    FailureOr<hw::HWModuleExternOp> externModuleResult =
        getOrCreateExternModule(op, rewriter, gateType, modulePorts);
    if (failed(externModuleResult)) {
      return failure(); // Errors emitted within getOrCreateExternModule
    }
    hw::HWModuleExternOp externModule = externModuleResult.value();

    // 3. Create hw.instance operation
    auto instanceOp =
        rewriter.create<hw::InstanceOp>(loc, externModule, // Module reference
                                        "", // Instance name (as StringRef)
                                        instanceInputs // Input operands only
        );

    // Add attributes from the original operation to the instance, avoiding name
    // clashes
    instanceOp->setAttr("parameters", getParameters(op, rewriter));

    // 4. Replace original operation results
    rewriter.replaceOp(op, instanceOp.getResults());

    return success();
  }
};

struct XlnxToHWPass : public circt::impl::ConvertXlnxToHWBase<XlnxToHWPass> {
  /**
   * @brief Configures the conversion target, defining which dialects are legal.
   * @param target The conversion target to configure.
   */
  void populateLegality(ConversionTarget &target) {
    target.addIllegalDialect<xlnx::XlnxDialect>();

    // Mark HW and Builtin dialects as legal targets
    target.addLegalDialect<hw::HWDialect, mlir::BuiltinDialect>();
  }

  /**
   * @brief Adds operation conversion patterns to the pattern set.
   * @param patterns The RewritePatternSet to add patterns to.
   */
  void populateOpConversion(RewritePatternSet &patterns) {
    // Add the XlnxOpToHWInstLowering pattern to convert all Xlnx dialect
    // operations that implement the HWInstable interface
    patterns.add<XlnxOpToHWInstLowering>(patterns.getContext());
  }

  /**
   * @brief Runs the conversion pass on the module.
   *
   * Sets up the conversion target and rewrite patterns, then applies the
   * partial conversion.
   */
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

/**
 * @brief Creates an instance of the Xlnx to HW conversion pass.
 * @return A unique pointer to the newly created pass instance.
 */
std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertXlnxToHWPass() {
  return std::make_unique<XlnxToHWPass>();
}