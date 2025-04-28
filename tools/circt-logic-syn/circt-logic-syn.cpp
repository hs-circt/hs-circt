//===- circt-lec.cpp - The circt-lec driver ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file initiliazes the 'circt-logic-syn' tool, it will do logic synthesis
/// on the input circt core dialect module and output a gate level netlist
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Support/Passes.h"
#include "circt/Support/Version.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

// mockturtle includes
#include "fmt/format.h"
#include "lorina/aiger.hpp"
#include "lorina/genlib.hpp"
#include "mockturtle/algorithms/aig_balancing.hpp"
#include "mockturtle/algorithms/lut_mapper.hpp"
#include "mockturtle/generators/arithmetic.hpp"
#include "mockturtle/io/aiger_reader.hpp"
#include "mockturtle/io/genlib_reader.hpp"
#include "mockturtle/io/write_blif.hpp"
#include "mockturtle/io/write_verilog.hpp"
#include "mockturtle/networks/aig.hpp"
#include "mockturtle/utils/name_utils.hpp"
#include "mockturtle/utils/tech_library.hpp"
#include "mockturtle/views/cell_view.hpp"
#include "mockturtle/views/depth_view.hpp"
#include "mockturtle/views/names_view.hpp"

namespace cl = llvm::cl;

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Command-line options declaration
//===----------------------------------------------------------------------===//

static cl::OptionCategory mainCategory("circt-logic-syn Options");

static cl::opt<std::string> inputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(mainCategory));

static cl::opt<std::string> topName("top", cl::desc("Top module name"),
                                    cl::value_desc("name"), cl::init(""),
                                    cl::cat(mainCategory));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    verbosePassExecutions("verbose-pass-executions",
                          cl::desc("Log executions of toplevel module passes"),
                          cl::init(false), cl::cat(mainCategory));

enum OutputFormat { OutputMLIR, OutputVerilog };
static cl::opt<OutputFormat> outputFormat(
    cl::desc("Specify output format"),
    cl::values(clEnumValN(OutputMLIR, "emit-mlir", "Emit LLVM MLIR dialect"),
               clEnumValN(OutputVerilog, "emit-verilog", "Emit Verilog")),
    cl::init(OutputMLIR), cl::cat(mainCategory));

//===----------------------------------------------------------------------===//
// Tool implementation
//===----------------------------------------------------------------------===//

void tryMockturtle() {
  using namespace mockturtle;

  aig_network aig, tmpaig;

  std::vector<aig_network::signal> a(8), b(8);
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

  fmt::print("[i] END2\n");
  fmt::print("[i] Lut mapper success\n");
}

LogicalResult executeLogicSyn(MLIRContext &context) {
  // auto module = parseSourceFile<ModuleOp>(inputFilename, &context);

  // auto outFile = openOutputFile(outputFilename);

  // module->print(outFile->os());
  // outFile->keep();

  tryMockturtle();

  return success();
}

//===----------------------------------------------------------------------===//
// Conversion Infrastructure
//===----------------------------------------------------------------------===//
static void populateSynthesisPipeline(PassManager &pm) {
  auto pipeline = [](OpPassManager &mpm) {
    // Add the AIG to Comb at the scope exit if requested.
    auto addAIGToComb =
        llvm::make_scope_exit([&]() { mpm.addPass(createCSEPass()); });

    // {
    //   // Partially legalize Comb to AIG, run CSE and canonicalization.
    //   circt::ConvertCombToAIGOptions options;
    //   partiallyLegalizeCombToAIG<comb::AndOp, comb::OrOp, comb::XorOp,
    //                              comb::MuxOp, comb::ICmpOp, hw::ArrayGetOp,
    //                              hw::ArraySliceOp, hw::ArrayCreateOp,
    //                              hw::ArrayConcatOp, hw::AggregateConstantOp>(
    //       options.additionalLegalOps);
    //   mpm.addPass(circt::createConvertCombToAIG(options));
    // }
    mpm.addPass(createCSEPass());
  };

  if (topName.empty()) {
    pipeline(pm.nest<hw::HWModuleOp>());
  } else {
    pm.addPass(circt::createHierarchicalRunner(topName, pipeline));
  }
  // TODO: Add LUT mapping, etc.
}

/// The entry point for the `circt-logic-syn` tool:
int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  cl::HideUnrelatedOptions(mainCategory);

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream &os) { os << circt::getCirctVersion() << '\n'; });

  // Parse the command-line options provided by the user.
  cl::ParseCommandLineOptions(
      argc, argv,
      "circt-logic-syn - logic synthesis tool\n\n"
      "\tThis tool synthesizes circt HW modules to a gate level netlist.\n");

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  // Register the supported CIRCT dialects and create a context to work with.
  DialectRegistry registry;
  // clang-format off
  registry.insert<
    circt::comb::CombDialect,
    circt::seq::SeqDialect,
    circt::hw::HWDialect,
    mlir::BuiltinDialect
  >();
  // clang-format on
  // mlir::func::registerInlinerExtension(registry);
  MLIRContext context(registry);

  // Setup of diagnostic handling.
  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  // Avoid printing a superfluous note on diagnostic emission.
  context.printOpOnDiagnostic(false);

  // Perform the logical equivalence checking; using `exit` to avoid the slow
  // teardown of the MLIR context.
  exit(failed(executeLogicSyn(context)));
}
