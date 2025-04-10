#include "circt/Conversion/CoreMemoryMapping.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cmath>

namespace circt {
#define GEN_PASS_DEF_COREMEMORYMAPPING
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace moore;

using comb::ICmpPredicate;
using llvm::SmallDenseSet;

//===----------------------------------------------------------------------===//
// Core Memory Mapping Pass
//===----------------------------------------------------------------------===//

namespace {
struct CoreMemoryMappingPass
    : public circt::impl::CoreMemoryMappingBase<CoreMemoryMappingPass> {
  void runOnOperation() override;
};
} // namespace

/// Create a Core Memory Mapping pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::createCoreMemoryMappingPass() {
  return std::make_unique<CoreMemoryMappingPass>();
}

llvm::SmallVector<Operation *> analyseHwArray(ModuleOp module) {
  llvm::SmallVector<Operation *> arrayCreates;
  module.walk([&](hw::ArrayCreateOp op) { arrayCreates.push_back(op); });
  module.walk([&](hw::ArrayConcatOp op) { arrayCreates.push_back(op); });
  module.walk([&](hw::ArraySliceOp op) { arrayCreates.push_back(op); });
  return arrayCreates;
}

llvm::SmallVector<seq::FirMemOp> analyseFirMem(ModuleOp module) {
  llvm::SmallVector<seq::FirMemOp> firMemOps;
  module.walk([&](seq::FirMemOp op) { // cascade = true
    firMemOps.push_back(op);
  });
  return firMemOps;
}

hw::HWModuleExternOp create32Dram6(ModuleOp module, Operation *op,
                                   MLIRContext &context, int module_num) {
  OpBuilder builder(&context);
  builder.setInsertionPointToStart(module.getBody());
  SmallVector<hw::PortInfo> ports;

  StringAttr moduleName =
      builder.getStringAttr("RAM32M_" + std::to_string(module_num));

  ports.push_back(hw::PortInfo{builder.getStringAttr("DIA"),
                               builder.getIntegerType(2),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DIB"),
                               builder.getIntegerType(2),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DIC"),
                               builder.getIntegerType(2),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DID"),
                               builder.getIntegerType(2),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("ADDRA"),
                               builder.getIntegerType(5),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("ADDRB"),
                               builder.getIntegerType(5),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("ADDRC"),
                               builder.getIntegerType(5),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("ADDRD"),
                               builder.getIntegerType(5),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("WE"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("WCLK"),
                               seq::ClockType::get(&context),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DOA"),
                               builder.getIntegerType(2),
                               hw::ModulePort::Direction::Output});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DOB"),
                               builder.getIntegerType(2),
                               hw::ModulePort::Direction::Output});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DOC"),
                               builder.getIntegerType(2),
                               hw::ModulePort::Direction::Output});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DOD"),
                               builder.getIntegerType(2),
                               hw::ModulePort::Direction::Output});
  auto externmodule =
      builder.create<hw::HWModuleExternOp>(op->getLoc(), moduleName, ports);
  return externmodule;
}

hw::HWModuleExternOp create64Dram3(ModuleOp module, Operation *op,
                                   MLIRContext &context, int module_num) {
  OpBuilder builder(&context);
  builder.setInsertionPointToStart(module.getBody());
  SmallVector<hw::PortInfo> ports;

  StringAttr moduleName =
      builder.getStringAttr("RAM64M_" + std::to_string(module_num));

  ports.push_back(hw::PortInfo{builder.getStringAttr("DIA"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DIB"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DIC"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DID"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("ADDRA"),
                               builder.getIntegerType(6),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("ADDRB"),
                               builder.getIntegerType(6),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("ADDRC"),
                               builder.getIntegerType(6),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("ADDRD"),
                               builder.getIntegerType(6),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("WE"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("WCLK"),
                               seq::ClockType::get(&context),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DOA"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Output});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DOB"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Output});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DOC"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Output});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DOD"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Output});
  auto externmodule =
      builder.create<hw::HWModuleExternOp>(op->getLoc(), moduleName, ports);
  return externmodule;
}

hw::HWModuleExternOp create32Dram(ModuleOp module, Operation *op,
                                  MLIRContext &context, int module_num) {
  OpBuilder builder(&context);
  builder.setInsertionPointToStart(module.getBody());
  SmallVector<hw::PortInfo> ports;

  StringAttr moduleName =
      builder.getStringAttr("RAM32X1D_" + std::to_string(module_num));

  ports.push_back(hw::PortInfo{builder.getStringAttr("WCLK"),
                               seq::ClockType::get(&context),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("D"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("WE"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("A"),
                               builder.getIntegerType(5),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DPRA"),
                               builder.getIntegerType(5),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DPO"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Output});
  ports.push_back(hw::PortInfo{builder.getStringAttr("SPO"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Output});

  auto externmodule =
      builder.create<hw::HWModuleExternOp>(op->getLoc(), moduleName, ports);

  return externmodule;
}

hw::HWModuleExternOp create64Dram(ModuleOp module, Operation *op,
                                  MLIRContext &context, int module_num) {
  OpBuilder builder(&context);
  builder.setInsertionPointToStart(module.getBody());
  SmallVector<hw::PortInfo> ports;

  StringAttr moduleName =
      builder.getStringAttr("RAM64X1D_" + std::to_string(module_num));

  ports.push_back(hw::PortInfo{builder.getStringAttr("WCLK"),
                               seq::ClockType::get(&context),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("D"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("WE"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("A"),
                               builder.getIntegerType(6),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DPRA"),
                               builder.getIntegerType(6),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DPO"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Output});
  ports.push_back(hw::PortInfo{builder.getStringAttr("SPO"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Output});

  auto externmodule =
      builder.create<hw::HWModuleExternOp>(op->getLoc(), moduleName, ports);

  return externmodule;
}

hw::HWModuleExternOp create128Dram(ModuleOp module, Operation *op,
                                   MLIRContext &context, int module_num) {
  OpBuilder builder(&context);
  builder.setInsertionPointToStart(module.getBody());
  SmallVector<hw::PortInfo> ports;

  StringAttr moduleName =
      builder.getStringAttr("RAM128X1D_" + std::to_string(module_num));

  ports.push_back(hw::PortInfo{builder.getStringAttr("WCLK"),
                               seq::ClockType::get(&context),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("D"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("WE"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("A"),
                               builder.getIntegerType(8),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DPRA"),
                               builder.getIntegerType(8),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DPO"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Output});
  ports.push_back(hw::PortInfo{builder.getStringAttr("SPO"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Output});

  auto externmodule =
      builder.create<hw::HWModuleExternOp>(op->getLoc(), moduleName, ports);

  return externmodule;
}

hw::HWModuleExternOp create256Dram(ModuleOp module, Operation *op,
                                   MLIRContext &context, int module_num) {
  OpBuilder builder(&context);
  builder.setInsertionPointToStart(module.getBody());
  SmallVector<hw::PortInfo> ports;

  StringAttr moduleName =
      builder.getStringAttr("RAM256X1D_" + std::to_string(module_num));

  ports.push_back(hw::PortInfo{builder.getStringAttr("WCLK"),
                               seq::ClockType::get(&context),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("D"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("WE"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("A"),
                               builder.getIntegerType(8),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DPRA"),
                               builder.getIntegerType(8),
                               hw::ModulePort::Direction::Input});
  ports.push_back(hw::PortInfo{builder.getStringAttr("DPO"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Output});
  ports.push_back(hw::PortInfo{builder.getStringAttr("SPO"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Output});

  auto externmodule =
      builder.create<hw::HWModuleExternOp>(op->getLoc(), moduleName, ports);

  return externmodule;
}

hw::HWModuleExternOp createDram(ModuleOp module, Operation *op,
                                MLIRContext &context, int module_num,
                                int bitsize, int logdepth) {
  if (logdepth <= 5) {
    return create32Dram(module, op, context, module_num);
  } else if (logdepth <= 6) {
    return create64Dram(module, op, context, module_num);
  } else if (logdepth <= 7) {
    return create128Dram(module, op, context, module_num);
  } else if (logdepth <= 8) {
    return create256Dram(module, op, context, module_num);
  } else {
    assert(false && "logdepth is too large");
  }
}

hw::HWModuleExternOp createDram6(ModuleOp module, Operation *op,
                                 MLIRContext &context, int module_num) {
  return create32Dram6(module, op, context, module_num);
}

hw::HWModuleExternOp createDram3(ModuleOp module, Operation *op,
                                 MLIRContext &context, int module_num) {
  return create64Dram3(module, op, context, module_num);
}

hw::HWModuleExternOp createModuleExtern(ModuleOp op, MLIRContext &context,
                                        Type ADDRAtype, Type DIADItype,
                                        Type DOADOtype, Type MASKtype,
                                        int module_num) {
  OpBuilder builder(&context);
  builder.setInsertionPointToStart(op.getBody());
  StringAttr moduleName =
      builder.getStringAttr("RAMB36E2_" + std::to_string(module_num));
  SmallVector<hw::PortInfo> ports;
  ports.push_back(hw::PortInfo{builder.getStringAttr("CLKARDCLK"),
                               seq::ClockType::get(&context),
                               hw::ModulePort::Direction::Input});

  ports.push_back(hw::PortInfo{builder.getStringAttr("ADDRARDADDR"), ADDRAtype,
                               hw::ModulePort::Direction::Input});

  ports.push_back(hw::PortInfo{builder.getStringAttr("WEA"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Input});

  ports.push_back(hw::PortInfo{builder.getStringAttr("ENA"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Input});

  ports.push_back(hw::PortInfo{builder.getStringAttr("DIADI"), DIADItype,
                               hw::ModulePort::Direction::Input});

  ports.push_back(hw::PortInfo{builder.getStringAttr("DOADO"), DOADOtype,
                               hw::ModulePort::Direction::Output});

  ports.push_back(hw::PortInfo{builder.getStringAttr("REGCEA"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Input});

  ports.push_back(hw::PortInfo{builder.getStringAttr("RSTRAMA"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Input});

  ports.push_back(hw::PortInfo{builder.getStringAttr("RSTREGA"),
                               builder.getIntegerType(1),
                               hw::ModulePort::Direction::Input});

  ports.push_back(hw::PortInfo{builder.getStringAttr("MASK"), MASKtype,
                               hw::ModulePort::Direction::Input});

  auto externmodule =
      builder.create<hw::HWModuleExternOp>(op->getLoc(), moduleName, ports);

  return externmodule;
}

hw::InstanceOp
createInsetance(ModuleOp module, seq::FirMemOp op, seq::FirMemReadOp readop,
                MLIRContext &context, int &module_num,
                llvm::SmallVector<Value> &SymbolInputs,
                llvm::DenseMap<int, hw::HWModuleExternOp> &dram_map,
                int bitsize, int logdepth, OpBuilder &builder) {
  SymbolInputs.push_back(readop.getAddress());
  auto externop = dram_map[logdepth];
  if (externop == nullptr) {
    externop = createDram(module, op, context, module_num++, bitsize, logdepth);
    dram_map[logdepth] = externop;
  }
  if (SymbolInputs[4].getType().getIntOrFloatBitWidth() < logdepth) {
    auto consttype =
        logdepth - SymbolInputs[4].getType().getIntOrFloatBitWidth();
    auto constant32 = builder.create<hw::ConstantOp>(
        op->getLoc(), builder.getIntegerType(consttype),
        builder.getIntegerAttr(builder.getIntegerType(consttype), 0));
    auto concatop = builder.create<comb::ConcatOp>(
        op->getLoc(), constant32.getResult(), SymbolInputs[4]);
    SymbolInputs[4] = concatop.getResult();
  }
  if (SymbolInputs[3].getType().getIntOrFloatBitWidth() < logdepth) {
    auto consttype =
        logdepth - SymbolInputs[3].getType().getIntOrFloatBitWidth();
    auto constant32 = builder.create<hw::ConstantOp>(
        op->getLoc(), builder.getIntegerType(consttype),
        builder.getIntegerAttr(builder.getIntegerType(consttype), 0));
    auto concatop = builder.create<comb::ConcatOp>(
        op->getLoc(), constant32.getResult(), SymbolInputs[3]);
    SymbolInputs[3] = concatop.getResult();
  }
  auto instancemodule = builder.create<hw::InstanceOp>(
      op.getLoc(), externop,
      builder.getStringAttr("RAM" + std::to_string(1 << logdepth) + "X1D_" +
                            std::to_string(module_num++)),
      SymbolInputs);
  return instancemodule;
}

void useX1D(ModuleOp module, seq::FirMemOp op, MLIRContext &context,
            int &module_num, llvm::SmallVector<Value> &SymbolInputs,
            llvm::DenseMap<int, hw::HWModuleExternOp> &dram_map,
            OpBuilder &builder, int bitsize, int i, int depth,
            seq::FirMemReadOp readop, seq::FirMemWriteOp writeop,
            llvm::SmallVector<hw::InstanceOp> &dram_instances,
            llvm::SmallVector<Value> &dram_result) {
  auto extractop =
      builder.create<comb::ExtractOp>(op->getLoc(), writeop.getData(), i, 1);
  SymbolInputs.push_back(writeop.getClk());
  SymbolInputs.push_back(extractop.getResult());
  SymbolInputs.push_back(writeop.getEnable());
  SymbolInputs.push_back(writeop.getAddress());

  hw::InstanceOp instancemodule;
  if (depth <= 32) {
    instancemodule =
        createInsetance(module, op, readop, context, module_num, SymbolInputs,
                        dram_map, bitsize, 5, builder);
    instancemodule->setAttr("INIT", builder.getI32IntegerAttr(0));
  } else if (depth <= 64) {
    instancemodule =
        createInsetance(module, op, readop, context, module_num, SymbolInputs,
                        dram_map, bitsize, 6, builder);
    instancemodule->setAttr("INIT", builder.getI64IntegerAttr(0));
  } else if (depth <= 128) {
    instancemodule =
        createInsetance(module, op, readop, context, module_num, SymbolInputs,
                        dram_map, bitsize, 7, builder);
    instancemodule->setAttr(
        "INIT", builder.getIntegerAttr(builder.getIntegerType(128), 0));
  } else if (depth <= 256) {
    instancemodule =
        createInsetance(module, op, readop, context, module_num, SymbolInputs,
                        dram_map, bitsize, 8, builder);
    instancemodule->setAttr(
        "INIT", builder.getIntegerAttr(builder.getIntegerType(256), 0));
  } else {
    assert(false && "depth is not supported in dram");
  }
  instancemodule->setAttr("IS_WCLK_INVERTED",
                          builder.getIntegerAttr(builder.getIntegerType(1), 0));
  dram_instances.push_back(instancemodule);
  dram_result.push_back(instancemodule.getResult(0));
}

void use6bitRAM(ModuleOp module, seq::FirMemOp op, MLIRContext &context,
                int &module_num, llvm::SmallVector<Value> &SymbolInputs,
                llvm::DenseMap<int, hw::HWModuleExternOp> &dram_map,
                OpBuilder &builder, int bitsize, int i, int depth,
                seq::FirMemReadOp readop, seq::FirMemWriteOp writeop,
                llvm::SmallVector<hw::InstanceOp> &dram_instances,
                llvm::SmallVector<Value> &dram_result,
                seq::FirMemReadOp sec_readop) {
  auto extractopa =
      builder.create<comb::ExtractOp>(op->getLoc(), writeop.getData(), i, 2);
  auto extractopb = builder.create<comb::ExtractOp>(
      op->getLoc(), writeop.getData(), i + 2, 2);
  auto extractopc = builder.create<comb::ExtractOp>(
      op->getLoc(), writeop.getData(), i + 4, 2);
  SymbolInputs.push_back(extractopa);
  SymbolInputs.push_back(extractopb);
  SymbolInputs.push_back(extractopc);
  auto constant = builder.create<hw::ConstantOp>(
      op->getLoc(), builder.getIntegerType(2),
      builder.getIntegerAttr(builder.getIntegerType(2), 0));
  SymbolInputs.push_back(constant.getResult()); // empty

  mlir::Value addra = readop.getAddress();
  int logdepth = log2(depth < 32 ? 32 : depth);

  if (addra.getType().getIntOrFloatBitWidth() < logdepth) {
    auto consttype = logdepth - addra.getType().getIntOrFloatBitWidth();
    auto constant32 = builder.create<hw::ConstantOp>(
        op->getLoc(), builder.getIntegerType(consttype),
        builder.getIntegerAttr(builder.getIntegerType(consttype), 0));
    auto concatop = builder.create<comb::ConcatOp>(
        op->getLoc(), constant32.getResult(), addra);
    addra = concatop.getResult();
  }
  mlir::Value writeaddr = writeop.getAddress();
  if (writeaddr.getType().getIntOrFloatBitWidth() < logdepth) {
    auto consttype = logdepth - writeaddr.getType().getIntOrFloatBitWidth();
    auto constant32 = builder.create<hw::ConstantOp>(
        op->getLoc(), builder.getIntegerType(consttype),
        builder.getIntegerAttr(builder.getIntegerType(consttype), 0));
    auto concatop = builder.create<comb::ConcatOp>(
        op->getLoc(), constant32.getResult(), writeaddr);
    writeaddr = concatop.getResult();
  }

  SymbolInputs.push_back(addra);               // addra
  SymbolInputs.push_back(addra);               // addrb
  SymbolInputs.push_back(addra);               // addrc
  SymbolInputs.push_back(writeaddr);           // addrd
  SymbolInputs.push_back(writeop.getEnable()); // ena
  SymbolInputs.push_back(writeop.getClk());    // clka
  auto externop = dram_map[logdepth * 10];     // avoid 6bitRAM and 1bit mixed
  if (externop == nullptr) {
    externop = createDram6(module, op, context, module_num++);
    dram_map[logdepth * 10] = externop;
  }
  auto instancemodule = builder.create<hw::InstanceOp>(
      op.getLoc(), externop,
      builder.getStringAttr("RAM32M_" + std::to_string(module_num++)),
      SymbolInputs);
  instancemodule->setAttr(
      "INIT_A", builder.getIntegerAttr(builder.getIntegerType(64), 0));
  instancemodule->setAttr(
      "INIT_B", builder.getIntegerAttr(builder.getIntegerType(64), 0));
  instancemodule->setAttr(
      "INIT_C", builder.getIntegerAttr(builder.getIntegerType(64), 0));
  instancemodule->setAttr(
      "INIT_D", builder.getIntegerAttr(builder.getIntegerType(64), 0));
  instancemodule->setAttr("IS_WCLK_INVERTED",
                          builder.getIntegerAttr(builder.getIntegerType(1), 0));
  dram_instances.push_back(instancemodule);
  dram_result.push_back(instancemodule.getResult(0));
  dram_result.push_back(instancemodule.getResult(1));
  dram_result.push_back(instancemodule.getResult(2));
} // DOA-DOC output，DOD empty

void use3bitRAM(ModuleOp module, seq::FirMemOp op, MLIRContext &context,
                int &module_num, llvm::SmallVector<Value> &SymbolInputs,
                llvm::DenseMap<int, hw::HWModuleExternOp> &dram_map,
                OpBuilder &builder, int bitsize, int i, int depth,
                seq::FirMemReadOp readop, seq::FirMemWriteOp writeop,
                llvm::SmallVector<hw::InstanceOp> &dram_instances,
                llvm::SmallVector<Value> &dram_result,
                seq::FirMemReadOp sec_readop) {
  auto extractopa =
      builder.create<comb::ExtractOp>(op->getLoc(), writeop.getData(), i, 1);
  auto extractopb = builder.create<comb::ExtractOp>(
      op->getLoc(), writeop.getData(), i + 1, 1);
  auto extractopc = builder.create<comb::ExtractOp>(
      op->getLoc(), writeop.getData(), i + 2, 1);
  SymbolInputs.push_back(extractopa);
  SymbolInputs.push_back(extractopb);
  SymbolInputs.push_back(extractopc);
  auto constant = builder.create<hw::ConstantOp>(
      op->getLoc(), builder.getIntegerType(1),
      builder.getIntegerAttr(builder.getIntegerType(1), 0));
  SymbolInputs.push_back(constant.getResult()); // empty

  mlir::Value addra = readop.getAddress();
  int logdepth = log2(depth < 64 ? 64 : depth);

  if (addra.getType().getIntOrFloatBitWidth() <
      logdepth) { // read address length and port length alignment
    auto consttype = logdepth - addra.getType().getIntOrFloatBitWidth();
    auto constant32 = builder.create<hw::ConstantOp>(
        op->getLoc(), builder.getIntegerType(consttype),
        builder.getIntegerAttr(builder.getIntegerType(consttype), 0));
    auto concatop = builder.create<comb::ConcatOp>(
        op->getLoc(), constant32.getResult(), addra);
    addra = concatop.getResult();
  }
  mlir::Value writeaddr = writeop.getAddress();
  if (writeaddr.getType().getIntOrFloatBitWidth() <
      logdepth) { // write address length and port length alignment
    auto consttype = logdepth - writeaddr.getType().getIntOrFloatBitWidth();
    auto constant32 = builder.create<hw::ConstantOp>(
        op->getLoc(), builder.getIntegerType(consttype),
        builder.getIntegerAttr(builder.getIntegerType(consttype), 0));
    auto concatop = builder.create<comb::ConcatOp>(
        op->getLoc(), constant32.getResult(), writeaddr);
    writeaddr = concatop.getResult();
  }

  SymbolInputs.push_back(addra);               // addra
  SymbolInputs.push_back(addra);               // addrb
  SymbolInputs.push_back(addra);               // addrc
  SymbolInputs.push_back(writeaddr);           // addrd
  SymbolInputs.push_back(writeop.getEnable()); // ena
  SymbolInputs.push_back(writeop.getClk());    // clka
  auto externop = dram_map[logdepth * 10];     // avoid 6bitRAM and 1bit mixed
  if (externop == nullptr) {
    externop = createDram3(module, op, context, module_num++);
    dram_map[logdepth * 10] = externop;
  }
  auto instancemodule = builder.create<hw::InstanceOp>(
      op.getLoc(), externop,
      builder.getStringAttr("RAM64M_" + std::to_string(module_num++)),
      SymbolInputs);
  instancemodule->setAttr(
      "INIT_A", builder.getIntegerAttr(builder.getIntegerType(64), 0));
  instancemodule->setAttr(
      "INIT_B", builder.getIntegerAttr(builder.getIntegerType(64), 0));
  instancemodule->setAttr(
      "INIT_C", builder.getIntegerAttr(builder.getIntegerType(64), 0));
  instancemodule->setAttr(
      "INIT_D", builder.getIntegerAttr(builder.getIntegerType(64), 0));
  instancemodule->setAttr("IS_WCLK_INVERTED",
                          builder.getIntegerAttr(builder.getIntegerType(1), 0));
  dram_instances.push_back(instancemodule);
  dram_result.push_back(instancemodule.getResult(0));
  dram_result.push_back(instancemodule.getResult(1));
  dram_result.push_back(instancemodule.getResult(2));
} // DOA-DOC output, DOD empty

void setDRam(ModuleOp module, seq::FirMemOp op, MLIRContext &context,
             int &module_num,
             llvm::DenseMap<int, hw::HWModuleExternOp> &dram_map) {
  OpBuilder builder(&context);
  auto bitsize = op.getType().getWidth();
  auto depth = op.getType().getDepth();
  llvm::SmallVector<hw::InstanceOp> dram_instances;
  seq::FirMemReadOp readop = nullptr, sec_readop = nullptr;
  seq::FirMemWriteOp writeop;
  for (auto &useop : op->getUses()) {
    if (auto readop1 = llvm::dyn_cast<seq::FirMemReadOp>(useop.getOwner())) {
      if (readop != nullptr) {
        sec_readop = readop;
      }
      readop = readop1;
    } else if (auto writeop1 =
                   llvm::dyn_cast<seq::FirMemWriteOp>(useop.getOwner())) {
      writeop = writeop1;
    } else {
      assert(
          false &&
          "firmemop.usesop is not a seq::FirMemReadOp or seq::FirMemWriteOp");
    }
  }
  if (writeop == nullptr) {
    assert(false && "writeop is nullptr");
  }
  if (readop == nullptr) {
    assert(false && "readop is nullptr");
  }
  builder.setInsertionPointAfter(writeop);

  llvm::SmallVector<Value> dram_result;

  for (int i = 0; i < bitsize; i++) {
    llvm::SmallVector<Value> SymbolInputs;
    if ((bitsize <= 12 || bitsize - i < 6) && sec_readop == nullptr) {
      useX1D(module, op, context, module_num, SymbolInputs, dram_map, builder,
             bitsize, i, depth, readop, writeop, dram_instances, dram_result);
    } else if (depth <= 32) {
      use6bitRAM(module, op, context, module_num, SymbolInputs, dram_map,
                 builder, bitsize, i, depth, readop, writeop, dram_instances,
                 dram_result, sec_readop);
      i += 5;
    } else if (depth <= 64) {
      use3bitRAM(module, op, context, module_num, SymbolInputs, dram_map,
                 builder, bitsize, i, depth, readop, writeop, dram_instances,
                 dram_result, sec_readop);
      i += 2;
    } else
      assert(false && "depth is not supported");
  }

  auto resultop = builder.create<comb::ConcatOp>(op->getLoc(), dram_result);
  readop->replaceAllUsesWith(resultop);
  readop->erase();
  writeop->erase();
  op->erase();
}

void setBram(ModuleOp module, seq::FirMemOp op, MLIRContext &context,
             int &module_num, OpBuilder &builder,
             llvm::StringMap<hw::HWModuleExternOp> &externmodule_map) {
  auto bitsize = op.getType().getWidth();
  auto depth = op.getType().getDepth();
  seq::FirMemReadWriteOp rwop;
  if (bitsize < 36864) {
    llvm::SmallVector<Value> SymbolInputs;
    llvm::SmallVector<Type> SymbolInputsType;
    if (auto read_write_op = llvm::dyn_cast<seq::FirMemReadWriteOp>(
            op->getUses().begin()->getOwner())) {
      rwop = read_write_op;
      SymbolInputs.push_back(read_write_op.getClk()); // clka

      auto addrValue =
          read_write_op.getAddress(); // get the first operand as address

      SymbolInputs.push_back(addrValue);                    // addra
      SymbolInputs.push_back(read_write_op.getMode());      // wea
      SymbolInputs.push_back(read_write_op.getEnable());    // ena
      SymbolInputs.push_back(read_write_op.getWriteData()); // diadi

      SymbolInputsType.push_back(addrValue.getType());
      SymbolInputsType.push_back(read_write_op.getWriteData().getType());

      auto regcea = builder.create<hw::ConstantOp>(
          op.getLoc(),
          builder.getIntegerType(1),                             // (i1)
          builder.getIntegerAttr(builder.getIntegerType(1), 1)); // REGCEA
      SymbolInputs.push_back(regcea.getResult());                // REGCEA

      auto zero = builder.create<hw::ConstantOp>(
          op.getLoc(),
          builder.getIntegerType(1), // (i1)
          builder.getIntegerAttr(builder.getIntegerType(1), 0));
      SymbolInputs.push_back(zero.getResult()); // RSTRAMA
      SymbolInputs.push_back(zero.getResult()); // RSTREGA
      mlir::Value mask = read_write_op.getMask();
      if (mask == nullptr) {
        mask = builder.create<hw::ConstantOp>(
            op.getLoc(),
            builder.getIntegerType(1), // (i1)
            builder.getIntegerAttr(builder.getIntegerType(1), 0));
      }
      SymbolInputs.push_back(mask); // MASK

      llvm::SmallVector<Type> resultTypes;
      resultTypes.push_back(read_write_op.getReadData().getType());

      SymbolInputsType.push_back(read_write_op.getReadData().getType());
      SymbolInputsType.push_back(mask.getType());
      std::string key =
          "RAMB36E2_" +
          std::to_string(SymbolInputsType[0].getIntOrFloatBitWidth()) + "_" +
          std::to_string(SymbolInputsType[1].getIntOrFloatBitWidth()) + "_" +
          std::to_string(SymbolInputsType[2].getIntOrFloatBitWidth()) + "_" +
          std::to_string(SymbolInputsType[3].getIntOrFloatBitWidth());

      auto it = externmodule_map.find(key);
      if (it == externmodule_map.end()) {
        auto newModule = createModuleExtern(
            module, context, SymbolInputsType[0], SymbolInputsType[1],
            SymbolInputsType[2], SymbolInputsType[3], module_num++);
        externmodule_map.insert({key, newModule});
        it = externmodule_map.find(key);
      }

      auto &externModule = it->second;
      auto instancemodule = builder.create<hw::InstanceOp>(
          op.getLoc(), externModule,
          builder.getStringAttr("RAMB36E2_" + std::to_string(module_num++)),
          SymbolInputs);

      instancemodule->setAttr("RAM_MODE", builder.getStringAttr("WRITE_FIRST"));
      instancemodule->setAttr("WRITE_MODE_A",
                              builder.getStringAttr("WRITE_FIRST"));
      instancemodule->setAttr(
          "READ_WIDTH_A",
          builder.getIntegerAttr(builder.getIntegerType(bitsize), bitsize));
      instancemodule->setAttr(
          "WRITE_WIDTH_A",
          builder.getIntegerAttr(builder.getIntegerType(bitsize), bitsize));

      read_write_op.replaceAllUsesWith(instancemodule.getResult(0));
      read_write_op->erase();
      op->erase();
    } else
      assert(false && "firmemop.usesop is not a seq::FirMemReadWriteOp");
  }
}

/// This is the main entrypoint for the Core Memory Mapping conversion pass.
void CoreMemoryMappingPass::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();
  // auto arrayCreates = analyseHwArray(module);
  llvm::SmallVector<seq::FirMemOp> firMemOps = analyseFirMem(module);
  OpBuilder builder(&context);
  int module_num = 0;
  llvm::StringMap<hw::HWModuleExternOp> externmodule_map;
  llvm::DenseMap<int, hw::HWModuleExternOp> dram_map;
  for (auto op : firMemOps) {
    builder.setInsertionPoint(op);
    auto bitsize = op.getType().getWidth();
    auto depth = op.getType().getDepth();
    if (op.getReadLatency() == 1) {
      setBram(module, op, context, module_num, builder, externmodule_map);
    } else {
      setDRam(module, op, context, module_num, dram_map);
    }
  }
}
