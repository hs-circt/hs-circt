#include "circt/Conversion/CoreMemoryMapping.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include <cmath>

namespace circt {
#define GEN_PASS_DEF_COREMEMORYMAPPING
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;

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
  module.walk([&](hw::ArrayCreateOp op) { arrayCreates.emplace_back(op); });
  module.walk([&](hw::ArrayConcatOp op) { arrayCreates.emplace_back(op); });
  module.walk([&](hw::ArraySliceOp op) { arrayCreates.emplace_back(op); });
  return arrayCreates;
}

llvm::SmallVector<seq::FirMemOp> analyseFirMem(ModuleOp module) {
  llvm::SmallVector<seq::FirMemOp> firMemOps;
  module.walk([&](seq::FirMemOp op) { // cascade = true
    firMemOps.emplace_back(op);
  });
  return firMemOps;
}

hw::HWModuleExternOp create32Dram6(ModuleOp module, Operation *op,
                                   MLIRContext &context) {
  OpBuilder builder(&context);
  builder.setInsertionPointToStart(module.getBody());
  SmallVector<hw::PortInfo> ports;

  StringAttr moduleName = builder.getStringAttr("RAM32M");

  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DIA"),
                                  builder.getIntegerType(2),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DIB"),
                                  builder.getIntegerType(2),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DIC"),
                                  builder.getIntegerType(2),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DID"),
                                  builder.getIntegerType(2),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("ADDRA"),
                                  builder.getIntegerType(5),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("ADDRB"),
                                  builder.getIntegerType(5),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("ADDRC"),
                                  builder.getIntegerType(5),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("ADDRD"),
                                  builder.getIntegerType(5),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("WE"),
                                  builder.getIntegerType(1),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("WCLK"),
                                  seq::ClockType::get(&context),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DOA"),
                                  builder.getIntegerType(2),
                                  hw::ModulePort::Direction::Output});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DOB"),
                                  builder.getIntegerType(2),
                                  hw::ModulePort::Direction::Output});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DOC"),
                                  builder.getIntegerType(2),
                                  hw::ModulePort::Direction::Output});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DOD"),
                                  builder.getIntegerType(2),
                                  hw::ModulePort::Direction::Output});

  auto externmodule =
      builder.create<hw::HWModuleExternOp>(op->getLoc(), moduleName, ports);
  return externmodule;
}

hw::HWModuleExternOp create64Dram3(ModuleOp module, Operation *op,
                                   MLIRContext &context) {
  OpBuilder builder(&context);
  builder.setInsertionPointToStart(module.getBody());
  SmallVector<hw::PortInfo> ports;

  StringAttr moduleName = builder.getStringAttr("RAM64M");

  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DIA"),
                                  builder.getIntegerType(1),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DIB"),
                                  builder.getIntegerType(1),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DIC"),
                                  builder.getIntegerType(1),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DID"),
                                  builder.getIntegerType(1),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("ADDRA"),
                                  builder.getIntegerType(6),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("ADDRB"),
                                  builder.getIntegerType(6),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("ADDRC"),
                                  builder.getIntegerType(6),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("ADDRD"),
                                  builder.getIntegerType(6),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("WE"),
                                  builder.getIntegerType(1),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("WCLK"),
                                  seq::ClockType::get(&context),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DOA"),
                                  builder.getIntegerType(1),
                                  hw::ModulePort::Direction::Output});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DOB"),
                                  builder.getIntegerType(1),
                                  hw::ModulePort::Direction::Output});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DOC"),
                                  builder.getIntegerType(1),
                                  hw::ModulePort::Direction::Output});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DOD"),
                                  builder.getIntegerType(1),
                                  hw::ModulePort::Direction::Output});

  auto externmodule =
      builder.create<hw::HWModuleExternOp>(op->getLoc(), moduleName, ports);

  return externmodule;
}

hw::HWModuleExternOp create32Dram(ModuleOp module, Operation *op,
                                  MLIRContext &context, int width) {
  OpBuilder builder(&context);
  builder.setInsertionPointToStart(module.getBody());
  SmallVector<hw::PortInfo> ports;
  StringAttr moduleName = builder.getStringAttr("RAM" + std::to_string(width) + "X1D");

  int logwidth = std::log2(width);

  ports.emplace_back(hw::PortInfo{builder.getStringAttr("WCLK"),
                                  seq::ClockType::get(&context),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("D"),
                                  builder.getIntegerType(1),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("WE"),
                                  builder.getIntegerType(1),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("A"),
                                  builder.getIntegerType(logwidth),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DPRA"),
                                  builder.getIntegerType(logwidth),
                                  hw::ModulePort::Direction::Input});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DPO"),
                                  builder.getIntegerType(1),
                                  hw::ModulePort::Direction::Output});
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("SPO"),
                                  builder.getIntegerType(1),
                                  hw::ModulePort::Direction::Output});

  auto externmodule =
      builder.create<hw::HWModuleExternOp>(op->getLoc(), moduleName, ports);

  return externmodule;
}


hw::HWModuleExternOp createModuleExtern(ModuleOp op, MLIRContext &context,
                                        int moduleNum, int width) {
  OpBuilder builder(&context);
  builder.setInsertionPointToStart(op.getBody());
  StringAttr moduleName = builder.getStringAttr("RAMB36E2");
  SmallVector<hw::PortInfo> ports;
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("CLKARDCLK"),
                                  seq::ClockType::get(&context),
                                  hw::ModulePort::Direction::Input});

  // Port A configuration
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("ADDRARDADDR"),
                                  builder.getIntegerType(15),
                                  hw::ModulePort::Direction::Input});

  ports.emplace_back(hw::PortInfo{builder.getStringAttr("ENARDEN"),
                                  builder.getIntegerType(1),
                                  hw::ModulePort::Direction::Input});

  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DINADIN"),
                                  builder.getIntegerType(32),
                                  hw::ModulePort::Direction::Input});

  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DINPADINP"),
                                  builder.getIntegerType(4),
                                  hw::ModulePort::Direction::Input});

  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DOUTADOUT"),
                                  builder.getIntegerType(32),
                                  hw::ModulePort::Direction::Output});

  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DOUTPADOUTP"),
                                  builder.getIntegerType(4),
                                  hw::ModulePort::Direction::Output});
  // Port B configuration
  ports.emplace_back(hw::PortInfo{builder.getStringAttr("ADDRBWRADDR"),
                                  builder.getIntegerType(15),
                                  hw::ModulePort::Direction::Input});

  ports.emplace_back(hw::PortInfo{builder.getStringAttr("ENBWREN"),
                                  builder.getIntegerType(1),
                                  hw::ModulePort::Direction::Input});

  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DINBDIN"),
                                  builder.getIntegerType(32),
                                  hw::ModulePort::Direction::Input});

  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DINPBDINP"),
                                  builder.getIntegerType(4),
                                  hw::ModulePort::Direction::Input});

  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DOUTBDOUT"),
                                  builder.getIntegerType(32),
                                  hw::ModulePort::Direction::Output});

  ports.emplace_back(hw::PortInfo{builder.getStringAttr("DOUTPBDOUTP"),
                                  builder.getIntegerType(4),
                                  hw::ModulePort::Direction::Output});

  ports.emplace_back(hw::PortInfo{builder.getStringAttr("RSTRAMARSTRAM"),
                                  builder.getIntegerType(1),
                                  hw::ModulePort::Direction::Input});

  ports.emplace_back(hw::PortInfo{builder.getStringAttr("RSTREGARSTREG"),
                                  builder.getIntegerType(1),
                                  hw::ModulePort::Direction::Input});

  ports.emplace_back(hw::PortInfo{builder.getStringAttr("REGCEAREGCE"),
                                  builder.getIntegerType(1),
                                  hw::ModulePort::Direction::Input});

  ports.emplace_back(hw::PortInfo{builder.getStringAttr("WEA"),
                                  builder.getIntegerType(4),
                                  hw::ModulePort::Direction::Input});

  ports.emplace_back(hw::PortInfo{builder.getStringAttr("WEBWE"),
                                  builder.getIntegerType(8),
                                  hw::ModulePort::Direction::Input});

  auto externmodule =
      builder.create<hw::HWModuleExternOp>(op->getLoc(), moduleName, ports);
  return externmodule;
}

hw::InstanceOp
createInsetance(ModuleOp module, seq::FirMemOp op, seq::FirMemReadOp readop,
                MLIRContext &context, int &moduleNum,
                llvm::SmallVector<Value> &symbolInputs,
                llvm::DenseMap<int, hw::HWModuleExternOp> &dramMap, int bitSize,
                int logDepth, OpBuilder &builder) {
  symbolInputs.emplace_back(readop.getAddress());
  auto externop = dramMap[logDepth];
  if (externop == nullptr) {
    externop = create32Dram(module, op, context, 1 << logDepth);
    externop.setVerilogName(
        builder.getStringAttr("RAM" + std::to_string(1 << logDepth) + "X1D"));
    dramMap[logDepth] = externop;
  }
  if (symbolInputs[4].getType().getIntOrFloatBitWidth() < logDepth) {
    auto constType =
        logDepth - symbolInputs[4].getType().getIntOrFloatBitWidth();
    auto constant32 = builder.createOrFold<hw::ConstantOp>(
        op->getLoc(), builder.getIntegerType(constType),
        builder.getIntegerAttr(builder.getIntegerType(constType), 0));
    auto concatop = builder.create<comb::ConcatOp>(op->getLoc(), constant32,
                                                   symbolInputs[4]);
    symbolInputs[4] = concatop.getResult();
  }
  if (symbolInputs[3].getType().getIntOrFloatBitWidth() < logDepth) {
    auto constType =
        logDepth - symbolInputs[3].getType().getIntOrFloatBitWidth();
    auto constant32 = builder.createOrFold<hw::ConstantOp>(
        op->getLoc(), builder.getIntegerType(constType),
        builder.getIntegerAttr(builder.getIntegerType(constType), 0));
    auto concatop = builder.create<comb::ConcatOp>(op->getLoc(), constant32,
                                                   symbolInputs[3]);
    symbolInputs[3] = concatop.getResult();
  }
  auto instanceModule = builder.create<hw::InstanceOp>(
      op.getLoc(), externop,
      builder.getStringAttr("RAM" + std::to_string(1 << logDepth) + "X1D_" +
                            std::to_string(moduleNum++)),
      symbolInputs);
  return instanceModule;
}

void useX1D(ModuleOp module, seq::FirMemOp op, MLIRContext &context,
            int &moduleNum, llvm::SmallVector<Value> &symbolInputs,
            llvm::DenseMap<int, hw::HWModuleExternOp> &dramMap,
            OpBuilder &builder, int bitSize, int i, int depth,
            seq::FirMemReadOp readop, seq::FirMemWriteOp writeop,
            llvm::SmallVector<hw::InstanceOp> &dramInstances,
            llvm::SmallVector<Value> &dramResult) {
  auto extractop =
      builder.create<comb::ExtractOp>(op->getLoc(), writeop.getData(), i, 1);
  symbolInputs.emplace_back(writeop.getClk());
  symbolInputs.emplace_back(extractop.getResult());
  symbolInputs.emplace_back(writeop.getEnable());
  symbolInputs.emplace_back(writeop.getAddress());

  hw::InstanceOp instanceModule =
      createInsetance(module, op, readop, context, moduleNum, symbolInputs,
                      dramMap, bitSize, log2(depth), builder);
  instanceModule->setAttr(
      "INIT", builder.getIntegerAttr(builder.getIntegerType(log2(depth)), 0));
  instanceModule->setAttr("IS_WCLK_INVERTED",
                          builder.getIntegerAttr(builder.getIntegerType(1), 0));
  dramInstances.emplace_back(instanceModule);
  dramResult.emplace_back(instanceModule.getResult(0));
}

void use6bitRAM(ModuleOp module, seq::FirMemOp op, MLIRContext &context,
                int &moduleNum, llvm::SmallVector<Value> &symbolInputs,
                llvm::DenseMap<int, hw::HWModuleExternOp> &dramMap,
                OpBuilder &builder, int bitSize, int i, int depth,
                seq::FirMemReadOp readop, seq::FirMemWriteOp writeop,
                llvm::SmallVector<hw::InstanceOp> &dramInstances,
                llvm::SmallVector<Value> &dramResult,
                seq::FirMemReadOp secReadop) {
  auto extractopA =
      builder.create<comb::ExtractOp>(op->getLoc(), writeop.getData(), i, 2);
  auto extractopB = builder.create<comb::ExtractOp>(
      op->getLoc(), writeop.getData(), i + 2, 2);
  auto extractopC = builder.create<comb::ExtractOp>(
      op->getLoc(), writeop.getData(), i + 4, 2);
  symbolInputs.emplace_back(extractopA);
  symbolInputs.emplace_back(extractopB);
  symbolInputs.emplace_back(extractopC);
  auto constant = builder.createOrFold<hw::ConstantOp>(
      op->getLoc(), builder.getIntegerType(2),
      builder.getIntegerAttr(builder.getIntegerType(2), 0));
  symbolInputs.emplace_back(constant); // empty

  mlir::Value addra = readop.getAddress();
  int logDepth = log2(depth < 32 ? 32 : depth);

  if (addra.getType().getIntOrFloatBitWidth() < logDepth) {
    auto constType = logDepth - addra.getType().getIntOrFloatBitWidth();
    auto constant32 = builder.createOrFold<hw::ConstantOp>(
        op->getLoc(), builder.getIntegerType(constType),
        builder.getIntegerAttr(builder.getIntegerType(constType), 0));
    auto concatop =
        builder.create<comb::ConcatOp>(op->getLoc(), constant32, addra);
    addra = concatop.getResult();
  }
  mlir::Value writeAddr = writeop.getAddress();
  if (writeAddr.getType().getIntOrFloatBitWidth() < logDepth) {
    auto constType = logDepth - writeAddr.getType().getIntOrFloatBitWidth();
    auto constant32 = builder.createOrFold<hw::ConstantOp>(
        op->getLoc(), builder.getIntegerType(constType),
        builder.getIntegerAttr(builder.getIntegerType(constType), 0));
    auto concatop =
        builder.create<comb::ConcatOp>(op->getLoc(), constant32, writeAddr);
    writeAddr = concatop.getResult();
  }

  symbolInputs.emplace_back(addra);               // addra
  symbolInputs.emplace_back(addra);               // addrb
  symbolInputs.emplace_back(addra);               // addrc
  symbolInputs.emplace_back(writeAddr);           // addrd
  symbolInputs.emplace_back(writeop.getEnable()); // ena
  symbolInputs.emplace_back(writeop.getClk());    // clka
  auto externop = dramMap[logDepth * 10]; // avoid 6bitRAM and 1bit mixed
  if (externop == nullptr) {
    externop = create32Dram6(module, op, context);
    dramMap[logDepth * 10] = externop;
  }
  auto instanceModule = builder.create<hw::InstanceOp>(
      op.getLoc(), externop,
      builder.getStringAttr("RAM32M_" + std::to_string(moduleNum++)),
      symbolInputs);
  instanceModule->setAttr(
      "INIT_A", builder.getIntegerAttr(builder.getIntegerType(64), 0));
  instanceModule->setAttr(
      "INIT_B", builder.getIntegerAttr(builder.getIntegerType(64), 0));
  instanceModule->setAttr(
      "INIT_C", builder.getIntegerAttr(builder.getIntegerType(64), 0));
  instanceModule->setAttr(
      "INIT_D", builder.getIntegerAttr(builder.getIntegerType(64), 0));
  instanceModule->setAttr("IS_WCLK_INVERTED",
                          builder.getIntegerAttr(builder.getIntegerType(1), 0));
  dramInstances.emplace_back(instanceModule);
  dramResult.emplace_back(instanceModule.getResult(0));
  dramResult.emplace_back(instanceModule.getResult(1));
  dramResult.emplace_back(instanceModule.getResult(2));
} // DOA-DOC output, DOD empty

void use3BitRAM(ModuleOp module, seq::FirMemOp op, MLIRContext &context,
                int &moduleNum, llvm::SmallVector<Value> &symbolInputs,
                llvm::DenseMap<int, hw::HWModuleExternOp> &dramMap,
                OpBuilder &builder, int bitSize, int i, int depth,
                seq::FirMemReadOp readop, seq::FirMemWriteOp writeop,
                llvm::SmallVector<hw::InstanceOp> &dramInstances,
                llvm::SmallVector<Value> &dramResult,
                seq::FirMemReadOp secReadop) {
  auto extractopA =
      builder.create<comb::ExtractOp>(op->getLoc(), writeop.getData(), i, 1);
  auto extractopB = builder.create<comb::ExtractOp>(
      op->getLoc(), writeop.getData(), i + 1, 1);
  auto extractopC = builder.create<comb::ExtractOp>(
      op->getLoc(), writeop.getData(), i + 2, 1);
  symbolInputs.emplace_back(extractopA);
  symbolInputs.emplace_back(extractopB);
  symbolInputs.emplace_back(extractopC);
  auto constant = builder.create<hw::ConstantOp>(
      op->getLoc(), builder.getIntegerType(1),
      builder.getIntegerAttr(builder.getIntegerType(1), 0));
  symbolInputs.emplace_back(constant.getResult()); // empty

  mlir::Value addra = readop.getAddress();
  int logDepth = log2(depth < 64 ? 64 : depth);

  if (addra.getType().getIntOrFloatBitWidth() <
      logDepth) { // read address length and port length alignment
    auto constType = logDepth - addra.getType().getIntOrFloatBitWidth();
    auto constant32 = builder.create<hw::ConstantOp>(
        op->getLoc(), builder.getIntegerType(constType),
        builder.getIntegerAttr(builder.getIntegerType(constType), 0));
    auto concatop = builder.create<comb::ConcatOp>(
        op->getLoc(), constant32.getResult(), addra);
    addra = concatop.getResult();
  }
  mlir::Value writeAddr = writeop.getAddress();
  if (writeAddr.getType().getIntOrFloatBitWidth() <
      logDepth) { // write address length and port length alignment
    auto constType = logDepth - writeAddr.getType().getIntOrFloatBitWidth();
    auto constant32 = builder.create<hw::ConstantOp>(
        op->getLoc(), builder.getIntegerType(constType),
        builder.getIntegerAttr(builder.getIntegerType(constType), 0));
    auto concatop = builder.create<comb::ConcatOp>(
        op->getLoc(), constant32.getResult(), writeAddr);
    writeAddr = concatop.getResult();
  }

  symbolInputs.emplace_back(addra);               // addra
  symbolInputs.emplace_back(addra);               // addrb
  symbolInputs.emplace_back(addra);               // addrc
  symbolInputs.emplace_back(writeAddr);           // addrd
  symbolInputs.emplace_back(writeop.getEnable()); // ena
  symbolInputs.emplace_back(writeop.getClk());    // clka
  auto externop = dramMap[logDepth * 10]; // avoid 6bitRAM and 1bit mixed
  if (externop == nullptr) {
    externop = create64Dram3(module, op, context);
    dramMap[logDepth * 10] = externop;
  }
  auto instanceModule = builder.create<hw::InstanceOp>(
      op.getLoc(), externop,
      builder.getStringAttr("RAM64M_" + std::to_string(moduleNum++)),
      symbolInputs);
  instanceModule->setAttr(
      "INIT_A", builder.getIntegerAttr(builder.getIntegerType(64), 0));
  instanceModule->setAttr(
      "INIT_B", builder.getIntegerAttr(builder.getIntegerType(64), 0));
  instanceModule->setAttr(
      "INIT_C", builder.getIntegerAttr(builder.getIntegerType(64), 0));
  instanceModule->setAttr(
      "INIT_D", builder.getIntegerAttr(builder.getIntegerType(64), 0));
  instanceModule->setAttr("IS_WCLK_INVERTED",
                          builder.getIntegerAttr(builder.getIntegerType(1), 0));
  dramInstances.emplace_back(instanceModule);
  dramResult.emplace_back(instanceModule.getResult(0));
  dramResult.emplace_back(instanceModule.getResult(1));
  dramResult.emplace_back(instanceModule.getResult(2));
} // DOA-DOC output, DOD empty

void setDistRam(ModuleOp module, seq::FirMemOp op, MLIRContext &context,
                int &moduleNum,
                llvm::DenseMap<int, hw::HWModuleExternOp> &dramMap) {
  OpBuilder builder(&context);
  int bitSize = op.getType().getWidth();
  int depth = op.getType().getDepth();
  int depthArray[] = {32, 64, 128, 256}, depthIndex = 0;
  while (depthArray[depthIndex] < depth)
    depthIndex++;
  depth = depthArray[depthIndex];

  llvm::SmallVector<hw::InstanceOp> dramInstances;
  seq::FirMemReadOp readop = nullptr, secReadop = nullptr;
  seq::FirMemWriteOp writeop;
  for (auto &useop : op->getUses()) {
    if (auto readop1 = llvm::dyn_cast<seq::FirMemReadOp>(useop.getOwner())) {
      if (readop != nullptr) {
        secReadop = readop;
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

  llvm::SmallVector<Value> dramResult;

  for (int i = 0; i < bitSize; i++) {
    llvm::SmallVector<Value> symbolInputs;
    if ((bitSize <= 12 || bitSize - i < 6) || depth > 64) {
      useX1D(module, op, context, moduleNum, symbolInputs, dramMap, builder,
             bitSize, i, depth, readop, writeop, dramInstances, dramResult);
    } else if (depth <= 32) {
      use6bitRAM(module, op, context, moduleNum, symbolInputs, dramMap, builder,
                 bitSize, i, depth, readop, writeop, dramInstances, dramResult,
                 secReadop);
      i += 5;
    } else if (depth <= 64) {
      use3BitRAM(module, op, context, moduleNum, symbolInputs, dramMap, builder,
                 bitSize, i, depth, readop, writeop, dramInstances, dramResult,
                 secReadop);
      i += 2;
    }
  }

  auto resultop = builder.create<comb::ConcatOp>(op->getLoc(), dramResult);
  readop->replaceAllUsesWith(resultop);
  readop->erase();
  writeop->erase();
  op->erase();
}

std::pair<hw::InstanceOp, seq::FirMemReadWriteOp>
setBram(ModuleOp module, seq::FirMemOp op, MLIRContext &context, int &moduleNum,
        OpBuilder &builder,
        llvm::StringMap<hw::HWModuleExternOp> &moduleGeneratedMap, int portsize,
        int ExtractSize) {
  auto bitSize = op.getType().getWidth();
  llvm::SmallVector<Value> symbolInputs;
  hw::InstanceOp instanceop = nullptr;
  if (auto readWriteOp = llvm::dyn_cast<seq::FirMemReadWriteOp>(
          op->getUses().begin()->getOwner())) {
    if (portsize == 72 && readWriteOp.getMask() != nullptr)
      portsize = 64;

    if ((bitSize > 32 || (bitSize > 16 && bitSize <= 18) || bitSize == 9) &&
        readWriteOp.getMask() == nullptr) {
    }

    symbolInputs.emplace_back(readWriteOp.getClk()); // clka

    mlir::Value addra = readWriteOp.getAddress();
    if (addra.getType().getIntOrFloatBitWidth() < 15) {
      auto constantop = builder.createOrFold<hw::ConstantOp>(
          op->getLoc(),
          builder.getIntegerType(15 - addra.getType().getIntOrFloatBitWidth()),
          builder.getIntegerAttr(
              builder.getIntegerType(15 -
                                     addra.getType().getIntOrFloatBitWidth()),
              0));
      auto concatop =
          builder.create<comb::ConcatOp>(op->getLoc(), constantop, addra);
      addra = concatop.getResult();
    }
    symbolInputs.emplace_back(addra);                   // addra
    symbolInputs.emplace_back(readWriteOp.getEnable()); // ena

    mlir::Value writeData = readWriteOp.getWriteData();
    mlir::Value extractValue = writeData;
    if (ExtractSize >= 0) {
      auto extrLength = std::min(
          int(writeData.getType().getIntOrFloatBitWidth() - ExtractSize),
          portsize);
      extractValue = builder.create<comb::ExtractOp>(op->getLoc(), writeData,
                                                     ExtractSize, extrLength);
    }
    if (extractValue.getType().getIntOrFloatBitWidth() < 32) {
      auto constantop = builder.createOrFold<hw::ConstantOp>(
          op->getLoc(),
          builder.getIntegerType(
              32 - extractValue.getType().getIntOrFloatBitWidth()),
          builder.getIntegerAttr(
              builder.getIntegerType(
                  32 - extractValue.getType().getIntOrFloatBitWidth()),
              0));
      auto concatop = builder.create<comb::ConcatOp>(op->getLoc(), constantop,
                                                     extractValue);

      writeData = concatop.getResult();
      symbolInputs.emplace_back(concatop.getResult()); // diadi
    } else if (extractValue.getType().getIntOrFloatBitWidth() > 32) {
      auto extractop =
          builder.create<comb::ExtractOp>(op->getLoc(), extractValue, 0, 32);
      symbolInputs.emplace_back(extractop.getResult()); // diadi
    } else {
      symbolInputs.emplace_back(extractValue); // diadi
    }
    auto nullop1 = builder.createOrFold<hw::ConstantOp>(
        op->getLoc(), builder.getIntegerType(4),
        builder.getIntegerAttr(builder.getIntegerType(4), 0));
    mlir::Value dipadi;
    if ((bitSize > 32 || (bitSize > 16 && bitSize <= 18) || bitSize == 9) &&
        readWriteOp.getMask() == nullptr) {
      dipadi = builder.create<comb::ExtractOp>(
          op->getLoc(), writeData, bitSize - bitSize % 8,
          std::min(bitSize / 8,
                   extractValue.getType().getIntOrFloatBitWidth() - 32));
      if (dipadi.getType().getIntOrFloatBitWidth() != 4) {
        auto constantop = builder.createOrFold<hw::ConstantOp>(
            op->getLoc(),
            builder.getIntegerType(4 -
                                   dipadi.getType().getIntOrFloatBitWidth()),
            builder.getIntegerAttr(
                builder.getIntegerType(
                    4 - dipadi.getType().getIntOrFloatBitWidth()),
                0));
        auto concatop =
            builder.create<comb::ConcatOp>(op->getLoc(), constantop, dipadi);
        dipadi = concatop.getResult();
      }
    } else
      dipadi = nullop1;
    symbolInputs.emplace_back(dipadi); // dinpadi
    auto nullop = builder.createOrFold<hw::ConstantOp>(
        op->getLoc(), builder.getIntegerType(15),
        builder.getIntegerAttr(builder.getIntegerType(15), 0));
    auto nullop2 = builder.createOrFold<hw::ConstantOp>(
        op->getLoc(), builder.getIntegerType(32),
        builder.getIntegerAttr(builder.getIntegerType(32), 0));
    symbolInputs.emplace_back(nullop); // addrb
    auto nullopi1 = builder.createOrFold<hw::ConstantOp>(
        op->getLoc(), builder.getIntegerType(1),
        builder.getIntegerAttr(builder.getIntegerType(1), 0));
    symbolInputs.emplace_back(nullopi1); // enb
    symbolInputs.emplace_back(nullop2);  // dinbdin
    symbolInputs.emplace_back(nullop1);  // dinpbdin
    auto constOne = builder.createOrFold<hw::ConstantOp>(
        op.getLoc(),
        builder.getIntegerType(1),                             // (i1)
        builder.getIntegerAttr(builder.getIntegerType(1), 1)); // constOne

    symbolInputs.emplace_back(nullopi1); // RSTRAMARSTRAM
    symbolInputs.emplace_back(nullopi1); // RSTREGARSTREG
    symbolInputs.emplace_back(constOne); // REGCEAREGCE
    mlir::Value weaValue = readWriteOp.getOperand(5);
    SmallVector<Value> values = {weaValue, weaValue, weaValue, weaValue};
    auto concatwea = builder.create<comb::ConcatOp>(op->getLoc(), values);
    symbolInputs.emplace_back(concatwea.getResult()); // wea

    auto mask = builder.createOrFold<hw::ConstantOp>(
        op.getLoc(),
        builder.getIntegerType(8), // (i1)
        builder.getIntegerAttr(builder.getIntegerType(8), 0));
    if (ExtractSize != -1 && readWriteOp.getMask() != nullptr) {
      mask = builder.createOrFold<comb::ExtractOp>(
          op->getLoc(), readWriteOp.getMask(), ExtractSize / 8, portsize / 8);
      if (mask.getType().getIntOrFloatBitWidth() < 8) {
        auto constantop = builder.createOrFold<hw::ConstantOp>(
            op->getLoc(),
            builder.getIntegerType(8 - mask.getType().getIntOrFloatBitWidth()),
            builder.getIntegerAttr(
                builder.getIntegerType(8 -
                                       mask.getType().getIntOrFloatBitWidth()),
                0));
        mask = builder.create<comb::ConcatOp>(op->getLoc(), constantop, mask);
      }
    }
    symbolInputs.emplace_back(mask); // WEBWE

    std::string key = "RAMB36E2";

    auto it = moduleGeneratedMap.find(key);
    if (it == moduleGeneratedMap.end()) {
      auto newModule =
          createModuleExtern(module, context, moduleNum++, portsize);
      moduleGeneratedMap.insert({key, newModule});
      it = moduleGeneratedMap.find(key);
    }
    auto &externModule = it->second;
    instanceop = builder.create<hw::InstanceOp>(
        op.getLoc(), externModule,
        builder.getStringAttr("RAMB36E2_" + std::to_string(moduleNum++)),
        symbolInputs);

    return {instanceop, readWriteOp};
  } else
    assert(false && "not a readWriteOp");
  return {nullptr, nullptr};
}

void CoreMemoryMappingPass::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();
  // auto arrayCreates = analyseHwArray(module);
  llvm::SmallVector<seq::FirMemOp> firMemOps = analyseFirMem(module);
  OpBuilder builder(&context);
  int moduleNum = 0;
  llvm::StringMap<hw::HWModuleExternOp> moduleGeneratedMap;
  llvm::DenseMap<int, hw::HWModuleExternOp> dramMap;
  for (auto op : firMemOps) {
    builder.setInsertionPoint(op);
    int bitSize = op.getType().getWidth();
    int depth = op.getType().getDepth();
    if (op.getReadLatency() == 1) {
      int maxBitsize = bitSize;
      int bramSize = 0;
      std::vector<std::pair<Operation *, int>> instUses;
      std::vector<mlir::Value> instValue;
      if (bitSize * depth > 36864 ||
          bitSize > 36) { // Need to split width for storage
        maxBitsize = 36864.0 / depth;
        int totalSize = bitSize;
        if (bitSize == 64 &&
            op.getMemory().getType().getMaskWidth() != std::nullopt)
          maxBitsize = 32;
        bramSize = std::ceil(bitSize * 1.0 / maxBitsize * 1.0);

        for (int i = 0; i < bramSize; i++) {
          hw::InstanceOp instanceop;
          seq::FirMemReadWriteOp readWriteOp;
          std::tie(instanceop, readWriteOp) =
              setBram(module, op, context, moduleNum, builder,
                      moduleGeneratedMap, maxBitsize, i * maxBitsize);
          if (instUses.empty()) {
            for (auto &uses : readWriteOp->getUses()) {
              instUses.emplace_back(uses.getOwner(), uses.getOperandNumber());
            }
          }
          mlir::Value instV = instanceop.getResult(0);
          instV = builder.create<comb::ExtractOp>(
              op->getLoc(), instanceop.getResult(0), 0,
              std::min(maxBitsize, totalSize));
          instValue.emplace_back(instV);
          if (op.getMemory().getType().getMaskWidth() == std::nullopt &&
              (maxBitsize == 9 || (maxBitsize > 16 && maxBitsize <= 18) ||
               maxBitsize >= 32)) { // Mask exists at this time
            int len = 0;
            if (maxBitsize == 9)
              len = 1;
            else if (maxBitsize > 16 && maxBitsize <= 18)
              len = maxBitsize - 16;
            else if (maxBitsize >= 32)
              len = maxBitsize - 32;
            instV = builder.create<comb::ExtractOp>(
                op->getLoc(), instanceop.getResult(1), 0, len);
            instValue.emplace_back(instV);
          }
          totalSize -= maxBitsize;
        }
        if (!instUses.empty()) {
          auto concatop =
              builder.create<comb::ConcatOp>(op->getLoc(), instValue);
          for (auto &useop : instUses) {
            useop.first->setOperand(useop.second, concatop.getResult());
          }
        }
      } else { // Can be implemented directly in a single memory block
        auto portsize = bitSize;
        bool hasdoutpadout = false;
        if (bitSize > 32 || (bitSize > 16 && bitSize <= 18) || bitSize == 9)
          hasdoutpadout = true;
        std::pair<hw::InstanceOp, seq::FirMemReadWriteOp> instanceOp;
        instanceOp = setBram(module, op, context, moduleNum, builder,
                             moduleGeneratedMap, portsize, -1);
        auto instV = instanceOp.first.getResult(0);
        if (bitSize < 32) {
          if (hasdoutpadout) {
            int len = 0;
            if (bitSize == 9)
              len = 8;
            else if (bitSize > 16 && bitSize <= 18)
              len = 16;
            else if (bitSize > 32)
              len = 32;
            instV = builder.create<comb::ExtractOp>(
                op->getLoc(), instanceOp.first.getResult(0), 0, len);
          } else {
            instV = builder.create<comb::ExtractOp>(
                op->getLoc(), instanceOp.first.getResult(0), 0, bitSize);
          }
        }
        if (op.getMemory().getType().getMaskWidth() == std::nullopt &&
            hasdoutpadout) {
          int len = 0;
          if (bitSize == 9)
            len = 1;
          else if (bitSize > 16 && bitSize <= 18)
            len = bitSize - 16;
          else if (bitSize > 32)
            len = bitSize - 32;
          mlir::Value doutpadout = builder.create<comb::ExtractOp>(
              op->getLoc(), instanceOp.first.getResult(1), 0, len);
          instV =
              builder.create<comb::ConcatOp>(op->getLoc(), instV, doutpadout);
        }
        instanceOp.second.replaceAllUsesWith(instV);
      }
      for (auto &useop : op->getUses()) {
        useop.getOwner()->erase();
      }
      op->erase();
    } else {
      setDistRam(module, op, context, moduleNum, dramMap);
    }
  }
}
