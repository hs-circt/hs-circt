#include "circt/Conversion/CoreMemoryMapping.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/LLVM.h"
#include "mlir-c/IR.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <cmath>
#include <optional>

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

class CreateInstance {
public:
  CreateInstance(ModuleOp module, int bitSize, int depth) {
    this->module = module;
    this->depth = depth;
    this->bitSize = bitSize;
  }
  std::pair<hw::InstanceOp, seq::FirMemReadWriteOp>
  instanceBlockRam(seq::FirMemOp op, ConversionPatternRewriter &rewriter,
                   int portSize, int extractPlace,
                   hw::HWModuleExternOp &blockRam) {
    llvm::SmallVector<mlir::Value> symbolInputs;
    llvm::SmallVector<seq::FirMemReadWriteOp> firmemops;
    auto *context = op->getContext();
    int portIndex = 0;
    auto constant1 = rewriter.create<circt::hw::ConstantOp>(
        op->getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(true));
    auto constant0 = rewriter.create<circt::hw::ConstantOp>(
        op->getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false));
    auto constant04 = rewriter.create<hw::ConstantOp>(
        op->getLoc(), rewriter.getIntegerType(4),
        rewriter.getIntegerAttr(rewriter.getIntegerType(4), 0));
    auto constant08 = rewriter.create<hw::ConstantOp>(
        op->getLoc(), rewriter.getIntegerType(8),
        rewriter.getIntegerAttr(rewriter.getIntegerType(8), 0));
    seq::FirMemReadWriteOp tureport;
    for (auto &useop : op->getUses()) {
      if (auto readWriteOp =
              dyn_cast<seq::FirMemReadWriteOp>(useop.getOwner())) {
        firmemops.emplace_back(readWriteOp);
        if (portIndex == 0) {
          tureport = readWriteOp;
          symbolInputs.emplace_back(readWriteOp.getClk());
          mlir::Value addra = readWriteOp.getAddress();
          if (portSize > 1) {
            llvm::SmallVector<Value> constantVector;
            for (int i = 0; i < int(log2(portSize)); i++) {
              constantVector.emplace_back(constant1);
            }
            addra = rewriter.create<comb::ConcatOp>(op->getLoc(), addra,
                                                    constantVector);
            if (addra.getType().getIntOrFloatBitWidth() > 15) {
              assert(false && "address width is too large");
            }
          }
          if (addra.getType().getIntOrFloatBitWidth() < 15) {
            llvm::SmallVector<Value> constantVector;
            for (int i = 0;
                 i <
                 15 - static_cast<int>(addra.getType().getIntOrFloatBitWidth());
                 i++) {
              constantVector.emplace_back(constant0);
            }
            constantVector.emplace_back(addra);
            addra =
                rewriter.create<comb::ConcatOp>(op->getLoc(), constantVector);
          }
          symbolInputs.emplace_back(addra);
          symbolInputs.emplace_back(readWriteOp.getEnable());
          mlir::Value writeData = readWriteOp.getWriteData();
          mlir::Value extractValue = writeData;
          int extractSize = portSize;
          if (portSize > 32) {
            extractSize = 32;
          } else if (portSize > 16 && portSize <= 18) {
            extractSize = 16;
          } else if (portSize == 9) {
            extractSize = 8;
          }
          if (extractPlace != -1) {
            auto extrLength = std::min(
                int(writeData.getType().getIntOrFloatBitWidth()) - extractPlace,
                extractSize);
            extractValue = rewriter.create<comb::ExtractOp>(
                op->getLoc(), writeData, extractPlace, extrLength);
            writeData = extractValue;
          }
          if (extractValue.getType().getIntOrFloatBitWidth() < 32) {
            llvm::SmallVector<Value> constantVector;
            for (int i = 0;
                 i < 32 - static_cast<int>(
                              extractValue.getType().getIntOrFloatBitWidth());
                 i++) {
              constantVector.emplace_back(constant0);
            }
            constantVector.emplace_back(extractValue);
            writeData =
                rewriter.create<comb::ConcatOp>(op->getLoc(), constantVector);
          } else if (extractValue.getType().getIntOrFloatBitWidth() > 32) {
            writeData = rewriter.create<comb::ExtractOp>(op->getLoc(),
                                                         extractValue, 0, 32);
          }
          symbolInputs.emplace_back(writeData);
          mlir::Value dipadi = constant04;
          if ((portSize > 32 || (portSize > 16 && portSize <= 18) ||
               portSize == 9) &&
              readWriteOp.getMask() == nullptr &&
              extractPlace + portSize < bitSize) {
            dipadi = rewriter.create<comb::ExtractOp>(
                op->getLoc(), readWriteOp.getWriteData(),
                std::max(extractPlace, 0) + portSize - portSize % 8,
                int(portSize / 8));
            if (dipadi.getType().getIntOrFloatBitWidth() != 4) {
              llvm::SmallVector<Value> constantVector;
              for (int i = 0;
                   i < 4 - static_cast<int>(
                               dipadi.getType().getIntOrFloatBitWidth());
                   i++) {
                constantVector.emplace_back(constant0);
              }
              constantVector.emplace_back(dipadi);
              dipadi =
                  rewriter.create<comb::ConcatOp>(op->getLoc(), constantVector);
            }
          }
          symbolInputs.emplace_back(dipadi);
        } else if (portIndex == 1) {
          symbolInputs.emplace_back(readWriteOp.getClk()); // clkb

          mlir::Value addrb = readWriteOp.getAddress();
          if (portSize > 1) {
            llvm::SmallVector<Value> constantVector;
            for (int i = 0; i < int(log2(portSize)); i++) {
              constantVector.emplace_back(constant1);
            }
            addrb = rewriter.create<comb::ConcatOp>(op->getLoc(), addrb,
                                                    constantVector);
            if (addrb.getType().getIntOrFloatBitWidth() > 15) {
              assert(false && "address width is too large");
            }
          }
          if (addrb.getType().getIntOrFloatBitWidth() < 15) {
            llvm::SmallVector<Value> constantVector;
            for (int i = 0;
                 i <
                 15 - static_cast<int>(addrb.getType().getIntOrFloatBitWidth());
                 i++) {
              constantVector.emplace_back(constant0);
            }
            constantVector.emplace_back(addrb);
            addrb =
                rewriter.create<comb::ConcatOp>(op->getLoc(), constantVector);
          }
          symbolInputs.emplace_back(addrb);
          symbolInputs.emplace_back(readWriteOp.getEnable()); // enb
          mlir::Value writeData = readWriteOp.getWriteData();
          mlir::Value extractValue = writeData;
          if (extractPlace != -1) {
            auto extrLength = std::min(
                int(writeData.getType().getIntOrFloatBitWidth() - extractPlace),
                portSize);
            extractValue = rewriter.create<comb::ExtractOp>(
                op->getLoc(), writeData, extractPlace, extrLength);
            writeData = extractValue;
          }
          if (extractValue.getType().getIntOrFloatBitWidth() < 32) {
            llvm::SmallVector<Value> constantVector;
            for (int i = 0;
                 i < 32 - static_cast<int>(
                              extractValue.getType().getIntOrFloatBitWidth());
                 i++) {
              constantVector.emplace_back(constant0);
            }
            constantVector.emplace_back(extractValue);
            writeData =
                rewriter.create<comb::ConcatOp>(op->getLoc(), constantVector);

          } else if (extractValue.getType().getIntOrFloatBitWidth() > 32) {
            writeData = rewriter.create<comb::ExtractOp>(op->getLoc(),
                                                         extractValue, 0, 32);
          }

          symbolInputs.emplace_back(writeData);
          mlir::Value dipadi = constant04;
          if ((bitSize > 32 || (bitSize > 16 && bitSize <= 18) ||
               bitSize == 9) &&
              readWriteOp.getMask() == nullptr) {
            dipadi = rewriter.create<comb::ExtractOp>(
                op->getLoc(), writeData, bitSize - bitSize % 8,
                std::min(bitSize / 8, portSize));
          }
          symbolInputs.emplace_back(dipadi);
          symbolInputs.emplace_back(constant0); // RSTRAMARSTRAM
          symbolInputs.emplace_back(constant0); // RSTREGARSTREG
          symbolInputs.emplace_back(constant1); // REGCEAREGCE
        }
        portIndex++;
      }
    }
    if (portIndex == 1) {
      auto constant015 = rewriter.create<hw::ConstantOp>(
          op->getLoc(), rewriter.getIntegerType(15),
          rewriter.getIntegerAttr(rewriter.getIntegerType(15), 0));
      auto constant032 = rewriter.create<hw::ConstantOp>(
          op->getLoc(), rewriter.getIntegerType(32),
          rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0));
      auto constant0clk =
          rewriter.create<seq::ToClockOp>(op->getLoc(), constant0);
      symbolInputs.emplace_back(constant0clk); // clkbwrclk
      symbolInputs.emplace_back(constant015);  // addrb
      symbolInputs.emplace_back(constant0);    // enb
      symbolInputs.emplace_back(constant032);  // dinbdin
      symbolInputs.emplace_back(constant04);   // dinpbdin
      symbolInputs.emplace_back(constant0);    // RSTRAMARSTRAM
      symbolInputs.emplace_back(constant0);    // RSTREGARSTREG
      symbolInputs.emplace_back(constant1);    // REGCEAREGCE
    }
    mlir::Value weaValue = firmemops[0].getOperand(5);
    SmallVector<Value> values = {weaValue, weaValue, weaValue, weaValue};
    weaValue = rewriter.create<comb::ConcatOp>(op->getLoc(), values);

    symbolInputs.emplace_back(weaValue); // wea

    mlir::Value webValue = constant08;
    if (portSize == 2) {
      webValue = firmemops[1].getOperand(5);
      SmallVector<Value> values = {constant0, constant0, constant0, constant0,
                                   weaValue,  weaValue,  weaValue,  weaValue};
      webValue = rewriter.create<comb::ConcatOp>(op->getLoc(), values);
    } else if (extractPlace != -1 && firmemops[0].getMask() != nullptr) {
      weaValue = rewriter.createOrFold<comb::ExtractOp>(
          op->getLoc(), firmemops[0].getMask(), extractPlace / 8, portSize / 8);
      if (weaValue.getType().getIntOrFloatBitWidth() < 4) {
        llvm::SmallVector<Value> constantVector;
        for (int i = 0; i < 4 - static_cast<int>(
                                    weaValue.getType().getIntOrFloatBitWidth());
             i++) {
          constantVector.emplace_back(constant0);
        }
        constantVector.emplace_back(weaValue);
        weaValue =
            rewriter.create<comb::ConcatOp>(op->getLoc(), constantVector);
      }
      symbolInputs.back() = weaValue;
    }
    symbolInputs.emplace_back(webValue);
    if (blockRam == nullptr) {
      blockRam = createModuleExtern(module, *context, moduleNum++, portSize);
    }
    auto &externModule = blockRam;
    auto instanceop = rewriter.create<hw::InstanceOp>(
        op.getLoc(), externModule,
        rewriter.getStringAttr("RAMB36E2_" + std::to_string(moduleNum++)),
        symbolInputs);
    {
      SmallVector<Attribute> params;
      params.emplace_back(hw::ParamDeclAttr::get(
          rewriter.getContext(), rewriter.getStringAttr("READ_WIDTH_A"),
          rewriter.getIntegerType(32),
          rewriter.getIntegerAttr(rewriter.getIntegerType(32), portSize)));
      params.emplace_back(hw::ParamDeclAttr::get(
          rewriter.getContext(), rewriter.getStringAttr("WRITE_WIDTH_A"),
          rewriter.getIntegerType(32),
          rewriter.getIntegerAttr(rewriter.getIntegerType(32), portSize)));
      params.emplace_back(hw::ParamDeclAttr::get(
          rewriter.getContext(), rewriter.getStringAttr("DOA_REG"),
          rewriter.getIntegerType(1),
          rewriter.getIntegerAttr(rewriter.getIntegerType(1), 0)));
      params.emplace_back(hw::ParamDeclAttr::get(
          rewriter.getContext(), rewriter.getStringAttr("INIT_A"),
          rewriter.getIntegerType(36),
          rewriter.getIntegerAttr(rewriter.getIntegerType(36), 0)));
      params.emplace_back(hw::ParamDeclAttr::get(
          rewriter.getContext(), rewriter.getStringAttr("WRITE_MODE_A"),
          rewriter.getStringAttr("WRITE_FIRST").getType(),
          rewriter.getStringAttr("WRITE_FIRST")));
      instanceop->setAttr("parameters",
                          ArrayAttr::get(rewriter.getContext(), params));
    }
    return {instanceop, firmemops[0]};
  }

  void instanceDistRam(int bitSize, int &place, seq::FirMemReadOp readOp,
                       seq::FirMemWriteOp writeOp, seq::FirMemOp op,
                       ConversionPatternRewriter &rewriter,
                       llvm::DenseMap<int, hw::HWModuleExternOp> &dramMap) {
    this->dramMap = dramMap;
    this->locOp = op;
    if ((bitSize <= 12 || bitSize - place < 6) || depth > 64) {
      hw::InstanceOp inst = useX1D(op, rewriter, readOp, writeOp, place);
      this->locOp = inst;
      dramResult.emplace_back(inst.getResult(0));
    } else if (depth <= 32) {
      hw::InstanceOp inst = use6bitRAM(op, rewriter, place, readOp, writeOp);
      this->locOp = inst;
      dramResult.emplace_back(inst.getResult(0));
      dramResult.emplace_back(inst.getResult(1));
      dramResult.emplace_back(inst.getResult(2));
      place += 5;
    } else if (depth <= 64) {
      hw::InstanceOp inst = use3BitRAM(op, rewriter, readOp, writeOp, place);
      this->locOp = inst;
      dramResult.emplace_back(inst.getResult(0));
      dramResult.emplace_back(inst.getResult(1));
      dramResult.emplace_back(inst.getResult(2));
      place += 2;
    }
    dramMap = this->dramMap;
  }

  llvm::SmallVector<Value> getDramResult() { return dramResult; }
  ~CreateInstance() = default;

private:
  hw::HWModuleExternOp
  createX1DModuleExtern(ModuleOp module, seq::FirMemOp op,
                        ConversionPatternRewriter &rewriter) {
    rewriter.setInsertionPointToStart(module.getBody());
    SmallVector<hw::PortInfo> ports;
    StringAttr moduleName =
        rewriter.getStringAttr("RAM" + std::to_string(this->depth) + "X1D");
    int logdepth = std::log2(depth);
    ports.emplace_back(hw::PortInfo{{rewriter.getStringAttr("WCLK"),
                                     seq::ClockType::get(rewriter.getContext()),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("D"), rewriter.getIntegerType(1),
                      hw::ModulePort::Direction::Input}});
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("WE"), rewriter.getIntegerType(1),
                      hw::ModulePort::Direction::Input}});
    for (int i = 0; i < logdepth; i++) {
      ports.emplace_back(hw::PortInfo{
          {rewriter.getStringAttr("A" + std::to_string(i)),
           rewriter.getIntegerType(1), hw::ModulePort::Direction::Input}});
    }
    for (int i = 0; i < logdepth; i++) {
      ports.emplace_back(hw::PortInfo{
          {rewriter.getStringAttr("DPRA" + std::to_string(i)),
           rewriter.getIntegerType(1), hw::ModulePort::Direction::Input}});
    }
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("DPO"), rewriter.getIntegerType(1),
                      hw::ModulePort::Direction::Output}});
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("SPO"), rewriter.getIntegerType(1),
                      hw::ModulePort::Direction::Output}});
    auto externmodule =
        rewriter.create<hw::HWModuleExternOp>(op->getLoc(), moduleName, ports);
    rewriter.setInsertionPointAfter(this->locOp);
    return externmodule;
  }

  hw::HWModuleExternOp
  create6bitModuleExtern(ModuleOp module, seq::FirMemOp op,
                         ConversionPatternRewriter &rewriter) {

    rewriter.setInsertionPointToStart(module.getBody());
    SmallVector<hw::PortInfo> ports;
    StringAttr moduleName = rewriter.getStringAttr("RAM32M");
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("DIA"), rewriter.getIntegerType(2),
                      hw::ModulePort::Direction::Input}});
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("DIB"), rewriter.getIntegerType(2),
                      hw::ModulePort::Direction::Input}});
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("DIC"), rewriter.getIntegerType(2),
                      hw::ModulePort::Direction::Input}});
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("DID"), rewriter.getIntegerType(2),
                      hw::ModulePort::Direction::Input}});
    ports.emplace_back(hw::PortInfo{{rewriter.getStringAttr("ADDRA"),
                                     rewriter.getIntegerType(5),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(hw::PortInfo{{rewriter.getStringAttr("ADDRB"),
                                     rewriter.getIntegerType(5),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(hw::PortInfo{{rewriter.getStringAttr("ADDRC"),
                                     rewriter.getIntegerType(5),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(hw::PortInfo{{rewriter.getStringAttr("ADDRD"),
                                     rewriter.getIntegerType(5),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("WE"), rewriter.getIntegerType(1),
                      hw::ModulePort::Direction::Input}});
    ports.emplace_back(hw::PortInfo{{rewriter.getStringAttr("WCLK"),
                                     seq::ClockType::get(rewriter.getContext()),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("DOA"), rewriter.getIntegerType(2),
                      hw::ModulePort::Direction::Output}});
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("DOB"), rewriter.getIntegerType(2),
                      hw::ModulePort::Direction::Output}});
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("DOC"), rewriter.getIntegerType(2),
                      hw::ModulePort::Direction::Output}});
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("DOD"), rewriter.getIntegerType(2),
                      hw::ModulePort::Direction::Output}});
    auto externmodule =
        rewriter.create<hw::HWModuleExternOp>(op->getLoc(), moduleName, ports);
    rewriter.setInsertionPointAfter(this->locOp);
    return externmodule;
  }

  hw::HWModuleExternOp
  create3bitModuleExtern(ModuleOp module, seq::FirMemOp op,
                         ConversionPatternRewriter &rewriter) {
    rewriter.setInsertionPointToStart(module.getBody());
    SmallVector<hw::PortInfo> ports;
    StringAttr moduleName = rewriter.getStringAttr("RAM64M");
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("DIA"), rewriter.getIntegerType(1),
                      hw::ModulePort::Direction::Input}});
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("DIB"), rewriter.getIntegerType(1),
                      hw::ModulePort::Direction::Input}});
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("DIC"), rewriter.getIntegerType(1),
                      hw::ModulePort::Direction::Input}});
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("DID"), rewriter.getIntegerType(1),
                      hw::ModulePort::Direction::Input}});
    ports.emplace_back(hw::PortInfo{{rewriter.getStringAttr("ADDRA"),
                                     rewriter.getIntegerType(6),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(hw::PortInfo{{rewriter.getStringAttr("ADDRB"),
                                     rewriter.getIntegerType(6),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(hw::PortInfo{{rewriter.getStringAttr("ADDRC"),
                                     rewriter.getIntegerType(6),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(hw::PortInfo{{rewriter.getStringAttr("ADDRD"),
                                     rewriter.getIntegerType(6),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("WE"), rewriter.getIntegerType(1),
                      hw::ModulePort::Direction::Input}});
    ports.emplace_back(hw::PortInfo{{rewriter.getStringAttr("WCLK"),
                                     seq::ClockType::get(rewriter.getContext()),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("DOA"), rewriter.getIntegerType(1),
                      hw::ModulePort::Direction::Output}});
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("DOB"), rewriter.getIntegerType(1),
                      hw::ModulePort::Direction::Output}});
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("DOC"), rewriter.getIntegerType(1),
                      hw::ModulePort::Direction::Output}});
    ports.emplace_back(
        hw::PortInfo{{rewriter.getStringAttr("DOD"), rewriter.getIntegerType(1),
                      hw::ModulePort::Direction::Output}});
    auto externmodule =
        rewriter.create<hw::HWModuleExternOp>(op->getLoc(), moduleName, ports);
    rewriter.setInsertionPointAfter(this->locOp);
    return externmodule;
  }
  hw::InstanceOp createInsetance(ConversionPatternRewriter &rewriter,
                                 llvm::SmallVector<Value> &symbolInputs,
                                 seq::FirMemOp op) {
    auto exrernOp = dramMap[depth];
    if (exrernOp == nullptr) {
      exrernOp = createX1DModuleExtern(module, op, rewriter);
      dramMap[depth] = exrernOp;
    }

    auto instanceOp = rewriter.create<hw::InstanceOp>(
        op->getLoc(), exrernOp,
        rewriter.getStringAttr("RAM" + std::to_string(this->depth) + "X1D_" +
                               std::to_string(moduleNum++)),
        symbolInputs);
    this->locOp = instanceOp;
    return instanceOp;
  }

  hw::InstanceOp useX1D(seq::FirMemOp op, ConversionPatternRewriter &rewriter,
                        seq::FirMemReadOp readop, seq::FirMemWriteOp writeOp,
                        int extractIndex) {
    auto extractOp = rewriter.create<comb::ExtractOp>(
        op->getLoc(), writeOp.getData(), extractIndex, 1);
    llvm::SmallVector<Value> symbolInputs;
    symbolInputs.emplace_back(writeOp.getClk());
    symbolInputs.emplace_back(extractOp.getResult());
    symbolInputs.emplace_back(writeOp.getEnable());

    mlir::Value wraddr = writeOp.getAddress();
    auto constant0 =
        rewriter.create<hw::ConstantOp>(op->getLoc(), rewriter.getI1Type(), 0);
    for (int i = 0; i < int(log2(depth)); i++) {
      if (static_cast<int>(wraddr.getType().getIntOrFloatBitWidth()) <= i) {
        symbolInputs.emplace_back(constant0);
      } else {
        auto extractop =
            rewriter.create<comb::ExtractOp>(op->getLoc(), wraddr, i, 1);
        symbolInputs.emplace_back(extractop.getResult());
      }
    }
    mlir::Value rdaddr = readop.getAddress();
    for (int i = 0; i < int(log2(depth)); i++) {
      if (static_cast<int>(rdaddr.getType().getIntOrFloatBitWidth()) <= i) {
        symbolInputs.emplace_back(constant0);
      } else {
        auto extractop =
            rewriter.create<comb::ExtractOp>(op->getLoc(), rdaddr, i, 1);
        this->locOp = extractop;
        symbolInputs.emplace_back(extractop.getResult());
      }
    }
    return createInsetance(rewriter, symbolInputs, op);
  }
  hw::InstanceOp use6bitRAM(seq::FirMemOp op,
                            ConversionPatternRewriter &rewriter,
                            int extractPlace, seq::FirMemReadOp readOp,
                            seq::FirMemWriteOp writeOp) {
    auto extractOp0 = rewriter.create<comb::ExtractOp>(
        op->getLoc(), writeOp.getData(), extractPlace, 2);
    auto extractOp1 = rewriter.create<comb::ExtractOp>(
        op->getLoc(), writeOp.getData(), extractPlace + 2, 2);
    auto extractOp2 = rewriter.create<comb::ExtractOp>(
        op->getLoc(), writeOp.getData(), extractPlace + 4, 2);

    this->locOp = extractOp2;
    llvm::SmallVector<Value> symbolInputs;
    symbolInputs.emplace_back(extractOp0.getResult());
    symbolInputs.emplace_back(extractOp1.getResult());
    symbolInputs.emplace_back(extractOp2.getResult());
    auto constant02 = rewriter.create<hw::ConstantOp>(
        op->getLoc(), rewriter.getIntegerType(2),
        rewriter.getIntegerAttr(rewriter.getIntegerType(2), 0));
    auto constant0 = rewriter.create<hw::ConstantOp>(
        op->getLoc(), rewriter.getIntegerType(1),
        rewriter.getIntegerAttr(rewriter.getIntegerType(1), 0));
    symbolInputs.emplace_back(constant02);

    mlir::Value rdaddr = readOp.getAddress();
    mlir::Value wraddr = writeOp.getAddress();
    if (static_cast<int>(rdaddr.getType().getIntOrFloatBitWidth()) <
        int(log2(this->depth))) {
      llvm::SmallVector<mlir::Value> constantVec;
      for (int i = 0;
           i < int(log2(this->depth)) -
                   static_cast<int>(rdaddr.getType().getIntOrFloatBitWidth());
           i++) {
        constantVec.emplace_back(constant0);
      }
      constantVec.emplace_back(rdaddr);
      rdaddr = rewriter.create<comb::ConcatOp>(op->getLoc(), constantVec);
      this->locOp = rdaddr.getDefiningOp();
    } else if (static_cast<int>(rdaddr.getType().getIntOrFloatBitWidth()) >
               int(log2(this->depth))) {
      rdaddr = rewriter.create<comb::ExtractOp>(op->getLoc(), rdaddr, 0,
                                                int(log2(this->depth)));
      this->locOp = rdaddr.getDefiningOp();
    }
    if (static_cast<int>(wraddr.getType().getIntOrFloatBitWidth()) <
        int(log2(this->depth))) {
      llvm::SmallVector<mlir::Value> constantVec;
      for (int i = 0;
           i < int(log2(this->depth)) -
                   static_cast<int>(wraddr.getType().getIntOrFloatBitWidth());
           i++) {
        constantVec.emplace_back(constant0);
      }
      constantVec.emplace_back(wraddr);
      wraddr = rewriter.create<comb::ConcatOp>(op->getLoc(), constantVec);
      this->locOp = wraddr.getDefiningOp();
    } else if (static_cast<int>(wraddr.getType().getIntOrFloatBitWidth()) >
               int(log2(this->depth))) {
      wraddr = rewriter.create<comb::ExtractOp>(op->getLoc(), wraddr, 0,
                                                int(log2(this->depth)));
      this->locOp = wraddr.getDefiningOp();
    }
    symbolInputs.emplace_back(rdaddr);
    symbolInputs.emplace_back(rdaddr);
    symbolInputs.emplace_back(rdaddr);
    symbolInputs.emplace_back(wraddr);
    symbolInputs.emplace_back(writeOp.getEnable());
    symbolInputs.emplace_back(writeOp.getClk());

    auto externOp = dramMap[this->depth + 1];
    if (externOp == nullptr) {
      externOp = create6bitModuleExtern(module, op, rewriter);
      dramMap[this->depth + 1] = externOp;
    }
    return rewriter.create<hw::InstanceOp>(
        op->getLoc(), externOp,
        rewriter.getStringAttr("RAM32M_" + std::to_string(moduleNum++)),
        symbolInputs);
  }
  hw::InstanceOp use3BitRAM(seq::FirMemOp op,
                            ConversionPatternRewriter &rewriter,
                            seq::FirMemReadOp readOp,
                            seq::FirMemWriteOp writeOp, int extractPlace) {
    auto extractOp0 = rewriter.create<comb::ExtractOp>(
        op->getLoc(), writeOp.getData(), extractPlace, 1);
    auto extractOp1 = rewriter.create<comb::ExtractOp>(
        op->getLoc(), writeOp.getData(), extractPlace + 1, 1);
    auto extractOp2 = rewriter.create<comb::ExtractOp>(
        op->getLoc(), writeOp.getData(), extractPlace + 2, 1);

    this->locOp = extractOp2;
    llvm::SmallVector<Value> symbolInputs;
    symbolInputs.emplace_back(extractOp0.getResult());
    symbolInputs.emplace_back(extractOp1.getResult());
    symbolInputs.emplace_back(extractOp2.getResult());
    auto constant0 = rewriter.create<hw::ConstantOp>(
        op->getLoc(), rewriter.getIntegerType(1),
        rewriter.getIntegerAttr(rewriter.getIntegerType(1), 0));
    symbolInputs.emplace_back(constant0); // DOD

    mlir::Value rdaddra = readOp.getAddress();
    mlir::Value wraddra = writeOp.getAddress();

    if (static_cast<int>(rdaddra.getType().getIntOrFloatBitWidth()) <
        int(log2(this->depth))) {
      llvm::SmallVector<mlir::Value> constantVec;
      for (int i = 0;
           i < int(log2(this->depth)) -
                   static_cast<int>(rdaddra.getType().getIntOrFloatBitWidth());
           i++) {
        constantVec.emplace_back(constant0);
      }
      constantVec.emplace_back(rdaddra);
      rdaddra = rewriter.create<comb::ConcatOp>(op->getLoc(), constantVec);
      this->locOp = rdaddra.getDefiningOp();
    } else if (static_cast<int>(rdaddra.getType().getIntOrFloatBitWidth()) >
               int(log2(this->depth))) {
      rdaddra = rewriter.create<comb::ExtractOp>(op->getLoc(), rdaddra, 0,
                                                 int(log2(this->depth)));
      this->locOp = rdaddra.getDefiningOp();
    }

    if (static_cast<int>(wraddra.getType().getIntOrFloatBitWidth()) <
        int(log2(this->depth))) {
      llvm::SmallVector<mlir::Value> constantVec;
      for (int i = 0;
           i < int(log2(this->depth)) -
                   static_cast<int>(wraddra.getType().getIntOrFloatBitWidth());
           i++) {
        constantVec.emplace_back(constant0);
      }
      constantVec.emplace_back(wraddra);
      wraddra = rewriter.create<comb::ConcatOp>(op->getLoc(), constantVec);
      this->locOp = wraddra.getDefiningOp();
    } else if (static_cast<int>(wraddra.getType().getIntOrFloatBitWidth()) >
               int(log2(this->depth))) {
      wraddra = rewriter.create<comb::ExtractOp>(op->getLoc(), wraddra, 0,
                                                 int(log2(this->depth)));
      this->locOp = wraddra.getDefiningOp();
    }
    symbolInputs.emplace_back(rdaddra);
    symbolInputs.emplace_back(rdaddra);
    symbolInputs.emplace_back(rdaddra);
    symbolInputs.emplace_back(wraddra);
    symbolInputs.emplace_back(writeOp.getEnable());
    symbolInputs.emplace_back(writeOp.getClk());

    auto externOp = dramMap[this->depth + 1];
    if (externOp == nullptr) {
      externOp = create3bitModuleExtern(module, op, rewriter);
      dramMap[this->depth + 1] = externOp;
    }

    return rewriter.create<hw::InstanceOp>(
        op->getLoc(), externOp,
        rewriter.getStringAttr("RAM64M_" + std::to_string(moduleNum++)),
        symbolInputs);
  }
  hw::HWModuleExternOp createModuleExtern(ModuleOp op, MLIRContext &context,
                                          int moduleNum, int width) {
    OpBuilder builder(&context);
    builder.setInsertionPointToStart(op.getBody());
    StringAttr moduleName = builder.getStringAttr("RAMB36E2");
    SmallVector<hw::PortInfo> ports;
    ports.emplace_back(hw::PortInfo{{builder.getStringAttr("CLKARDCLK"),
                                     seq::ClockType::get(&context),
                                     hw::ModulePort::Direction::Input}});
    // Port A configuration
    ports.emplace_back(hw::PortInfo{{builder.getStringAttr("ADDRARDADDR"),
                                     builder.getIntegerType(15),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(hw::PortInfo{{builder.getStringAttr("ENARDEN"),
                                     builder.getIntegerType(1),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(hw::PortInfo{{builder.getStringAttr("DINADIN"),
                                     builder.getIntegerType(32),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(hw::PortInfo{{builder.getStringAttr("DINPADINP"),
                                     builder.getIntegerType(4),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(hw::PortInfo{{builder.getStringAttr("DOUTADOUT"),
                                     builder.getIntegerType(32),
                                     hw::ModulePort::Direction::Output}});
    ports.emplace_back(hw::PortInfo{{builder.getStringAttr("DOUTPADOUTP"),
                                     builder.getIntegerType(4),
                                     hw::ModulePort::Direction::Output}});
    // Port B configuration
    ports.emplace_back(hw::PortInfo{{builder.getStringAttr("CLKBWRCLK"),
                                     seq::ClockType::get(&context),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(hw::PortInfo{{builder.getStringAttr("ADDRBWRADDR"),
                                     builder.getIntegerType(15),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(hw::PortInfo{{builder.getStringAttr("ENBWREN"),
                                     builder.getIntegerType(1),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(hw::PortInfo{{builder.getStringAttr("DINBDIN"),
                                     builder.getIntegerType(32),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(hw::PortInfo{{builder.getStringAttr("DINPBDINP"),
                                     builder.getIntegerType(4),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(hw::PortInfo{{builder.getStringAttr("DOUTBDOUT"),
                                     builder.getIntegerType(32),
                                     hw::ModulePort::Direction::Output}});
    ports.emplace_back(hw::PortInfo{{builder.getStringAttr("DOUTPBDOUTP"),
                                     builder.getIntegerType(4),
                                     hw::ModulePort::Direction::Output}});
    ports.emplace_back(hw::PortInfo{{builder.getStringAttr("RSTRAMARSTRAM"),
                                     builder.getIntegerType(1),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(hw::PortInfo{{builder.getStringAttr("RSTREGARSTREG"),
                                     builder.getIntegerType(1),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(hw::PortInfo{{builder.getStringAttr("REGCEAREGCE"),
                                     builder.getIntegerType(1),
                                     hw::ModulePort::Direction::Input}});
    ports.emplace_back(
        hw::PortInfo{{builder.getStringAttr("WEA"), builder.getIntegerType(4),
                      hw::ModulePort::Direction::Input}});
    ports.emplace_back(
        hw::PortInfo{{builder.getStringAttr("WEBWE"), builder.getIntegerType(8),
                      hw::ModulePort::Direction::Input}});
    auto externmodule =
        builder.create<hw::HWModuleExternOp>(op->getLoc(), moduleName, ports);
    SmallVector<Attribute> params;
    params.emplace_back(hw::ParamDeclAttr::get(
        builder.getContext(), builder.getStringAttr("READ_WIDTH_A"),
        builder.getIntegerType(32),
        builder.getIntegerAttr(builder.getIntegerType(32), -1)));
    params.emplace_back(hw::ParamDeclAttr::get(
        builder.getContext(), builder.getStringAttr("WRITE_WIDTH_A"),
        builder.getIntegerType(32),
        builder.getIntegerAttr(builder.getIntegerType(32), -1)));
    params.emplace_back(hw::ParamDeclAttr::get(
        builder.getContext(), builder.getStringAttr("DOA_REG"),
        builder.getIntegerType(1),
        builder.getIntegerAttr(builder.getIntegerType(1), 1)));
    params.emplace_back(hw::ParamDeclAttr::get(
        builder.getContext(), builder.getStringAttr("INIT_A"),
        builder.getIntegerType(36),
        builder.getIntegerAttr(builder.getIntegerType(36), 1)));
    params.emplace_back(hw::ParamDeclAttr::get(
        builder.getContext(), builder.getStringAttr("WRITE_MODE_A"),
        builder.getStringAttr("NONE").getType(),
        builder.getStringAttr("NONE")));
    {
      externmodule->setAttr("parameters",
                            ArrayAttr::get(builder.getContext(), params));
    }
    return externmodule;
  }

  Operation *locOp;
  int depth, bitSize;
  llvm::SmallVector<Value> dramResult;
  llvm::DenseMap<int, hw::HWModuleExternOp> dramMap;
  int moduleNum = 0;
  ModuleOp module;
  // seq::FirMemReadOp readOp,writeOp;
};

struct FirMemOpConversion : public OpConversionPattern<seq::FirMemOp> {
  using OpConversionPattern::OpConversionPattern;

  // Handle small BRAM memory
  void handleSmallBram(seq::FirMemOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter, int index,
                       CreateInstance &bram) const {
    while (bitSize > portSizeArray[index]) {
      index++;
    }
    auto portSize = portSizeArray[index];
    bool hasdoutpadout =
        (bitSize > 32 || (bitSize > 16 && bitSize <= 18) || bitSize == 9);

    std::pair<hw::InstanceOp, seq::FirMemReadWriteOp> instanceOp;
    // int portSize = portsize;
    if (portSize > 32) {
      portSize = 32;
    } else if (portSize > 16 && portSize <= 18) {
      portSize = 16;
    } else if (portSize == 9) {
      portSize = 8;
    }
    this->extractPlace = -1;
    instanceOp =
        bram.instanceBlockRam(op, rewriter, portSize, extractPlace, blockRam);
    auto instV = instanceOp.first.getResult(0);

    // Adjust the result according to bit width
    if (bitSize < 32) {
      if (hasdoutpadout) {
        int len = 0;
        if (bitSize == 9)
          len = 8;
        else if (bitSize > 16 && bitSize <= 18)
          len = 16;
        else if (bitSize > 32)
          len = 32;
        instV = rewriter.create<comb::ExtractOp>(
            op->getLoc(), instanceOp.first.getResult(0), 0, len);
      } else {
        instV = rewriter.create<comb::ExtractOp>(
            op->getLoc(), instanceOp.first.getResult(0), 0, bitSize);
      }
    }

    // Process additional mask bits if needed
    if (op.getMemory().getType().getMaskWidth() == std::nullopt &&
        hasdoutpadout) {
      int len = 0;
      if (bitSize == 9)
        len = 1;
      else if (bitSize > 16 && bitSize <= 18)
        len = bitSize - 16;
      else if (bitSize > 32)
        len = bitSize - 32;
      mlir::Value doutpadout = rewriter.create<comb::ExtractOp>(
          op->getLoc(), instanceOp.first.getResult(1), 0, len);
      instV = rewriter.create<comb::ConcatOp>(op->getLoc(), instV, doutpadout);
    }

    instanceOp.second.replaceAllUsesWith(instV);

    // Cleanup operations
    for (auto &useop : op->getUses()) {
      useop.getOwner()->erase();
    }
    op->erase();
  }

  // Create single BRAM slice
  void createBramSlice(seq::FirMemOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter, int maxBitsize,
                       int totalSize, int sliceIndex,
                       std::vector<mlir::Value> &instValue,
                       std::vector<std::pair<Operation *, int>> &instUses,
                       CreateInstance &bram) const {
    hw::InstanceOp instanceop;
    seq::FirMemReadWriteOp readWriteOp;
    int portSize = maxBitsize;
    this->extractPlace = sliceIndex * maxBitsize;

    std::tie(instanceop, readWriteOp) =
        bram.instanceBlockRam(op, rewriter, portSize, extractPlace, blockRam);
    // Collect usage information
    if (instUses.empty()) {
      for (auto &uses : readWriteOp->getUses()) {
        instUses.emplace_back(uses.getOwner(), uses.getOperandNumber());
      }
    }

    // Create data bits
    int extractPlace = std::min(maxBitsize, totalSize);
    if (extractPlace > 32) {
      extractPlace = 32;
    } else if (extractPlace > 16 && extractPlace <= 18) {
      extractPlace = 16;
    } else if (extractPlace == 9) {
      extractPlace = 8;
    }
    mlir::Value instV = rewriter.create<comb::ExtractOp>(
        op->getLoc(), instanceop.getResult(0), 0, extractPlace);
    instValue.emplace_back(instV);

    // Create mask bits (if needed)
    bool needsMask = op.getMemory().getType().getMaskWidth() == std::nullopt &&
                     (maxBitsize == 9 ||
                      (maxBitsize > 16 && maxBitsize <= 18) || maxBitsize > 32);
    if (needsMask && totalSize >= maxBitsize) {
      int len = 0;
      if (maxBitsize == 9)
        len = 1;
      else if (maxBitsize > 16 && maxBitsize <= 18)
        len = maxBitsize - 16;
      else if (maxBitsize > 32)
        len = maxBitsize - 32;

      instV = rewriter.create<comb::ExtractOp>(op->getLoc(),
                                               instanceop.getResult(1), 0, len);
      instValue.emplace_back(instV);
    }
  }
  // Handle large BRAM memory
  void handleLargeBram(seq::FirMemOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter, int index,
                       CreateInstance &bram) const {
    // Calculate appropriate bit width
    int maxBitsize = 36864.0 / depth;
    while (maxBitsize > portSizeArray[index]) {
      index++;
    }
    maxBitsize = portSizeArray[index];

    // Special case handling
    if (bitSize == 64 &&
        op.getMemory().getType().getMaskWidth() != std::nullopt) {
      maxBitsize = 32;
    }

    std::vector<mlir::Value> instValue;
    std::vector<std::pair<Operation *, int>> instUses;
    int totalSize = bitSize;
    int bramSize =
        static_cast<int>(std::ceil(bitSize * 1.0 / maxBitsize * 1.0));

    // Create BRAM instance for each slice
    for (int i = 0; i < bramSize; i++) {
      createBramSlice(op, adaptor, rewriter, maxBitsize, totalSize, i,
                      instValue, instUses, bram);
      totalSize -= maxBitsize;
    }

    // Connect all results
    std::reverse(instValue.begin(), instValue.end());
    if (!instUses.empty()) {
      auto concatop = rewriter.create<comb::ConcatOp>(op->getLoc(), instValue);
      for (auto &useop : instUses) {
        useop.first->setOperand(useop.second, concatop.getResult());
      }
    }

    // Cleanup operations
    for (auto &useop : op->getUses()) {
      useop.getOwner()->erase();
    }
    op->erase();
  }

  void setBram(seq::FirMemOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const {
    int index = 0;
    bitSize = op.getType().getWidth();
    depth = op.getType().getDepth();

    CreateInstance bram(module, bitSize, depth);
    // For small memory and standard sizes, use simple processing logic
    if (!(bitSize * depth > 36864 || bitSize > 36)) {
      handleSmallBram(op, adaptor, rewriter, index, bram);
      return;
    }

    // Handle large memory and non-standard sizes
    handleLargeBram(op, adaptor, rewriter, index, bram);
  }

  void setDistRam(seq::FirMemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter, int depth) const {
    int depthIndex = 0;
    while (depthIndex < 4 && depthArray[depthIndex] < depth) {
      depthIndex++;
    }
    this->bitSize = op.getType().getWidth();
    this->depth = depthArray[depthIndex];
    this->locOp = op;

    seq::FirMemReadOp readOp = nullptr;
    seq::FirMemWriteOp writeOp = nullptr;
    for (auto &useop : op->getUses()) {
      if (auto readop = llvm::dyn_cast<seq::FirMemReadOp>(useop.getOwner())) {
        readOp = readop;
      } else if (auto writeop =
                     llvm::dyn_cast<seq::FirMemWriteOp>(useop.getOwner())) {
        writeOp = writeop;
      } else {
        assert(
            false &&
            "firmemop.usesop is not a seq::FirMemReadOp or seq::FirMemWriteOp");
      }
    }
    if (writeOp == nullptr) {
      assert(false && "writeOp is null");
    }
    rewriter.setInsertionPointAfter(writeOp);
    CreateInstance dist(module, bitSize, this->depth);
    for (int i = 0; i < static_cast<int>(op.getType().getWidth()); i++) {
      dist.instanceDistRam(bitSize, i, readOp, writeOp, op, rewriter, dramMap);
    }
    auto dramResult = dist.getDramResult();
    std::reverse(dramResult.begin(), dramResult.end());
    auto resultOp = rewriter.create<comb::ConcatOp>(op->getLoc(), dramResult);
    rewriter.replaceAllUsesWith(readOp, resultOp.getResult());
    readOp.erase();
    writeOp.erase();
    op.erase();
    dramResult.clear();
    this->depth = 0;
    this->bitSize = 0;
  }
  LogicalResult
  matchAndRewrite(seq::FirMemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    this->module = op->getParentOfType<ModuleOp>();
    if (op.getReadLatency() == 1) {
      setBram(op, adaptor, rewriter);
    } else {
      setDistRam(op, adaptor, rewriter, op.getType().getDepth());
    }

    return success();
  }

private:
  mutable int bitSize = 0;
  mutable int depth = 0;
  mutable int extractPlace = 0;
  mutable hw::HWModuleExternOp blockRam;
  mutable llvm::DenseMap<int, hw::HWModuleExternOp> dramMap;
  int portSizeArray[6] = {1, 2, 4, 9, 18, 36};
  int depthArray[4] = {32, 64, 128, 256};
  mutable ModuleOp module;
  mutable Operation *locOp;
};

void firMemOpConversion(RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<FirMemOpConversion>(context);
}

void CoreMemoryMappingPass::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();
  OpBuilder builder(&context);
  IRRewriter rewriter(&context);
  ConversionTarget target(context);
  target.addLegalDialect<hw::HWDialect, seq::SeqDialect, comb::CombDialect>();
  target.addIllegalOp<seq::FirMemOp>();
  target.addLegalOp<ModuleOp>();
  RewritePatternSet patterns(&context);
  firMemOpConversion(patterns);
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
