#include "circt/Conversion/CoreDSPMapping.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqAttributes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/WithColor.h"

namespace circt {
#define GEN_PASS_DEF_COREDSPMAPPING
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;
using namespace seq;

//===----------------------------------------------------------------------===//
// Core DSP Mapping Pass
//===----------------------------------------------------------------------===//

namespace {
struct CoreDSPMappingPass
    : public circt::impl::CoreDSPMappingBase<CoreDSPMappingPass> {
  void runOnOperation() override;
};
} // namespace

/// Creates a divider implementation based on the generator attributes
static HWModuleOp createDividerImplementation(OpBuilder &builder,
                                              HWModuleGeneratedOp oldModule,
                                              StringAttr nameAttr) {
  // Extract attributes
  auto input1Size =
      oldModule->getAttrOfType<IntegerAttr>("input1Size").getUInt();
  auto input2Size =
      oldModule->getAttrOfType<IntegerAttr>("input2Size").getUInt();
  auto outputSize =
      oldModule->getAttrOfType<IntegerAttr>("outputSize").getUInt();

  // Create the new module with the same ports
  auto newModule = builder.create<HWModuleOp>(oldModule.getLoc(), nameAttr,
                                              oldModule.getPortList());

  // Start building the implementation
  ImplicitLocOpBuilder ob(oldModule.getLoc(), newModule.getBody());

  // Get the port values (numerator, denominator, quotient)
  Value numerator = newModule.getBody().getArgument(0); // numerator (dividend)
  Value denominator =
      newModule.getBody().getArgument(1); // denominator (divisor)

  // Implementation of a simple restoring division algorithm
  // This is a combinational implementation that performs long division

  // Create a function to implement LessThan operation
  auto createLessThan = [&](Value a, Value b, unsigned width) -> Value {
    // We'll implement a simple comparison logic
    // For each bit position, starting from MSB:
    // If bits differ, the value with 0 is less than the value with 1
    // If bits are equal, continue to next bit

    // Default result if everything is equal
    Value result = ob.create<hw::ConstantOp>(ob.getI1Type(), 0);

    for (int i = width - 1; i >= 0; i--) {
      // Extract bits at position i
      Value bitA = ob.create<comb::ExtractOp>(a, i, 1);
      Value bitB = ob.create<comb::ExtractOp>(b, i, 1);

      // If bits are different
      Value bitsNotEqual = ob.create<comb::XorOp>(bitA, bitB);

      // If B has 1 and A has 0, then A < B
      Value aLessThanB = ob.create<comb::AndOp>(
          bitsNotEqual, ob.create<comb::ICmpOp>(
                            comb::ICmpPredicate::eq, bitA,
                            ob.create<hw::ConstantOp>(ob.getI1Type(), 0)));

      // Update result if bits are different
      result = ob.create<comb::MuxOp>(bitsNotEqual, aLessThanB, result);
    }

    return result;
  };

  // Create working registers for the division algorithm
  Value remainder =
      ob.create<hw::ConstantOp>(ob.getIntegerType(input1Size + 1), 0);
  SmallVector<Value> quotientBits;

  // Implement restoring division algorithm
  for (unsigned i = 0; i < outputSize; i++) {
    // Shift remainder left by 1 bit
    Value shiftedRemainder = ob.create<comb::ConcatOp>(
        ob.create<comb::ExtractOp>(remainder, 0, input1Size),
        ob.create<comb::ExtractOp>(numerator, input1Size - i - 1, 1));

    // Try to subtract denominator from shifted remainder
    Value extendedDenominator = ob.create<comb::ConcatOp>(
        ob.create<hw::ConstantOp>(ob.getI1Type(), 0), denominator);

    Value diff = ob.create<comb::SubOp>(shiftedRemainder, extendedDenominator);

    // Check if remainder >= denominator
    Value notLessThan = ob.create<comb::XorOp>(
        createLessThan(shiftedRemainder, extendedDenominator, input1Size + 1),
        ob.create<hw::ConstantOp>(ob.getI1Type(), 1));

    // If remainder >= denominator, update remainder and set quotient bit to 1
    // Otherwise, keep the shifted remainder and set quotient bit to 0
    remainder = ob.create<comb::MuxOp>(notLessThan, diff, shiftedRemainder);
    quotientBits.push_back(notLessThan);
  }

  // Connect the output (quotient)
  Value resultValue = ob.create<comb::ConcatOp>(quotientBits);
  // ob.create<hw::OutputOp>(ValueRange{resultValue});
  auto *outputOp = newModule.getBodyBlock()->getTerminator();
  outputOp->setOperands(ValueRange{resultValue});

  return newModule;
}

/// Creates a multiplier implementation based on the generator attributes
static HWModuleOp createMultiplierImplementation(OpBuilder &builder,
                                                 HWModuleGeneratedOp oldModule,
                                                 StringAttr nameAttr) {
  // Extract attributes
  auto input1Size =
      oldModule->getAttrOfType<IntegerAttr>("input1Size").getUInt();
  auto input2Size =
      oldModule->getAttrOfType<IntegerAttr>("input2Size").getUInt();
  auto outputSize =
      oldModule->getAttrOfType<IntegerAttr>("outputSize").getUInt();

  // Create the new module with the same ports
  auto newModule = builder.create<HWModuleOp>(oldModule.getLoc(), nameAttr,
                                              oldModule.getPortList());

  // Start building the implementation
  ImplicitLocOpBuilder ob(oldModule.getLoc(), newModule.getBody());

  // Get the port values (i1, i2, o)
  Value i1 = newModule.getBody().getArgument(0); // first input
  Value i2 = newModule.getBody().getArgument(1); // second input

  // Create bitwise implementation for multiplier
  // Extract each bit from inputs
  SmallVector<Value> i1Bits, i2Bits;
  for (unsigned i = 0; i < input1Size; ++i) {
    i1Bits.push_back(ob.create<comb::ExtractOp>(i1, i, 1));
  }
  for (unsigned i = 0; i < input2Size; ++i) {
    i2Bits.push_back(ob.create<comb::ExtractOp>(i2, i, 1));
  }

  // Calculate partial products
  SmallVector<SmallVector<Value, 16>, 16> partialProducts;
  partialProducts.resize(input2Size);

  for (unsigned i = 0; i < input2Size; ++i) {
    partialProducts[i].resize(input1Size + i);

    // Initialize with zeros
    for (unsigned j = 0; j < i; ++j) {
      partialProducts[i][j] = ob.create<hw::ConstantOp>(ob.getI1Type(), 0);
    }

    // Calculate partial product for this row
    for (unsigned j = 0; j < input1Size; ++j) {
      Value partialProduct = ob.create<comb::AndOp>(i1Bits[j], i2Bits[i]);
      partialProducts[i][j + i] = partialProduct;
    }
  }

  // Sum up the partial products
  // We'll use a simple ripple-carry approach for adding the rows
  SmallVector<Value> resultBits;
  resultBits.resize(outputSize);

  // Initialize result bits to 0
  for (unsigned i = 0; i < outputSize; ++i) {
    resultBits[i] = ob.create<hw::ConstantOp>(ob.getI1Type(), 0);
  }

  // Add each partial product to the result
  for (unsigned row = 0; row < input2Size; ++row) {
    SmallVector<Value> tempSum = resultBits;
    Value carry = ob.create<hw::ConstantOp>(ob.getI1Type(), 0);

    for (unsigned bitPos = 0; bitPos < outputSize; ++bitPos) {
      // Get the bit from the partial product (or 0 if beyond its width)
      Value ppBit = (bitPos < partialProducts[row].size())
                        ? partialProducts[row][bitPos]
                        : ob.create<hw::ConstantOp>(ob.getI1Type(), 0);

      // Sum = a ^ b ^ carry
      Value sum = ob.create<comb::XorOp>(
          ob.create<comb::XorOp>(tempSum[bitPos], ppBit), carry);

      // Carry = (a & b) | ((a ^ b) & carry)
      Value and1 = ob.create<comb::AndOp>(tempSum[bitPos], ppBit);
      Value xor1 = ob.create<comb::XorOp>(tempSum[bitPos], ppBit);
      Value and2 = ob.create<comb::AndOp>(xor1, carry);
      carry = ob.create<comb::OrOp>(and1, and2);

      resultBits[bitPos] = sum;

      // If we're at the last bit, we need to handle overflow
      if (bitPos == outputSize - 1 &&
          carry != ob.create<hw::ConstantOp>(ob.getI1Type(), 0)) {
        // For simplicity, we'll ignore overflow
      }
    }
  }

  // Connect the output
  Value resultValue = ob.create<comb::ConcatOp>(resultBits);
  auto *outputOp = newModule.getBodyBlock()->getTerminator();
  outputOp->setOperands(ValueRange{resultValue});

  return newModule;
}

/// Create a Core DSP Mapping pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::createCoreDSPMappingPass() {
  return std::make_unique<CoreDSPMappingPass>();
}

/// Creates an adder implementation based on the generator attributes
static HWModuleOp createAdderImplementation(OpBuilder &builder,
                                            HWModuleGeneratedOp oldModule,
                                            StringAttr nameAttr) {
  // Extract attributes
  auto input1Size =
      oldModule->getAttrOfType<IntegerAttr>("input1Size").getUInt();
  auto input2Size =
      oldModule->getAttrOfType<IntegerAttr>("input2Size").getUInt();
  auto outputSize =
      oldModule->getAttrOfType<IntegerAttr>("outputSize").getUInt();

  // Create the new module with the same ports
  auto newModule = builder.create<HWModuleOp>(oldModule.getLoc(), nameAttr,
                                              oldModule.getPortList());

  // Start building the implementation
  ImplicitLocOpBuilder ob(oldModule.getLoc(), newModule.getBody());

  // Get the port values (cin, i1, i2, o, cout)
  Value cin = newModule.getBody().getArgument(0); // cin
  Value i1 = newModule.getBody().getArgument(1);  // first input
  Value i2 = newModule.getBody().getArgument(2);  // second input

  // Create bitwise implementation for adder
  // Extract each bit from inputs
  SmallVector<Value> i1Bits, i2Bits;
  for (unsigned i = 0; i < input1Size; ++i) {
    i1Bits.push_back(ob.create<comb::ExtractOp>(i1, i, 1));
  }
  for (unsigned i = 0; i < input2Size; ++i) {
    i2Bits.push_back(ob.create<comb::ExtractOp>(i2, i, 1));
  }

  // Implement full adder logic bit by bit
  SmallVector<Value> sumBits;
  Value carry = cin;

  for (unsigned i = 0; i < outputSize; ++i) {
    // For each bit position, perform full adder logic
    Value a = (i < input1Size) ? i1Bits[i]
                               : ob.create<hw::ConstantOp>(ob.getI1Type(), 0);
    Value b = (i < input2Size) ? i2Bits[i]
                               : ob.create<hw::ConstantOp>(ob.getI1Type(), 0);

    // XOR for sum bit
    Value sumBit = ob.create<comb::XorOp>(ob.create<comb::XorOp>(a, b), carry);
    sumBits.push_back(sumBit);

    // Carry logic: (a AND b) OR ((a XOR b) AND carry)
    Value ab = ob.create<comb::AndOp>(a, b);
    Value axorb = ob.create<comb::XorOp>(a, b);
    Value axorbAndCarry = ob.create<comb::AndOp>(axorb, carry);
    carry = ob.create<comb::OrOp>(ab, axorbAndCarry);
  }

  // Concatenate result bits to form output
  Value sumValue = ob.create<comb::ConcatOp>(sumBits);
  Value coutValue = carry;

  // Connect outputs 
  auto *outputOp = newModule.getBodyBlock()->getTerminator();
  if (outputSize > 1) {
    outputOp->setOperands(ValueRange{sumValue, coutValue});
  } else {
        outputOp->setOperands(ValueRange{
          ob.create<hw::ConstantOp>(ob.getIntegerType(1), 0), coutValue});
  }

  return newModule;
}

void CoreDSPMappingPass::runOnOperation() {
  auto topModule = getOperation();
  SymbolCache symbolCache;
  symbolCache.addDefinitions(topModule);
  Namespace mlirModuleNamespace;
  mlirModuleNamespace.add(symbolCache);

  SmallVector<HWModuleGeneratedOp> toErase;
  bool anythingChanged = false;

  for (auto op :
       llvm::make_early_inc_range(topModule.getOps<HWModuleGeneratedOp>())) {
    auto oldModule = cast<HWModuleGeneratedOp>(op);
    auto gen = oldModule.getGeneratorKind();
    auto genOp = cast<HWGeneratorSchemaOp>(
        SymbolTable::lookupSymbolIn(getOperation(), gen));

    OpBuilder builder(oldModule);
    auto nameAttr = builder.getStringAttr(oldModule.getName());

    HWModuleOp newModule;
    if (genOp.getDescriptor() == "OPERADDER") {
      newModule = createAdderImplementation(builder, oldModule, nameAttr);
      anythingChanged = true;
    } else if (genOp.getDescriptor() == "OPERMULT") {
      newModule = createMultiplierImplementation(builder, oldModule, nameAttr);
      anythingChanged = true;
    } else if (genOp.getDescriptor() == "OPERDIV") {
      newModule = createDividerImplementation(builder, oldModule, nameAttr);
      anythingChanged = true;
    }

    llvm::WithColor::warning()
        << "Core DSP Mapping pass: replacing " << oldModule.getName()
        << " with " << newModule.getName() << "\n";
        newModule.dump();
    oldModule.erase();
    anythingChanged = true;
  }
  if (!anythingChanged)
    markAllAnalysesPreserved();
}
