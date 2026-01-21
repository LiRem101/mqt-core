/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/OpType.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/ConstantPropagation/UnionTable.hpp"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/WalkResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MQTOpt/Transforms/ConstantPropagation/RewriteChecker.hpp>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mqt::ir::opt {

#define GEN_PASS_DEF_QUANTUMCONSTANTPROPAGATIONPASS
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

struct qcpObjects {
  qcp::UnionTable ut;
  llvm::DenseMap<mlir::Value, unsigned int> qubitToIndex;
  llvm::DenseMap<mlir::Value, std::vector<unsigned int>> memrefToQubitIndex;
  llvm::DenseMap<mlir::Value, std::vector<unsigned int>> memrefToBitIndex;
  llvm::DenseMap<mlir::Value, int64_t> integerValues;
  llvm::DenseMap<mlir::Value, double> doubleValues;
  llvm::DenseMap<mlir::Value, unsigned int> bitToIndex;
  mlir::Value trueValue;
  bool changed = false;
};

namespace {
using namespace mlir;

#define CREATE_OP_CASE(opType)                                                 \
  case qc::OpType::opType:                                                     \
    newOp = rewriter.create<opType##Op>(                                       \
        op->getLoc(), qubitType, newPosOutCtrlResultTypes,                     \
        newNegOutCtrlResultTypes, staticParams, mlir::DenseBoolArrayAttr{},    \
        op.getParams(), inQubits, newPosInCtrlOperands, newNegInCtrlOperands); \
    break;

LogicalResult
iterateThroughWorklist(PatternRewriter& rewriter,
                       std::vector<Operation*>& worklist, qcpObjects* qcp,
                       const std::vector<unsigned int>& posBitCtrls,
                       const std::vector<unsigned int>& negBitCtrls);

WalkResult removeIfElseBlock(scf::IfOp op, Block* blockToKeep,
                             Block* blockToErase,
                             std::vector<Operation*>& worklist,
                             PatternRewriter& rewriter) {
  scf::YieldOp const yieldOp = cast<scf::YieldOp>(blockToKeep->getTerminator());
  for (auto [result, yielded] :
       llvm::zip(op.getResults(), yieldOp->getOperands())) {
    result.replaceAllUsesWith(yielded);
  }
  std::ranges::replace(worklist, yieldOp, static_cast<Operation*>(nullptr));
  yieldOp->erase();
  rewriter.inlineBlockBefore(blockToKeep, op, op.getResults());
  for (Operation const& operation : *blockToErase) {
    std::ranges::replace(worklist, &operation,
                         static_cast<Operation*>(nullptr));
  }
  rewriter.eraseOp(op);
  return WalkResult::advance();
}

bool removeDiagonalGate(qcpObjects* qcp, UnitaryInterface op,
                        std::vector<unsigned int> targets,
                        std::vector<unsigned int> posCtrlsQuantum,
                        std::vector<unsigned int> negCtrlsQuantum) {
  const auto opName = op.getIdentifier().str();
  std::set<std::string> diagonalGates = {"z",   "s", "sdg", "t",
                                         "tdg", "p", "r",   "rzz"};
  if (!diagonalGates.contains(opName)) {
    return false;
  }

  if (opName == "r" || opName == "rzz") {
    // R and Rzz add a phase to all target values, meaning they only add a
    // global phase only if there are no controls
    if (!posCtrlsQuantum.empty() || !negCtrlsQuantum.empty()) {
      return false;
    }

    if (opName == "r") {
      return qcp::RewriteChecker::isOnlyOneSetNotZero(qcp->ut, targets,
                                                      {{0}, {1}});
    }
    return qcp::RewriteChecker::isOnlyOneSetNotZero(qcp->ut, targets,
                                                    {{0, 3}, {0, 1}});
  }

  std::vector<unsigned int> qubits;

  // All remaining gates put a phase to the one
  qubits.push_back(targets.at(0));
  unsigned int valueToAddPhase = 1;

  for (unsigned int i = 0; i < negCtrlsQuantum.size(); ++i) {
    qubits.push_back(negCtrlsQuantum.at(i));
  }
  for (unsigned int i = 0; i < posCtrlsQuantum.size(); ++i) {
    qubits.push_back(posCtrlsQuantum.at(i));
    valueToAddPhase +=
        static_cast<unsigned int>(pow(2, 1 + negCtrlsQuantum.size() + i) + 0.1);
  }

  std::vector<unsigned int> valuesThatDoNotGetPhase = {};
  for (unsigned int i = 0;
       i < static_cast<unsigned int>(pow(2, 1 + negCtrlsQuantum.size() +
                                                posCtrlsQuantum.size() + 0.1));
       ++i) {
    if (i != valueToAddPhase) {
      valuesThatDoNotGetPhase.push_back(i);
    }
  }

  return qcp::RewriteChecker::isOnlyOneSetNotZero(
      qcp->ut, qubits, {valuesThatDoNotGetPhase, {valueToAddPhase}});
}

/**
 *
 */
WalkResult handleFunc(qcpObjects* qcp, func::FuncOp op,
                      PatternRewriter& rewriter) {
  if (!isEntryPoint(op)) {
    throw std::domain_error(
        "Constant propagation does not support nested functions.");
  }
  // Create a true val
  rewriter.setInsertionPointToStart(&op.front());
  auto trueVal = rewriter.create<arith::ConstantIntOp>(op->getLoc(), 1, 1);
  qcp->trueValue = trueVal;
  return WalkResult::advance();
}

/**
 * For-loops are not yet part of the constant propagation.
 */
WalkResult handleFor() {
  throw std::domain_error(
      "Handling of for-loops not yet supported by constant propagation.");
}

WalkResult removeGate(UnitaryInterface op, PatternRewriter& rewriter) {
  for (const auto outQubit : op.getAllOutQubits()) {
    rewriter.replaceAllUsesWith(outQubit, op.getCorrespondingInput(outQubit));
  }
  rewriter.eraseOp(op);
  return WalkResult::advance();
}

UnitaryInterface removeCtrls(qcpObjects* qcp, UnitaryInterface op,
                             const std::set<unsigned int>& indicesToRemove,
                             PatternRewriter& rewriter) {
  SmallVector<Value> newPosInCtrlOperands;
  SmallVector<Type> newPosOutCtrlResultTypes;
  SmallVector<Value> newNegInCtrlOperands;
  SmallVector<Type> newNegOutCtrlResultTypes;

  auto posCtrlInQubitsOfOp = op.getPosCtrlInQubits();
  auto negCtrlInQubitsOfOp = op.getNegCtrlInQubits();
  for (const auto& qubitIn : op.getAllCtrlInQubits()) {
    const unsigned int qubitIndex = qcp->qubitToIndex.at(qubitIn);
    if (indicesToRemove.contains(qubitIndex)) {
      rewriter.replaceAllUsesWith(op.getCorrespondingOutput(qubitIn), qubitIn);
    } else if (auto it = llvm::find(posCtrlInQubitsOfOp, qubitIn);
               it != posCtrlInQubitsOfOp.end()) {
      newPosInCtrlOperands.push_back(qubitIn);
      newPosOutCtrlResultTypes.push_back(
          op.getCorrespondingOutput(qubitIn).getType());
    } else if (auto nit = llvm::find(negCtrlInQubitsOfOp, qubitIn);
               nit != negCtrlInQubitsOfOp.end()) {
      newNegInCtrlOperands.push_back(qubitIn);
      newNegOutCtrlResultTypes.push_back(
          op.getCorrespondingOutput(qubitIn).getType());
    }
  }

  const auto qubitType = QubitType::get(rewriter.getContext());
  auto inQubits = op.getInQubits();
  auto staticParams = DenseF64ArrayAttr{};
  if (op.getStaticParams().has_value()) {
    staticParams = DenseF64ArrayAttr::get(rewriter.getContext(),
                                          op.getStaticParams().value());
  }
  const auto opName = op.getIdentifier().str();
  const auto opType = qc::opTypeFromString(opName);
  UnitaryInterface newOp;
  switch (opType) {
    CREATE_OP_CASE(I)
    CREATE_OP_CASE(H)
    CREATE_OP_CASE(X)
    CREATE_OP_CASE(Y)
    CREATE_OP_CASE(Z)
    CREATE_OP_CASE(S)
    CREATE_OP_CASE(Sdg)
    CREATE_OP_CASE(T)
    CREATE_OP_CASE(Tdg)
    CREATE_OP_CASE(V)
    CREATE_OP_CASE(Vdg)
    CREATE_OP_CASE(U)
    CREATE_OP_CASE(U2)
    CREATE_OP_CASE(P)
    CREATE_OP_CASE(SX)
    CREATE_OP_CASE(SXdg)
    CREATE_OP_CASE(RX)
    CREATE_OP_CASE(RY)
    CREATE_OP_CASE(RZ)
    CREATE_OP_CASE(SWAP)
    CREATE_OP_CASE(iSWAP)
    CREATE_OP_CASE(iSWAPdg)
    CREATE_OP_CASE(Peres)
    CREATE_OP_CASE(Peresdg)
    CREATE_OP_CASE(DCX)
    CREATE_OP_CASE(ECR)
    CREATE_OP_CASE(RXX)
    CREATE_OP_CASE(RYY)
    CREATE_OP_CASE(RZZ)
    CREATE_OP_CASE(RZX)
    CREATE_OP_CASE(XXminusYY)
    CREATE_OP_CASE(XXplusYY)
  default:
    throw std::runtime_error("Unsupported operation type");
  }

  for (const auto qubit : newOp.getAllInQubits()) {
    rewriter.replaceAllUsesWith(op.getCorrespondingOutput(qubit),
                                newOp.getCorrespondingOutput(qubit));
  }
  rewriter.eraseOp(op);

  return newOp;
}

UnitaryInterface
putIntoBranch(qcpObjects* qcp, UnitaryInterface op,
              const std::set<std::pair<unsigned int, bool>>& conds,
              PatternRewriter& rewriter) {
  std::vector<std::pair<Value, bool>> valConds = {};
  for (auto& [val, index] : qcp->bitToIndex) {
    for (const auto& [indexOfCond, nonInverse] : conds) {
      if (indexOfCond == index) {
        valConds.emplace_back(val, nonInverse);
        break;
      }
    }
  }

  auto* ctx = rewriter.getContext();
  ctx->loadDialect<scf::SCFDialect>();
  ctx->loadDialect<arith::ArithDialect>();
  auto cond = valConds.at(0).first;
  rewriter.setInsertionPoint(op);
  // Negate bit if necessary
  if (!valConds.at(0).second) { // Build logical NOT: !cond == xor cond, true
    cond = rewriter.create<arith::XOrIOp>(op.getLoc(), valConds.at(0).first,
                                          qcp->trueValue);
  }

  for (auto it = std::next(valConds.begin()); it != valConds.end(); ++it) {
    auto currentCond = it->first;
    rewriter.setInsertionPoint(op);
    // Negate bit if necessary
    if (!it->second) { // Build logical NOT: !cond == xor cond, true
      currentCond = rewriter.create<arith::XOrIOp>(op.getLoc(), it->first,
                                                   qcp->trueValue);
    }
    cond = rewriter.create<arith::AndIOp>(op.getLoc(), cond, currentCond);
  }

  auto ifOp = rewriter.create<scf::IfOp>(
      op.getLoc(), /*resultTypes=*/op->getResultTypes(), cond,
      /*withElseRegion=*/true);
  rewriter.setInsertionPointToStart(ifOp.thenBlock());
  UnitaryInterface thenClone = cast<UnitaryInterface>(rewriter.clone(*op));
  auto yieldThen =
      rewriter.create<scf::YieldOp>(op.getLoc(), thenClone->getResults());

  rewriter.setInsertionPointToStart(ifOp.elseBlock());
  rewriter.create<scf::YieldOp>(op.getLoc(), thenClone.getAllInQubits());

  rewriter.replaceOp(op, ifOp.getResults());

  for (auto qubit : thenClone.getAllInQubits()) {
    auto newQubit = thenClone.getCorrespondingOutput(qubit);
    qcp->qubitToIndex[newQubit] = qcp->qubitToIndex.at(qubit);
  }

  for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
    Value const yieldedThen = yieldThen.getOperand(i);
    Value const ifResult = ifOp.getResult(i);
    qcp->qubitToIndex[ifResult] = qcp->qubitToIndex.at(yieldedThen);
  }

  return thenClone;
}

/**
 *
 */
WalkResult handleIf(qcpObjects* qcp, scf::IfOp op,
                    std::vector<Operation*>& worklist,
                    const std::vector<unsigned int>& posBitCtrls,
                    const std::vector<unsigned int>& negBitCtrls,
                    PatternRewriter& rewriter) {
  TypedValue<IntegerType> const cond = op.getCondition();
  // TODO: What if we have more than one bit here?
  unsigned int const conditionIndex = qcp->bitToIndex.at(cond);
  if (qcp->ut.isBitAlwaysOne(conditionIndex)) {
    qcp->changed = true;
    return removeIfElseBlock(op, op.thenBlock(), op.elseBlock(), worklist,
                             rewriter);
  }
  if (qcp->ut.isBitAlwaysZero(conditionIndex)) {
    qcp->changed = true;
    return removeIfElseBlock(op, op.elseBlock(), op.thenBlock(), worklist,
                             rewriter);
  }
  auto posBitForThen = posBitCtrls;
  posBitForThen.push_back(conditionIndex);
  auto negBitsForElse = negBitCtrls;
  negBitsForElse.push_back(conditionIndex);

  auto& thenRegion = op.getThenRegion();
  auto& elseRegion = op.getElseRegion();
  std::vector<Operation*> thenWorklist;
  std::vector<Operation*> elseWorklist;

  for (Block& block : thenRegion) {
    for (Operation& operation : block) {
      thenWorklist.push_back(&operation);
      std::ranges::replace(worklist, &operation,
                           static_cast<Operation*>(nullptr));
    }
  }
  for (Block& block : elseRegion) {
    for (Operation& operation : block) {
      elseWorklist.push_back(&operation);
      std::ranges::replace(worklist, &operation,
                           static_cast<Operation*>(nullptr));
    }
  }

  const auto thenResult = iterateThroughWorklist(rewriter, thenWorklist, qcp,
                                                 posBitForThen, negBitCtrls);

  auto* lastThenOp = op.getThenRegion().front().getTerminator();
  auto thenYield = cast<scf::YieldOp>(lastThenOp);
  const auto thenYieldedValues = thenYield.getResults();
  const auto ifResults = op.getResults();

  if (thenResult.failed()) {
    return thenResult;
  }

  for (unsigned int i = 0; i < thenYieldedValues.size(); ++i) {
    qcp->qubitToIndex[ifResults[i]] =
        qcp->qubitToIndex.at(thenYieldedValues[i]);
  }

  if (!elseWorklist.empty()) {
    auto elseResult = iterateThroughWorklist(rewriter, elseWorklist, qcp,
                                             posBitCtrls, negBitsForElse);

    if (elseResult.failed()) {
      return elseResult;
    }

    auto* elseTerm = op.getElseRegion().front().getTerminator();
    auto elseYield = cast<scf::YieldOp>(elseTerm);
    const auto elseYieldedValues = elseYield.getResults();
    for (unsigned int i = 0; i < elseYieldedValues.size(); ++i) {
      if (const auto indexOfElseValue =
              qcp->qubitToIndex.at(elseYieldedValues[i]);
          qcp->qubitToIndex[ifResults[i]] != indexOfElseValue) {
        throw std::domain_error(
            "Yield of else-Block yields different qubits than yield of "
            "then-BLock. Not supported by constant propagation.");
      }
      qcp->qubitToIndex.erase(elseYieldedValues[i]);
    }
  }

  if (llvm::isa<scf::YieldOp>(op.thenBlock()->front()) &&
      (elseWorklist.empty() ||
       llvm::isa<scf::YieldOp>(op.elseBlock()->front()))) {
    // Remove if-else
    qcp->changed = true;
    scf::YieldOp const yieldOp =
        cast<scf::YieldOp>(op.thenBlock()->getTerminator());
    for (auto [result, yielded] :
         llvm::zip(op.getResults(), yieldOp->getOperands())) {
      result.replaceAllUsesWith(yielded);
    }
    for (Operation const& operation : *op.thenBlock()) {
      std::ranges::replace(worklist, &operation,
                           static_cast<Operation*>(nullptr));
    }
    for (Operation const& operation : *op.elseBlock()) {
      std::ranges::replace(worklist, &operation,
                           static_cast<Operation*>(nullptr));
    }
    rewriter.eraseOp(op);
  }

  return WalkResult::advance();
}

/**
 * Add new Qubit to the UnionTable
 */
WalkResult handleQubitAlloc(qcpObjects* qcp, const AllocQubitOp op) {
  for (auto res : op->getOpResults()) {
    unsigned int const newQubitIndex = qcp->ut.propagateQubitAlloc();
    qcp->qubitToIndex[res] = newQubitIndex;
  }
  return WalkResult::advance();
}

/**
 * Add allocated memref into memrefToQubitIndex
 */
WalkResult handleAlloc(qcpObjects* qcp, const memref::AllocOp op) {
  for (auto res : op->getOpResults()) {
    auto shape = cast<MemRefType>(res.getType()).getShape();
    if (shape.size() > 1) {
      throw std::logic_error("Cannot handle memref.alloc dimension higher than "
                             "1 in constant propagation (is " +
                             std::to_string(shape.size()) + ").");
    }
    auto elementTypeOfMemref = cast<MemRefType>(res.getType())
                                   .getElementType()
                                   .getAbstractType()
                                   .getName()
                                   .str();
    if (elementTypeOfMemref != "mqtopt.Qubit") {
      throw std::logic_error("Cannot handle memref.alloc on type " +
                             elementTypeOfMemref +
                             " during constant propagation.");
    }
    unsigned int const numberOfQubits = shape.vec().at(0);
    qcp->memrefToQubitIndex[res] = std::vector<unsigned int>(numberOfQubits);
    for (unsigned int i = 0; i < numberOfQubits; ++i) {
      unsigned int const qubitIndex = qcp->ut.propagateQubitAlloc();
      qcp->memrefToQubitIndex[res].at(i) = qubitIndex;
    }
  }
  return WalkResult::advance();
}

/**
 * Add allocated memref into memrefToBitIndex
 */
WalkResult handleAlloca(qcpObjects* qcp, const memref::AllocaOp op) {
  for (auto res : op->getOpResults()) {
    auto shape = cast<MemRefType>(res.getType()).getShape();
    if (shape.size() > 1) {
      throw std::logic_error(
          "Cannot handle memref.alloca dimension higher than "
          "1 in constant propagation (is " +
          std::to_string(shape.size()) + ").");
    }
    auto elementTypeOfMemref = cast<MemRefType>(res.getType())
                                   .getElementType()
                                   .getAbstractType()
                                   .getName()
                                   .str();
    if (elementTypeOfMemref != "builtin.integer") {
      throw std::logic_error("Cannot handle memref.alloc on type " +
                             elementTypeOfMemref +
                             " during constant propagation.");
    }
    unsigned int const numberOfBits = shape.vec().at(0);
    qcp->memrefToBitIndex[res] = std::vector<unsigned int>(numberOfBits);
    for (unsigned int i = 0; i < numberOfBits; ++i) {
      unsigned int const bitIndex = qcp->ut.propagateBitDef(false);
      qcp->memrefToBitIndex[res].at(i) = bitIndex;
    }
  }
  return WalkResult::advance();
}

/**
 * Retrieve qubit from map and save in qubitToIndex
 */
WalkResult handleLoad(qcpObjects* qcp, memref::LoadOp op) {
  for (auto res : op->getOpResults()) {
    std::string const abstractTypeOfMemref =
        res.getType().getAbstractType().getName().str();
    if (abstractTypeOfMemref != "mqtopt.Qubit") {
      throw std::logic_error("Cannot handle memref.load on type " +
                             abstractTypeOfMemref +
                             " during constant propagation.");
    }
    std::vector<unsigned int> const qubitIndicesOfThisMemref =
        qcp->memrefToQubitIndex.at(op.getMemref());
    auto const calledIndices = op.getIndices();
    if (calledIndices.size() > 1) {
      throw std::logic_error("Cannot handle memref.load on multiple indices (" +
                             std::to_string(calledIndices.size()) +
                             " currently) during constant propagation.");
    }
    int64_t const indexValue = qcp->integerValues.at(calledIndices.front());
    unsigned int const qubitIndex = qubitIndicesOfThisMemref.at(indexValue);
    qcp->qubitToIndex[res] = qubitIndex;
  }
  return WalkResult::advance();
}

/**
 * Save index from stored qubit in respective memrefToQubit spot
 */
WalkResult handleStore(qcpObjects* qcp, memref::StoreOp op,
                       const std::vector<unsigned int>& posBitCtrls,
                       const std::vector<unsigned int>& negBitCtrls) {
  if (!posBitCtrls.empty() || !negBitCtrls.empty()) {
    throw std::logic_error("Cannot handle store operation in conditional "
                           "branches during constant propagation.");
  }
  std::string const abstractTypeOfMemref =
      op.getValue().getType().getAbstractType().getName().str();
  if (abstractTypeOfMemref != "mqtopt.Qubit" &&
      abstractTypeOfMemref != "builtin.integer") {
    throw std::logic_error("Cannot handle memref.load on type " +
                           abstractTypeOfMemref +
                           " during constant propagation.");
  }

  Value const storedValue = op.getValue();
  Value const memref = op.getMemref();
  auto const calledIndices = op.getIndices();
  if (calledIndices.size() > 1) {
    throw std::logic_error("Cannot handle memref.load on multiple indices (" +
                           std::to_string(calledIndices.size()) +
                           " currently) during constant propagation.");
  }
  int64_t const indexValue = qcp->integerValues.at(calledIndices.front());
  if (abstractTypeOfMemref == "mqtopt.Qubit") {
    qcp->memrefToQubitIndex[memref].at(indexValue) =
        qcp->qubitToIndex.at(storedValue);
  } else {
    qcp->memrefToBitIndex[memref].at(indexValue) =
        qcp->bitToIndex.at(storedValue);
  }
  return WalkResult::advance();
}

/**
 * Add constant value to qcp
 */
WalkResult handleConstant(qcpObjects* qcp, arith::ConstantOp op,
                          const std::vector<unsigned int>& posBitCtrls,
                          const std::vector<unsigned int>& negBitCtrls) {
  if (!posBitCtrls.empty() || !negBitCtrls.empty()) {
    throw std::logic_error("Cannot handle load operation in conditional "
                           "branches during constant propagation.");
  }

  Value const res = op.getResult();
  auto attr = op.getValue();
  if (const auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    const auto v = intAttr.getInt();
    qcp->integerValues[res] = v;
  }
  if (const auto doubleAttr = dyn_cast<FloatAttr>(attr)) {
    const double v = doubleAttr.getValueAsDouble();
    qcp->doubleValues[res] = v;
  }
  return WalkResult::advance();
}

/**
 * @brief Propagate the unitary.
 */
WalkResult handleUnitary(qcpObjects* qcp, UnitaryInterface op,
                         const std::vector<unsigned int>& posBitCtrls,
                         const std::vector<unsigned int>& negBitCtrls,
                         PatternRewriter& rewriter) {
  std::vector<unsigned int> targetQubitIndices = {};
  std::vector<unsigned int> posCtrlQubitIndices = {};
  std::vector<unsigned int> negCtrlQubitIndices = {};
  std::vector<unsigned int> posBitCtrlsHere = posBitCtrls;
  std::vector<unsigned int> negBitCtrlsHere = negBitCtrls;
  std::vector<double> params = {};
  std::set<std::pair<unsigned int, bool>> bitDependenceToAdd = {};
  for (auto targetQubit : op.getInQubits()) {
    targetQubitIndices.push_back(qcp->qubitToIndex.at(targetQubit));
  }
  for (auto posCtrlQubit : op.getPosCtrlInQubits()) {
    posCtrlQubitIndices.push_back(qcp->qubitToIndex.at(posCtrlQubit));
  }
  for (auto negCtrlQubit : op.getNegCtrlInQubits()) {
    negCtrlQubitIndices.push_back(qcp->qubitToIndex.at(negCtrlQubit));
  }
  for (auto param : op.getParams()) {
    params.push_back(qcp->doubleValues.at(param));
  }
  auto staticParams = op.getStaticParams();
  if (staticParams.has_value()) {
    for (auto param : staticParams.value()) {
      params.push_back(param);
    }
  }
  // Check if parts of or the whole gate are superfluous
  if (op.isControlled()) {
    std::pair<std::set<unsigned int>, std::set<unsigned int>> const
        superfluous = qcp::RewriteChecker::getSuperfluousControls(
            qcp->ut, targetQubitIndices, posCtrlQubitIndices,
            negCtrlQubitIndices, posBitCtrls, negBitCtrls);
    bool const areThereSatisfiableCombinations =
        qcp::RewriteChecker::areThereSatisfiableCombinations(
            qcp->ut, posCtrlQubitIndices, negCtrlQubitIndices, posBitCtrls,
            negBitCtrls);
    if (superfluous.first.contains(targetQubitIndices.at(0)) ||
        !areThereSatisfiableCombinations) {
      qcp->changed = true;
      return removeGate(op, rewriter);
    }

    std::set<unsigned int> ctrlQubitsToRemove = superfluous.first;
    std::set<unsigned int> remainingPosCtrlQubits = {};
    std::set<unsigned int> remainingNegCtrlQubits = {};
    std::ranges::set_difference(
        posCtrlQubitIndices, ctrlQubitsToRemove,
        std::inserter(remainingPosCtrlQubits, remainingPosCtrlQubits.begin()));
    std::ranges::set_difference(
        negCtrlQubitIndices, ctrlQubitsToRemove,
        std::inserter(remainingNegCtrlQubits, remainingNegCtrlQubits.begin()));

    // Find all antecedents to remove
    const std::set setPosBitCtrl(posBitCtrls.begin(), posBitCtrls.end());
    const std::set setNegBitCtrl(negBitCtrls.begin(), negBitCtrls.end());
    for (const unsigned int posCtrlQubit : remainingPosCtrlQubits) {
      const std::pair<std::set<unsigned int>, std::set<unsigned int>>
          antecedents = qcp::RewriteChecker::getAntecedentsOfQubit(
              qcp->ut, posCtrlQubit, false, remainingPosCtrlQubits,
              remainingNegCtrlQubits, setPosBitCtrl, setNegBitCtrl);
      if (!antecedents.first.empty() || !antecedents.second.empty()) {
        ctrlQubitsToRemove.insert(posCtrlQubit);
      }
    }
    for (const unsigned int negCtrlQubit : remainingNegCtrlQubits) {
      const std::pair<std::set<unsigned int>, std::set<unsigned int>>
          antecedents = qcp::RewriteChecker::getAntecedentsOfQubit(
              qcp->ut, negCtrlQubit, true, remainingPosCtrlQubits,
              remainingNegCtrlQubits, setPosBitCtrl, setNegBitCtrl);
      if (!antecedents.first.empty() || !antecedents.second.empty()) {
        ctrlQubitsToRemove.insert(negCtrlQubit);
      }
    }
    // Check whether to replace by bit
    std::ranges::set_difference(
        posCtrlQubitIndices, ctrlQubitsToRemove,
        std::inserter(remainingPosCtrlQubits, remainingPosCtrlQubits.begin()));
    std::ranges::set_difference(
        negCtrlQubitIndices, ctrlQubitsToRemove,
        std::inserter(remainingNegCtrlQubits, remainingNegCtrlQubits.begin()));
    for (const auto posCtrlQubit : remainingPosCtrlQubits) {
      const std::optional<std::pair<unsigned int, bool>> optionalImplBit =
          qcp::RewriteChecker::getEquivalentBit(qcp->ut, posCtrlQubit);
      if (optionalImplBit.has_value()) {
        std::pair<unsigned int, bool> const implyingBit = *optionalImplBit;
        // The return says whether the bit is equivalent to the qubit (true) or
        // its inverse (false). But we now want to now whether the dependence is
        // pos or neg
        bitDependenceToAdd.insert(implyingBit);
        ctrlQubitsToRemove.insert(posCtrlQubit);
      }
    }
    for (const auto negCtrlQubit : remainingNegCtrlQubits) {
      const std::optional<std::pair<unsigned int, bool>> optionalImplBit =
          qcp::RewriteChecker::getEquivalentBit(qcp->ut, negCtrlQubit);
      if (optionalImplBit.has_value()) {
        std::pair<unsigned int, bool> implyingBit = *optionalImplBit;
        implyingBit.second = !implyingBit.second;
        bitDependenceToAdd.insert(implyingBit);
        ctrlQubitsToRemove.insert(negCtrlQubit);
      }
    }
    for (const auto& [bitIndex, value] : bitDependenceToAdd) {
      if (value) {
        posBitCtrlsHere.push_back(bitIndex);
      } else {
        negBitCtrlsHere.push_back(bitIndex);
      }
    }

    if (removeDiagonalGate(qcp, op, targetQubitIndices, posCtrlQubitIndices,
                           negCtrlQubitIndices)) {
      qcp->changed = true;
      return removeGate(op, rewriter);
    }

    if (!ctrlQubitsToRemove.empty()) {
      qcp->changed = true;
      // Remove superfluous quantum controls
      op = removeCtrls(qcp, op, ctrlQubitsToRemove, rewriter);
      posCtrlQubitIndices = {};
      negCtrlQubitIndices = {};
      for (auto posCtrlQubit : op.getPosCtrlInQubits()) {
        posCtrlQubitIndices.push_back(qcp->qubitToIndex.at(posCtrlQubit));
      }
      for (auto negCtrlQubit : op.getNegCtrlInQubits()) {
        negCtrlQubitIndices.push_back(qcp->qubitToIndex.at(negCtrlQubit));
      }
    }

    if (!bitDependenceToAdd.empty()) {
      qcp->changed = true;
      op = putIntoBranch(qcp, op, bitDependenceToAdd, rewriter);
    }
  }

  if (removeDiagonalGate(qcp, op, targetQubitIndices, posCtrlQubitIndices,
                         negCtrlQubitIndices)) {
    qcp->changed = true;
    return removeGate(op, rewriter);
  }

  const auto opName = op.getIdentifier().str();
  const auto opType = qc::opTypeFromString(opName);
  qcp->ut.propagateGate(opType, targetQubitIndices, posCtrlQubitIndices,
                        negCtrlQubitIndices, posBitCtrlsHere, negBitCtrlsHere,
                        params);
  if (!bitDependenceToAdd.empty()) {
    return WalkResult::advance();
  }
  for (auto qubit : op.getAllInQubits()) {
    auto newQubit = op.getCorrespondingOutput(qubit);
    qcp->qubitToIndex[newQubit] = qcp->qubitToIndex.at(qubit);
    if (posBitCtrls.empty() && negBitCtrls.empty()) {
      qcp->qubitToIndex.erase(qubit);
    }
  }
  return WalkResult::advance();
}

/**
 * @brief Propagate the measurement.
 */
WalkResult handleReset(qcpObjects* qcp, ResetOp op,
                       const std::vector<unsigned int>& posBitCtrls,
                       const std::vector<unsigned int>& negBitCtrls) {
  if (!posBitCtrls.empty() || !negBitCtrls.empty()) {
    throw std::logic_error("Cannot handle store operation in conditional "
                           "branches during constant propagation.");
  }
  const auto qubit = op.getInQubit();
  const auto newQubit = op.getOutQubit();
  qcp->qubitToIndex[newQubit] = qcp->qubitToIndex.at(qubit);
  // TODO: Reset depending on pos/neg Bit ctrls
  qcp->ut.propagateReset(qcp->qubitToIndex.at(qubit));
  if (posBitCtrls.empty() && negBitCtrls.empty()) {
    qcp->qubitToIndex.erase(qubit);
  }
  return WalkResult::advance();
}

/**
 * @brief Propagate the measurement.
 */
WalkResult handleMeasure(qcpObjects* qcp, MeasureOp op,
                         const std::vector<unsigned int>& posBitCtrls,
                         const std::vector<unsigned int>& negBitCtrls) {
  const auto qubit = op.getInQubit();
  const auto newQubit = op.getOutQubit();
  const auto outBit = op.getOutBit();
  unsigned int bitIndex = 0;
  qcp->qubitToIndex[newQubit] = qcp->qubitToIndex.at(qubit);
  if (qcp->bitToIndex.contains(outBit)) {
    bitIndex = qcp->bitToIndex.at(outBit);
  } else {
    bitIndex = qcp->ut.propagateBitDef(false);
    qcp->bitToIndex[outBit] = bitIndex;
  }
  // TODO: Measurement depending on classical bits
  qcp->ut.propagateMeasurement(qcp->qubitToIndex.at(qubit), bitIndex);
  if (posBitCtrls.empty() && negBitCtrls.empty()) {
    qcp->qubitToIndex.erase(qubit);
  }
  return WalkResult::advance();
}

LogicalResult
iterateThroughWorklist(PatternRewriter& rewriter,
                       std::vector<Operation*>& worklist, qcpObjects* qcp,
                       const std::vector<unsigned int>& posBitCtrls,
                       const std::vector<unsigned int>& negBitCtrls) {
  /// Iterate work-list.
  for (Operation* curr : worklist) {
    if (curr == nullptr) {
      continue; // Skip erased ops.
    }
    auto n = curr->getName().stripDialect().str();

    rewriter.setInsertionPoint(curr);

    const auto res =
        TypeSwitch<Operation*, WalkResult>(curr)
            /// mqtopt Dialect
            .Case<UnitaryInterface>([&](const UnitaryInterface op) {
              return handleUnitary(qcp, op, posBitCtrls, negBitCtrls, rewriter);
            })
            .Case<ResetOp>([&](const ResetOp op) {
              return handleReset(qcp, op, posBitCtrls, negBitCtrls);
            })
            .Case<MeasureOp>([&](const MeasureOp op) {
              return handleMeasure(qcp, op, posBitCtrls, negBitCtrls);
            })
            .Case<AllocQubitOp>([&](const AllocQubitOp op) {
              return handleQubitAlloc(qcp, op);
            })
            .Case<DeallocQubitOp>([&]([[maybe_unused]] DeallocQubitOp op) {
              return WalkResult::advance();
            })
            /// built-in Dialect
            .Case<ModuleOp>([&]([[maybe_unused]] ModuleOp op) {
              return WalkResult::advance();
            })
            /// memref Dialect
            .Case<memref::AllocOp>(
                [&](const memref::AllocOp op) { return handleAlloc(qcp, op); })
            .Case<memref::AllocaOp>([&](const memref::AllocaOp op) {
              return handleAlloca(qcp, op);
            })
            .Case<memref::DeallocOp>(
                [&]([[maybe_unused]] const memref::DeallocOp op) {
                  return WalkResult::advance();
                })
            .Case<memref::LoadOp>(
                [&](const memref::LoadOp op) { return handleLoad(qcp, op); })
            .Case<memref::StoreOp>([&](const memref::StoreOp op) {
              return handleStore(qcp, op, posBitCtrls, negBitCtrls);
            })
            // arith dialect
            .Case<arith::ConstantOp>([&](const arith::ConstantOp op) {
              return handleConstant(qcp, op, posBitCtrls, negBitCtrls);
            })
            /// func Dialect
            .Case<func::FuncOp>([&](const func::FuncOp op) {
              return handleFunc(qcp, op, rewriter);
            })
            .Case<func::ReturnOp>([&]([[maybe_unused]] func::ReturnOp op) {
              return WalkResult::advance();
            })
            /// scf Dialect
            .Case<scf::ForOp>([&](scf::ForOp) { return handleFor(); })
            .Case<scf::IfOp>([&](const scf::IfOp op) {
              return handleIf(qcp, op, worklist, posBitCtrls, negBitCtrls,
                              rewriter);
            })
            .Case<scf::YieldOp>([&]([[maybe_unused]] scf::YieldOp op) {
              return WalkResult::advance();
            })
            /// Skip the rest.
            .Default([](auto) {
              throw std::runtime_error("Unsupported operation");
              return WalkResult::interrupt();
            });

    if (res.wasInterrupted()) {
      return failure();
    }
  }
  return success();
}

/**
 * @brief Do quantum constant propagation.
 *
 * @details
 * Collects all functions marked with the 'entry_point' attribute, builds
 * a preorder worklist of their operations, and processes that list.
 *
 * @note
 * We consciously avoid MLIR pattern drivers: Idiomatic MLIR
 * transformation patterns are independent and order-agnostic. Since we
 * require state-sharing between patterns for the transformation we
 * violate this assumption. Essentially this is also the reason why we
 * can't utilize MLIR's `applyPatternsGreedily` function. Moreover, we
 * require pre-order traversal which current drivers of MLIR don't
 * support. However, even if such a driver would exist, it would probably
 * not return logical results which we require for error-handling
 * (similarly to `walkAndApplyPatterns`). Consequently, a custom driver
 * would be required in any case, which adds unnecessary code to maintain.
 */
bool applyQCP(ModuleOp module, MLIRContext* ctx) {
  PatternRewriter rewriter(ctx);

  /// Prepare work-list.
  std::vector<Operation*> worklist;

  for (const auto func : module.getOps<func::FuncOp>()) {

    if (!isEntryPoint(func)) {
      continue; // Ignore non entry_point functions for now.
    }
    func->walk<WalkOrder::PreOrder>(
        [&](Operation* op) { worklist.push_back(op); });
    // auto n = func->getName().stripDialect().str();
  }

  const auto ut = qcp::UnionTable(8, 8);
  qcpObjects qcp = {
      .ut = ut,
      .qubitToIndex = llvm::DenseMap<Value, unsigned int>(),
      .memrefToQubitIndex = llvm::DenseMap<Value, std::vector<unsigned int>>(),
      .memrefToBitIndex = llvm::DenseMap<Value, std::vector<unsigned int>>(),
      .integerValues = llvm::DenseMap<Value, int64_t>(),
      .doubleValues = llvm::DenseMap<Value, double>(),
      .bitToIndex = llvm::DenseMap<Value, unsigned int>()};

  auto res = iterateThroughWorklist(rewriter, worklist, &qcp, {}, {});

  if (res.failed()) {
    throw std::runtime_error("Failure during QCP application.");
  }

  return qcp.changed;
}

LogicalResult route(ModuleOp module, MLIRContext* ctx) {
  PatternRewriter rewriter(ctx);

  /// Prepare work-list.
  std::vector<Operation*> worklist;

  for (const auto func : module.getOps<func::FuncOp>()) {

    if (!isEntryPoint(func)) {
      continue; // Ignore non entry_point functions for now.
    }
    func->walk<WalkOrder::PreOrder>(
        [&](Operation* op) { worklist.push_back(op); });
    // auto n = func->getName().stripDialect().str();
  }

  const auto ut = qcp::UnionTable(8, 8);
  qcpObjects qcp = {
      .ut = ut,
      .qubitToIndex = llvm::DenseMap<Value, unsigned int>(),
      .memrefToQubitIndex = llvm::DenseMap<Value, std::vector<unsigned int>>(),
      .memrefToBitIndex = llvm::DenseMap<Value, std::vector<unsigned int>>(),
      .integerValues = llvm::DenseMap<Value, int64_t>(),
      .doubleValues = llvm::DenseMap<Value, double>(),
      .bitToIndex = llvm::DenseMap<Value, unsigned int>()};

  return iterateThroughWorklist(rewriter, worklist, &qcp, {}, {});
}

/**
 * @brief This pass ensures
 */
struct QuantumConstantPropagationPass final
    : impl::QuantumConstantPropagationPassBase<QuantumConstantPropagationPass> {
  using QuantumConstantPropagationPassBase::QuantumConstantPropagationPassBase;

  void runOnOperation() override {
    if (failed(route(getOperation(), &getContext()))) {
      signalPassFailure();
    }
  }
};

/**
 * This method moves all measurements as far to the front as possible, in order
 * to execute QCP more efficiently.
 */
bool moveMeasurementsToFront(ModuleOp module, MLIRContext* ctx) {
  bool changed = false;
  PatternRewriter rewriter(ctx);
  module.walk([&](MeasureOp op) {
    mlir::Operation* previousInstruction = op.getInQubit().getDefiningOp();
    if (op->getPrevNode() != previousInstruction) {
      rewriter.moveOpAfter(op, previousInstruction);
      changed = true;
    }
  });

  return changed;
}

} // namespace
} // namespace mqt::ir::opt
