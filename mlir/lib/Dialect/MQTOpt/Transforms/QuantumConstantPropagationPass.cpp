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

#include <algorithm>
#include <chrono>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <stdexcept>
#include <string>
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
};

namespace {
using namespace mlir;

LogicalResult
iterateThroughWorklist(PatternRewriter& rewriter,
                       std::vector<Operation*>& worklist, qcpObjects* qcp,
                       const std::vector<unsigned int>& posBitCtrls,
                       const std::vector<unsigned int>& negBitCtrls);

/**
 *
 */
WalkResult handleFunc(const func::FuncOp op) {
  if (!isEntryPoint(op)) {
    throw std::domain_error(
        "Constant propagation does not support nested functions.");
  }
  return WalkResult::advance();
}

/**
 * For-loops are not yet part of the constant propagation.
 */
WalkResult handleFor() {
  throw std::domain_error(
      "Handling of for-loops not yet supported by constant propagation.");
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
    qcp->qubitToIndex[ifResults[i]] = qcp->qubitToIndex[thenYieldedValues[i]];
    qcp->qubitToIndex.erase(thenYieldedValues[i]);
  }

  if (!elseWorklist.empty()) {
    auto* elseTerm = op.getElseRegion().front().getTerminator();
    auto elseYield = cast<scf::YieldOp>(elseTerm);
    auto elseYieldedValues = elseYield.getResults();
    for (unsigned int i = 0; i < elseYieldedValues.size(); ++i) {
      auto indexOfElseValue = qcp->qubitToIndex[elseYieldedValues[i]];
      if (qcp->qubitToIndex[ifResults[i]] != indexOfElseValue) {
        throw std::domain_error(
            "Yield of else-Block yields different qubits than yield of "
            "then-BLock. Not supported by constant propagation.");
      }
      qcp->qubitToIndex.erase(elseYieldedValues[i]);
    }
  }

  return iterateThroughWorklist(rewriter, elseWorklist, qcp, posBitCtrls,
                                negBitsForElse);
}

/**
 *
 */
WalkResult handleQubit(QubitOp op) {
  // TODO: Check usages for QubitOp and handle
  throw std::domain_error("handleQubit op not yet implemented.");
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
    int const indexValue = qcp->integerValues.at(calledIndices.front());
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
  if (abstractTypeOfMemref != "mqtopt.Qubit" && abstractTypeOfMemref != "builtin.integer") {
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
  int const indexValue = qcp->integerValues.at(calledIndices.front());
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
  std::vector<double> params = {};
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
  const auto opName = op.getIdentifier().str();
  const auto opType = qc::opTypeFromString(opName);
  // TODO: Check if gate should be removed
  qcp->ut.propagateGate(opType, targetQubitIndices, posCtrlQubitIndices,
                        negCtrlQubitIndices, posBitCtrls, negBitCtrls, params);
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
                       const std::vector<unsigned int>& negBitCtrls,
                       PatternRewriter& rewriter) {
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
                         const std::vector<unsigned int>& negBitCtrls,
                         PatternRewriter& rewriter) {
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
            .Case<UnitaryInterface>([&](UnitaryInterface op) {
              return handleUnitary(qcp, op, posBitCtrls, negBitCtrls, rewriter);
            })
            .Case<QubitOp>([&](QubitOp op) { return handleQubit(op); })
            .Case<ResetOp>([&](ResetOp op) {
              return handleReset(qcp, op, posBitCtrls, negBitCtrls, rewriter);
            })
            .Case<MeasureOp>([&](MeasureOp op) {
              return handleMeasure(qcp, op, posBitCtrls, negBitCtrls, rewriter);
            })
            .Case<AllocQubitOp>(
                [&](AllocQubitOp op) { return handleQubitAlloc(qcp, op); })
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
            .Case<func::FuncOp>([&](func::FuncOp op) { return handleFunc(op); })
            .Case<func::ReturnOp>([&]([[maybe_unused]] func::ReturnOp op) {
              return WalkResult::advance();
            })
            /// scf Dialect
            .Case<scf::ForOp>([&](scf::ForOp op) { return handleFor(); })
            .Case<scf::IfOp>([&](scf::IfOp op) {
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
 * Collects all functions marked with the 'entry_point' attribute, builds a
 * preorder worklist of their operations, and processes that list.
 *
 * @note
 * We consciously avoid MLIR pattern drivers: Idiomatic MLIR transformation
 * patterns are independent and order-agnostic. Since we require state-sharing
 * between patterns for the transformation we violate this assumption.
 * Essentially this is also the reason why we can't utilize MLIR's
 * `applyPatternsGreedily` function. Moreover, we require pre-order traversal
 * which current drivers of MLIR don't support. However, even if such a driver
 * would exist, it would probably not return logical results which we require
 * for error-handling (similarly to `walkAndApplyPatterns`). Consequently, a
 * custom driver would be required in any case, which adds unnecessary code to
 * maintain.
 */
LogicalResult route(ModuleOp module, MLIRContext* ctx,
                    std::vector<std::string>& v) {
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
  qcpObjects qcp = {ut,
                    llvm::DenseMap<Value, unsigned int>(),
                    llvm::DenseMap<Value, std::vector<unsigned int>>(),
                    llvm::DenseMap<Value, std::vector<unsigned int>>(),
                    llvm::DenseMap<Value, int64_t>(),
                    llvm::DenseMap<Value, double>(),
                    llvm::DenseMap<Value, unsigned int>()};

  return iterateThroughWorklist(rewriter, worklist, &qcp, {}, {});
}

/**
 * @brief This pass ensures
 */
struct QuantumConstantPropagationPass final
    : impl::QuantumConstantPropagationPassBase<QuantumConstantPropagationPass> {
  using QuantumConstantPropagationPassBase::QuantumConstantPropagationPassBase;

  void runOnOperation() override {
    std::vector<std::string> v = getPropagator();

    if (failed(route(getOperation(), &getContext(), v))) {
      signalPassFailure();
    }
  }

private:
  [[nodiscard]] std::vector<std::string> getPropagator() {
    std::vector<std::string> v{"begin"};
    return v;
  }
};

} // namespace
} // namespace mqt::ir::opt
