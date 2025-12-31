/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MQTOpt/Transforms/ConstantPropagation/UnionTable.hpp"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"

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
#include <vector>

namespace mqt::ir::opt {

#define GEN_PASS_DEF_QUANTUMCONSTANTPROPAGATIONPASS
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

struct qcpObjects {
  qcp::UnionTable ut;
  llvm::DenseMap<mlir::Value, unsigned int> qubitToIndex;
  llvm::DenseMap<mlir::Value, std::vector<unsigned int>> memrefToQubitIndex;
  llvm::DenseMap<mlir::Value, int> integerValues;
  llvm::DenseMap<mlir::Value, double> doubleValues;
  std::map<std::string, unsigned int> bitToIndex;
};

namespace {
using namespace mlir;

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
 *
 */
WalkResult handleReturn() {
  // TODO: Handle return
  // As we only support entry point func, do not support that?
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
WalkResult handleIf(scf::IfOp op) {
  // TODO: Handle if (save conditional bits in an additional object?)
  return WalkResult::advance();
}

/**
 * @brief Handle yield/propagate qubit usage.
 */
WalkResult handleYield(scf::YieldOp op, PatternRewriter& rewriter) {
  // TODO: Handle yield, i.e. propagate qubit useage
  return WalkResult::advance();
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
    unsigned int numberOfQubits = shape.vec().at(0);
    qcp->memrefToQubitIndex[res] = std::vector<unsigned int>(numberOfQubits);
    for (unsigned int i = 0; i < numberOfQubits; ++i) {
      unsigned int qubitIndex = qcp->ut.propagateQubitAlloc();
      qcp->memrefToQubitIndex[res].at(i) = qubitIndex;
    }
  }
  return WalkResult::advance();
}

/**
 * Retrieve qubit from map and save in qubitToIndex
 */
WalkResult handleLoad(qcpObjects* qcp, memref::LoadOp op) {
  for (auto res : op->getOpResults()) {
    auto abstractTypeOfMemref = res.getType().getAbstractType().getName().str();
    if (abstractTypeOfMemref != "mqtopt.Qubit") {
      throw std::logic_error("Cannot handle memref.load on type " +
                             abstractTypeOfMemref +
                             " during constant propagation.");
    }
    std::vector<unsigned int> const qubitIndicesOfThisMemref =
        qcp->memrefToQubitIndex.at(op.getMemref());
    auto calledIndices = op.getIndices();
    if (calledIndices.size() > 1) {
      throw std::logic_error("Cannot handle memref.load on multiple indices (" +
                             std::to_string(calledIndices.size()) +
                             " currently) during constant propagation.");
    }
    int indexValue = qcp->integerValues[calledIndices.front()];
    unsigned int qubitIndex = qubitIndicesOfThisMemref.at(indexValue);
    qcp->qubitToIndex[res] = qubitIndex;
  }
  return WalkResult::advance();
}

/**
 * Save index from stored qubit in respective memrefToQubit spot
 */
WalkResult handleStore(qcpObjects* qcp, memref::StoreOp op) {
  mlir::Value const storedValue = op.getValue();
  mlir::Value const memref = op.getMemref();
  auto calledIndices = op.getIndices();
  if (calledIndices.size() > 1) {
    throw std::logic_error("Cannot handle memref.load on multiple indices (" +
                           std::to_string(calledIndices.size()) +
                           " currently) during constant propagation.");
  }
  int const indexValue = qcp->integerValues[calledIndices.front()];
  qcp->memrefToQubitIndex[memref].at(indexValue) =
      qcp->qubitToIndex.at(storedValue);
  return WalkResult::advance();
}

/**
 * Add constant value to qcp
 */
WalkResult handleConstant(qcpObjects* qcp, arith::ConstantOp op) {
  mlir::Value const res = op.getResult();
  mlir::Attribute attr = op.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    int v = intAttr.getInt();
    qcp->integerValues[res] = v;
  }
  if (auto doubleAttr = dyn_cast<FloatAttr>(attr)) {
    double v = doubleAttr.getValueAsDouble();
    qcp->doubleValues[res] = v;
  }
  return WalkResult::advance();
}

/**
 * @brief Propagte the unitary.
 */
WalkResult handleUnitary(qcpObjects* qcp, UnitaryInterface op,
                         PatternRewriter& rewriter) {
  std::vector<unsigned int> targetQubitIndices = {};
  std::vector<unsigned int> posCtrlQubitIndices = {};
  std::vector<unsigned int> negCtrlQubitIndices = {};
  std::vector<double> params = {};
  for (auto targetQubit : op.getInQubits()) {
    targetQubitIndices.push_back(qcp->qubitToIndex[targetQubit]);
  }
  for (auto posCtrlQubit : op.getPosCtrlInQubits()) {
    posCtrlQubitIndices.push_back(qcp->qubitToIndex[posCtrlQubit]);
  }
  for (auto negCtrlQubit : op.getNegCtrlInQubits()) {
    negCtrlQubitIndices.push_back(qcp->qubitToIndex[negCtrlQubit]);
  }
  for (auto qubit : op.getAllInQubits()) {
    auto newQubit = op.getCorrespondingOutput(qubit);
    qcp->qubitToIndex[newQubit] = qcp->qubitToIndex[qubit];
    // TODO: We can only do this if we have no classical dependence
    qcp->qubitToIndex.erase(qubit);
  }
  for (auto param : op.getParams()) {
    params.push_back(qcp->doubleValues[param]);
  }
  auto opName = op.getIdentifier().str();
  auto opType = qc::opTypeFromString(opName);
  // TODO: Bit dependence, check if gate should be removed
  qcp->ut.propagateGate(opType, targetQubitIndices, posCtrlQubitIndices,
                        negCtrlQubitIndices, {}, {}, params);
  return WalkResult::advance();
}

/**
 * @brief Propagate the measurement.
 */
WalkResult handleReset(qcpObjects* qcp, ResetOp op, PatternRewriter& rewriter) {
  auto qubit = op.getInQubit();
  auto newQubit = op.getOutQubit();
  qcp->qubitToIndex[newQubit] = qcp->qubitToIndex[qubit];
  qcp->ut.propagateReset(qcp->qubitToIndex[qubit]);
  // TODO: We can only do this if we have no classical dependence
  qcp->qubitToIndex.erase(qubit);
  return WalkResult::advance();
}

/**
 * @brief Propagate the measurement.
 */
WalkResult handleMeasure(MeasureOp op, PatternRewriter& rewriter) {
  // TODO: Propagate the measurement
  return WalkResult::advance();
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

  auto ut = qcp::UnionTable(8, 8);
  qcpObjects qcp = {ut,
                    llvm::DenseMap<mlir::Value, unsigned int>(),
                    llvm::DenseMap<mlir::Value, std::vector<unsigned int>>(),
                    llvm::DenseMap<mlir::Value, int>(),
                    llvm::DenseMap<mlir::Value, double>(),
                    {}};

  /// Iterate work-list.
  for (Operation* curr : worklist) {
    if (curr == nullptr) {
      continue; // Skip erased ops.
    }
    auto n = curr->getName().stripDialect().str();
    v.push_back(n);

    rewriter.setInsertionPoint(curr);

    // TODO: Handle declaration/initialization of classical bits
    const auto res =
        TypeSwitch<Operation*, WalkResult>(curr)
            /// mqtopt Dialect
            .Case<UnitaryInterface>([&](UnitaryInterface op) {
              return handleUnitary(&qcp, op, rewriter);
            })
            .Case<QubitOp>([&](QubitOp op) { return handleQubit(op); })
            .Case<ResetOp>(
                [&](ResetOp op) { return handleReset(&qcp, op, rewriter); })
            .Case<MeasureOp>(
                [&](MeasureOp op) { return handleMeasure(op, rewriter); })
            .Case<AllocQubitOp>(
                [&](AllocQubitOp op) { return handleQubitAlloc(&qcp, op); })
            .Case<DeallocQubitOp>([&]([[maybe_unused]] DeallocQubitOp op) {
              return WalkResult::advance();
            })
            /// built-in Dialect
            .Case<ModuleOp>([&]([[maybe_unused]] ModuleOp op) {
              return WalkResult::advance();
            })
            /// memref Dialect
            .Case<memref::AllocOp>(
                [&](const memref::AllocOp op) { return handleAlloc(&qcp, op); })
            .Case<memref::DeallocOp>(
                [&]([[maybe_unused]] const memref::DeallocOp op) {
                  return WalkResult::advance();
                })
            .Case<memref::LoadOp>(
                [&](const memref::LoadOp op) { return handleLoad(&qcp, op); })
            .Case<memref::StoreOp>(
                [&](const memref::StoreOp op) { return handleStore(&qcp, op); })
            // arith dialect
            .Case<arith::ConstantOp>([&](const arith::ConstantOp op) {
              return handleConstant(&qcp, op);
            })
            /// func Dialect
            .Case<func::FuncOp>([&](func::FuncOp op) { return handleFunc(op); })
            .Case<func::ReturnOp>([&]([[maybe_unused]] func::ReturnOp op) {
              return handleReturn();
            })
            /// scf Dialect
            .Case<scf::ForOp>([&](scf::ForOp op) { return handleFor(); })
            .Case<scf::IfOp>([&](scf::IfOp op) { return handleIf(op); })
            .Case<scf::YieldOp>(
                [&](scf::YieldOp op) { return handleYield(op, rewriter); })
            /// Skip the rest.
            .Default([](auto) { return WalkResult::skip(); });

    if (res.wasInterrupted()) {
      return failure();
    }
  }

  return failure();
}

/**
 * @brief This pass ensures
 */
struct QuantumConstantPropagationPass final
    : impl::QuantumConstantPropagationPassBase<QuantumConstantPropagationPass> {
  using QuantumConstantPropagationPassBase<
      QuantumConstantPropagationPass>::QuantumConstantPropagationPassBase;

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
