/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"

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

namespace {
using namespace mlir;

/**
 *
 */
WalkResult handleFunc([[maybe_unused]] func::FuncOp op) {
  // TODO: HandleFuncs, throw not supported if it is not the entry point?
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
 *
 */
WalkResult handleFor(scf::ForOp op) {
  // TODO: Throw exception that for handling is not yet supported
  return WalkResult::advance();
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
  return WalkResult::advance();
}

/**
 * @brief Propagte the unitary.
 */
WalkResult handleUnitary(UnitaryInterface op, PatternRewriter& rewriter) {
  // TODO: Propagate the unitary
  return WalkResult::advance();
}

/**
 * @brief Propagate the measurement.
 */
WalkResult handleReset(ResetOp op, PatternRewriter& rewriter) {
  // TODO: Propagate the reset
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
              return handleUnitary(op, rewriter);
            })
            .Case<QubitOp>([&](QubitOp op) { return handleQubit(op); })
            .Case<ResetOp>(
                [&](ResetOp op) { return handleReset(op, rewriter); })
            .Case<MeasureOp>(
                [&](MeasureOp op) { return handleMeasure(op, rewriter); })
            /// built-in Dialect
            .Case<ModuleOp>([&]([[maybe_unused]] ModuleOp op) {
              return WalkResult::advance();
            })
            /// func Dialect
            .Case<func::FuncOp>([&](func::FuncOp op) { return handleFunc(op); })
            .Case<func::ReturnOp>([&]([[maybe_unused]] func::ReturnOp op) {
              return handleReturn();
            })
            /// scf Dialect
            .Case<scf::ForOp>([&](scf::ForOp op) { return handleFor(op); })
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
