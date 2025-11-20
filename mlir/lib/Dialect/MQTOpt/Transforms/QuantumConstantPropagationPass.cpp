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
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Router.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Scheduler.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Stack.h"

#include <chrono>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <utility>
#include <vector>

namespace mqt::ir::opt {

#define GEN_PASS_DEF_QUANTUMCONSTANTPROPAGATIONPASS
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

namespace {
using namespace mlir;

class Mapper {
public:
  explicit Mapper(std::unique_ptr<Architecture> arch,
                  std::unique_ptr<SchedulerBase> scheduler,
                  std::unique_ptr<RouterBase> router, Pass::Statistic& numSwaps)
      : arch_(std::move(arch)), scheduler_(std::move(scheduler)),
        router_(std::move(router)), numSwaps_(&numSwaps) {}

  /**
   * @returns reference to the stack object.
   */
  [[nodiscard]] LayoutStack<Layout>& stack() { return stack_; }

  /**
   * @returns reference to the history stack object.
   */
  [[nodiscard]] LayoutStack<SmallVector<QubitIndexPair>>& historyStack() {
    return historyStack_;
  }

  /**
   * @returns reference to architecture object.
   */
  [[nodiscard]] Architecture& arch() const { return *arch_; }

private:
  std::unique_ptr<Architecture> arch_;
  std::unique_ptr<SchedulerBase> scheduler_;
  std::unique_ptr<RouterBase> router_;

  LayoutStack<Layout> stack_{};
  LayoutStack<SmallVector<QubitIndexPair>> historyStack_{};

  Pass::Statistic* numSwaps_;
};

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

  module.walk<WalkOrder::PreOrder>([&](Operation* op) {
    auto name = op->getName().stripDialect().str();
    v.push_back(name);
    worklist.push_back(op);
  });

  for (const auto func : module.getOps<func::FuncOp>()) {

    if (!isEntryPoint(func)) {
      continue; // Ignore non entry_point functions for now.
    }
    func->walk<WalkOrder::PreOrder>(
        [&](Operation* op) { worklist.push_back(op); });
    // auto n = func->getName().stripDialect().str();
  }

  int test = 2;

  /// Iterate work-list.
  for (Operation* curr : worklist) {
    if (curr == nullptr) {
      continue; // Skip erased ops.
    }

    // v.push_back(test);
    test++;
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
