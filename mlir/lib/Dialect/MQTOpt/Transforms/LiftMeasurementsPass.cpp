/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "QuantumConstantPropagationPass.cpp"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <stdexcept>
#include <utility>

namespace mqt::ir::opt {

#define GEN_PASS_DEF_LIFTMEASUREMENTSPASS
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

struct ChangeTrackingListener : public mlir::RewriterBase::Listener {
  bool changed = false;

  void notifyOperationModified(mlir::Operation* op) override { changed = true; }

  void notifyOperationReplaced(mlir::Operation* op,
                               mlir::ValueRange newValues) override {
    changed = true;
  }

  void notifyBlockErased(mlir::Block* block) override { changed = true; }
};

bool runLocalPatternsWithTracking(mlir::Operation* root,
                                  mlir::MLIRContext* ctx) {
  mlir::RewritePatternSet patterns(ctx);
  populateReplaceBasisStateControlsWithIfPatterns(patterns); // Comment out
  // if QCP os applied before (in a loop)
  // populateAdaptCtrldPauliZToLiftingPatterns(patterns);
  // populateLiftHadamardsAbovePauliGatesPatterns(patterns);
  // populateLiftHadamardAboveCNOTPattern(patterns);
  populateLiftMeasurementsAboveControlsPatterns(patterns);
  populateLiftMeasurementsAboveGatesPatterns(patterns);
  populateDeadGateEliminationPatterns(patterns);

  ChangeTrackingListener listener;

  mlir::GreedyRewriteConfig config;
  config.setListener(&listener);

  if (mlir::failed(
          mlir::applyPatternsGreedily(root, std::move(patterns), config))) {
    throw std::runtime_error("Failure during greedy pattern application.");
  }

  return listener.changed;
}

/**
 * @brief This pass attempts to lift measurements above certain operations.
 */
struct LiftMeasurementsPass final
    : impl::LiftMeasurementsPassBase<LiftMeasurementsPass> {

  void runOnOperation() override {
    // Get the current operation being operated on.
    auto op = getOperation();
    auto* ctx = &getContext();

    bool changed = true;
    while (changed) {
      changed = false; // Run your global routine
      changed |= moveMeasurementsToFront(op, ctx);
      changed |= applyQCP(op, ctx);
      // Run local patterns and detect changes
      changed |= runLocalPatternsWithTracking(op, ctx);
    }

    // Define the set of patterns to use.
    // mlir::RewritePatternSet patterns(ctx);
    // populateReplaceBasisStateControlsWithIfPatterns(patterns); // Comment out
    // if QCP os applied before (in a loop)
    // populateAdaptCtrldPauliZToLiftingPatterns(patterns);
    // populateLiftHadamardsAbovePauliGatesPatterns(patterns);
    // populateLiftHadamardAboveCNOTPattern(patterns);
    // populateLiftMeasurementsAboveControlsPatterns(patterns);
    // populateLiftMeasurementsAboveGatesPatterns(patterns);
    // populateDeadGateEliminationPatterns(patterns);

    // Apply patterns in an iterative and greedy manner.
    // if (mlir::failed(mlir::applyPatternsGreedily(op, std::move(patterns)))) {
    //   signalPassFailure();
    // }
  }
};

} // namespace mqt::ir::opt
