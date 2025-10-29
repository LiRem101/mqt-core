
/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/UseDefLists.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <string>
#include <unordered_set>

namespace mqt::ir::opt {

/**
 * @brief This pattern is responsible for lifting Hadamard gates above any pauli
 * gate.
 */
struct LiftHadamardsAbovePauliGatesPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {

  explicit LiftHadamardsAbovePauliGatesPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(UnitaryInterface op,
                  mlir::PatternRewriter& rewriter) const override {
    return mlir::failure();
  }
};

/**
 * @brief Populates the given pattern set with the
 * `LiftHadamardsAbovePauliGatesPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateLiftHadamardsAbovePauliGatesPatterns(
    mlir::RewritePatternSet& patterns) {
  patterns.add<LiftHadamardsAbovePauliGatesPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
