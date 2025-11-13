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

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

namespace mqt::ir::opt {

/**
 * @brief This pattern changes the target of a controlled Pauli Z gate if a
 * controlled hadamard gate is it successor.
 * If all out qubits of Pauli Z are equal to all in qubits of hadamard, we can
 * commute the gates and change Pauli Z to X. This is only possible if hadamard
 * and Pauli act on the same qubit as target. If the target of the Pauli gate is
 * a ctrl at the hadamard and vice versa, we can change the target of Pauli Z to
 * the hadamard's. This is done in this pattern.
 */
struct AdaptCtrldPauliZToLiftingPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {

  explicit AdaptCtrldPauliZToLiftingPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(UnitaryInterface op,
                  mlir::PatternRewriter& rewriter) const override {
    return mlir::failure();
  }
};

/**
 * @brief Populates the given pattern set with the
 * `AdaptCtrldPauliZToLiftingPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateAdaptCtrldPauliZToLiftingPatterns(
    mlir::RewritePatternSet& patterns) {
  patterns.add<AdaptCtrldPauliZToLiftingPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt