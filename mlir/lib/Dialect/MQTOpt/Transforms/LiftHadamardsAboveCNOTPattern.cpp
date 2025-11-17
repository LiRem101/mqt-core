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
 * @brief This pattern remove an H gate between a CNOT and a measurement.
 *
 * If there is a hadamard gate between the target qubit of a CNOT and a
 * measurement, we flip the CNOT and apply a hadamard gate to the incoming and
 * outcoming qubits. As H * H = id, the measurement is then the direct successor
 * of a CNOT ctrl, which is beneficial for the qubit reuse routine.
 * The procedure also works if there are additional (n)ctrls. Only the target
 * and ctrl involved in the transformation get hadamard gates assigned.
 * For now, the involved ctrl to be flipped with the target is chosen randomly.
 */
struct LiftHadamardAboveCNOTPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {

  explicit LiftHadamardAboveCNOTPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(UnitaryInterface op,
                  mlir::PatternRewriter& rewriter) const override {
    return mlir::failure();
  }
};

/**
 * @brief Populates the given pattern set with the
 * `LiftHadamardAboveCNOTPatternPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateLiftHadamardAboveCNOTPattern(
    mlir::RewritePatternSet& patterns) {
  patterns.add<LiftHadamardAboveCNOTPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt