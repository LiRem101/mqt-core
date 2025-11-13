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

#include <mlir/Dialect/Vector/IR/VectorOps.h>
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

  /**
   * @brief This method checks if two gates are connected by the same qubits.
   *
   * This method checks if the output qubits of the first gate are exactly the
   * input qubits of the second gate. There must be no qubit that is only used
   * by one of the gates. The qubits may have different tasks (e.g. bein target
   * in the first gate but ctrl in the second).
   *
   * @param firstGate The first unitary gate.
   * @param secondGate The second unitary gate.
   */
  static bool areGatesConnectedBySameQubits(UnitaryInterface firstGate,
                                            UnitaryInterface secondGate) {
    auto inQubits = secondGate.getAllInQubits();
    auto outQubits = firstGate.getAllOutQubits();

    bool result = true;
    result &= inQubits.size() == outQubits.size();
    for (auto element : inQubits) {
      result &= std::find(outQubits.begin(), outQubits.end(), element) !=
                outQubits.end();
    }
    return result;
  }

  mlir::LogicalResult
  matchAndRewrite(UnitaryInterface op,
                  mlir::PatternRewriter& rewriter) const override {
    // op needs to be a Pauli Z gate and controlled
    std::string opName = op->getName().stripDialect().str();
    if (opName != "z" || !op.isControlled()) {
      return mlir::failure();
    }

    // op needs to be in front of a controlled hadamard gate
    const auto& users = op->getUsers();
    if (users.empty()) {
      return mlir::failure();
    }
    auto user = *users.begin();
    if (user->getName().stripDialect().str() != "h") {
      return mlir::failure();
    }

    auto hadamardGate = mlir::dyn_cast<UnitaryInterface>(user);
    if (!areGatesConnectedBySameQubits(op, hadamardGate)) {
      return mlir::failure();
    }

    // If the target qubit of H is a ctrl in Z and vice versa, we can move Z's
    // target to H's target

    mlir::Value targetQubitHadamard = hadamardGate.getInQubits()[0];
    mlir::Value targetQubitZ = op.getOutQubits()[0];
    auto inCtrlHadamards = hadamardGate.getPosCtrlInQubits();
    auto outCtrlPauliZ = op.getPosCtrlOutQubits();

    bool zIsHadamardCtrlAndHadamardIsZCtrl =
        std::find(inCtrlHadamards.begin(), inCtrlHadamards.end(),
                  targetQubitZ) != inCtrlHadamards.end() &&
        std::find(outCtrlPauliZ.begin(), outCtrlPauliZ.end(),
                  targetQubitHadamard) != outCtrlPauliZ.end();

    if (!zIsHadamardCtrlAndHadamardIsZCtrl) {
      return mlir::failure();
    }

    auto targetQubitIn = op.getInQubits()[0];
    auto targetQubitOut = op.getOutQubits()[0];

    auto newTargetQubitInZ = op.getCorrespondingInput(targetQubitHadamard);

    auto between = hadamardGate.getOutQubits()[0];

    rewriter.replaceUsesWithIf(targetQubitIn, between,
                               [&](mlir::OpOperand& operand) {
                                 // We only replace the single use by the
                                 // modified operation.
                                 return operand.getOwner() == op;
                               });
    rewriter.replaceUsesWithIf(newTargetQubitInZ, targetQubitIn,
                               [&](mlir::OpOperand& operand) {
                                 // We only replace the single use by the
                                 // modified operation.
                                 return operand.getOwner() == op;
                               });
    rewriter.replaceUsesWithIf(between, newTargetQubitInZ,
                               [&](mlir::OpOperand& operand) {
                                 // We only replace the single use by the
                                 // modified operation.
                                 return operand.getOwner() == op;
                               });

    rewriter.replaceUsesWithIf(targetQubitHadamard, targetQubitOut,
                               [&](mlir::OpOperand& operand) {
                                 // We only replace the single use by the
                                 // modified operation.
                                 return operand.getOwner() == hadamardGate;
                               });

    return mlir::success();
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