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

  /**
   * @brief Checks if the target qubit of gate 1 is part of the ctrl qubits of
   * gate 2 and vice versa.
   *
   * This method checks if the output target qubit of gate 1 is used as control
   * qubit of gate 2. Additionally, it checks if the input target of gate 2 is
   * an output control of gate 2. Returns true if that is the case.
   * Must only be used on gates that have a single target qubit.
   *
   * @param gate1 First gate, predecessor of gate2.
   * @param gate2 Second gate, successor of gate1.
   * @return True if target qubit of gate1 is ctrl in gate2 and vice versa.
   * False otherwise.
   */
  static bool areTargetsControlsAtTheOtherGates(UnitaryInterface gate1,
                                                UnitaryInterface gate2) {
    mlir::Value targetQubitGate2 = gate2.getInQubits()[0];
    mlir::Value targetQubitGate1 = gate1.getOutQubits()[0];
    auto inCtrlGate2 = gate2.getPosCtrlInQubits();
    auto outCtrlGate1 = gate1.getPosCtrlOutQubits();

    return std::find(inCtrlGate2.begin(), inCtrlGate2.end(),
                     targetQubitGate1) != inCtrlGate2.end() &&
           std::find(outCtrlGate1.begin(), outCtrlGate1.end(),
                     targetQubitGate2) != outCtrlGate1.end();
  }

  /**
   * @brief This method exchanges the position of two qubits acting on the same
   * gate.
   *
   * This method exchanges two qubits acting on the same gate. E.g. if qubit 1
   * is a target qubit and qubit 2 a control qubit, that is exchanged.
   *
   *
   * @param rewriter The rewriter.
   * @param gate The gate both qubit1 and qubit2 belong to.
   * @param qubit1 First qubit, exchanged with second.
   * @param qubit2 Second qubit, exchanged with first.
   * @param temporary Qubit that is not used on the respective gate. Used as
   * temporary variable.
   */
  static void exchangeTwoQubitsAtGate(mlir::PatternRewriter& rewriter,
                                      UnitaryInterface gate, mlir::Value qubit1,
                                      mlir::Value qubit2,
                                      mlir::Value temporary) {

    rewriter.replaceUsesWithIf(
        qubit1, temporary,
        [&](mlir::OpOperand& operand) { return operand.getOwner() == gate; });
    rewriter.replaceUsesWithIf(qubit2, qubit1, [&](mlir::OpOperand& operand) {
      return operand.getOwner() == gate;
    });
    rewriter.replaceUsesWithIf(
        temporary, qubit2,
        [&](mlir::OpOperand& operand) { return operand.getOwner() == gate; });
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
    if (!areTargetsControlsAtTheOtherGates(op, hadamardGate)) {
      return mlir::failure();
    }

    // Put the Z target to the same qubit as the hadamard target is
    mlir::Value originalTargetQubitZ = op.getInQubits()[0];
    mlir::Value targetQubitHadamard = hadamardGate.getInQubits()[0];
    mlir::Value newTargetQubitZ = op.getCorrespondingInput(targetQubitHadamard);
    mlir::Value temporary = hadamardGate.getOutQubits()[0];

    exchangeTwoQubitsAtGate(rewriter, op, originalTargetQubitZ, newTargetQubitZ,
                            temporary);

    mlir::Value newTargetQubitH = op.getCorrespondingOutput(newTargetQubitZ);
    temporary = op.getInQubits()[0];

    exchangeTwoQubitsAtGate(rewriter, hadamardGate, targetQubitHadamard,
                            newTargetQubitH, temporary);

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