
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

#include <iostream>
#include <list>
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

// If one gate changes places with a hadamard gate, it is exchanged by the other
// gate in the same set
static const std::list<std::unordered_set<std::string>> INVERTING_GATES = {
    {"x", "z"}, {"y", "y"}};

/**
 * @brief This method swaps two single unitary gates. Does not yet work on
 * controlled gates.
 *
 * This method swaps two unitary gates. The gates need to be applied on one
 * single qubits. The gates may not be controlled.
 *
 * @param firstGate The first unitary gate.
 * @param secondGate The second unitary gate.
 * @param rewriter The used rewriter.
 */
void swapSingleGates(UnitaryInterface firstGate, UnitaryInterface secondGate,
                     mlir::PatternRewriter& rewriter) {
  auto firstGateInputs = firstGate.getAllInQubits();
  auto secondGateInputs = secondGate.getAllInQubits();
  auto secondGateOutputs = secondGate.getAllOutQubits();
  rewriter.replaceUsesWithIf(secondGateInputs, firstGateInputs,
                             [&](mlir::OpOperand& operand) {
                               // We only replace the single use by the
                               // second gate
                               return operand.getOwner() == secondGate;
                             });
  rewriter.replaceUsesWithIf(firstGateInputs, secondGateOutputs,
                             [&](mlir::OpOperand& operand) {
                               // We only replace the single use by the
                               // first gate
                               return operand.getOwner() == firstGate;
                             });
  rewriter.replaceUsesWithIf(secondGateOutputs, secondGateInputs,
                             [&](mlir::OpOperand& operand) {
                               // All further uses of the second gate output now
                               // use the first gate output
                               return operand.getOwner() != firstGate;
                             });
  rewriter.moveOpBefore(secondGate, firstGate);
}

/**
 * @brief This method swaps a gate with is succeeding hadamard gate, if
 * applicable.
 *
 * This method swaps a gate with its suceeding hadamard gate. This is only done
 * if there is a simple commutation rule to do so. Currently implemented:
 * - X - H - = - H - Z -
 * - Y - H - = - H - Y -
 * - Z - H - = - H - X -
 *
 * @param gate The unitary gate.
 * @param hadamardGate The hadamard gate.
 * @param rewriter The used rewriter.
 */
mlir::LogicalResult swapGateWithHadamard(UnitaryInterface gate,
                                         UnitaryInterface hadamardGate,
                                         mlir::PatternRewriter& rewriter) {
  const auto gateName = gate->getName().stripDialect().str();

  if (gateName == "x" || gateName == "y" || gateName == "z") {
    swapSingleGates(gate, hadamardGate, rewriter);

    const auto qubitType = QubitType::get(rewriter.getContext());

    if (gateName == "x") {
      rewriter.replaceOpWithNewOp<ZOp>(
          gate, qubitType, qubitType, qubitType,
          mlir::DenseF64ArrayAttr{}, mlir::DenseBoolArrayAttr{},
          mlir::ValueRange{}, gate.getInQubits(), gate.getPosCtrlInQubits(),
          gate.getNegCtrlInQubits());
    } else if (gateName == "z") {
      rewriter.replaceOpWithNewOp<XOp>(
          gate, qubitType, qubitType, qubitType,
          mlir::DenseF64ArrayAttr{}, mlir::DenseBoolArrayAttr{},
          mlir::ValueRange{}, gate.getInQubits(), gate.getPosCtrlInQubits(),
          gate.getNegCtrlInQubits());
    }

    return mlir::success();
  }
  return mlir::failure();
}

/**
 * @brief This pattern is responsible for lifting Hadamard gates above any pauli
 * gate.
 */
struct LiftHadamardsAbovePauliGatesPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {

  /**
   * @brief Checks if a gate is a hadamard gate.
   *
   * @param a The gate.
   * @return True if the gate is a hadamard gate, false otherwise.
   */
  [[nodiscard]] static bool isGateHadamardGate(mlir::Operation& a) {
    const auto aName = a.getName().stripDialect().str();
    return aName == "h";
  }

  explicit LiftHadamardsAbovePauliGatesPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(UnitaryInterface op,
                  mlir::PatternRewriter& rewriter) const override {

    // op needs to be in front of a hadamard gate
    const auto& users = op->getUsers();
    if (users.empty()) {
      return mlir::failure();
    }
    auto* user = *users.begin();
    const auto userName = user->getName().stripDialect().str();
    if (!isGateHadamardGate(*user)) {
      return mlir::failure();
    }

    return swapGateWithHadamard(op, mlir::dyn_cast<UnitaryInterface>(user),
                                rewriter);
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
