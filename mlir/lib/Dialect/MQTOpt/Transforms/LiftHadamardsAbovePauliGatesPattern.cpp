
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
#include <set>
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

  /**
   * @brief Checks if a gate is a z gate.
   *
   * @param a The gate.
   * @return True if the gate is a z gate, false otherwise.
   */
  [[nodiscard]] static bool isGateZGate(mlir::Operation& a) {
    const auto aName = a.getName().stripDialect().str();
    return aName == "z";
  }

  /**
   * @brief This method swaps two unitary gates. Does not yet work on
   * controlled gates.
   *
   * This method swaps two unitary gates. The gates need to be applied on one
   * single qubits. The gates may not be controlled.
   *
   * @param firstGate The first unitary gate.
   * @param secondGate The second unitary gate.
   * @param rewriter The used rewriter.
   */
  static void swapGates(UnitaryInterface firstGate, UnitaryInterface secondGate,
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
                                 // All further uses of the second gate output
                                 // now use the first gate output
                                 return operand.getOwner() != firstGate;
                               });
    rewriter.moveOpBefore(secondGate, firstGate);
  }

  /**
   * @brief This method checks if two gates are connected by exactly the same
   * target, ctrl and nctrl qubits.
   *
   * This method checks if the output target/ctrl/nctrl qubits of the first gate
   * are exactly the input target/ctrl/nctrl qubits of the second gate. There
   * must be no qubit that is only used by one of the gates.
   *
   * @param firstGate The first unitary gate.
   * @param secondGate The second unitary gate.
   */
  static bool areGatesConnectedBySameQubits(UnitaryInterface firstGate,
                                            UnitaryInterface secondGate) {
    auto ctrlOutQubits = firstGate.getPosCtrlOutQubits();
    auto ctrlInQubits = secondGate.getPosCtrlInQubits();
    auto nctrlOutQubits = firstGate.getNegCtrlOutQubits();
    auto nctrlInQubits = secondGate.getNegCtrlInQubits();
    auto targetOutQubits = firstGate.getOutQubits();
    auto targetInQubits = secondGate.getInQubits();

    bool ctrlEqual = std::equal(ctrlOutQubits.begin(), ctrlOutQubits.end(),
                                ctrlInQubits.begin(), ctrlInQubits.end());
    bool nctrlEqual = std::equal(nctrlOutQubits.begin(), nctrlOutQubits.end(),
                                 nctrlInQubits.begin(), nctrlInQubits.end());
    bool targetsEqual =
        std::equal(targetOutQubits.begin(), targetOutQubits.end(),
                   targetInQubits.begin(), targetInQubits.end());
    return ctrlEqual && nctrlEqual && targetsEqual;
  }

  /**
   * @brief This method swaps a gate with is succeeding hadamard gate, if
   * applicable.
   *
   * This method swaps a gate with its suceeding hadamard gate. This is only
   * done if there is a simple commutation rule to do so.
   * Currently implemented:
   * - X - H - = - H - Z -
   * - Y - H - = - H - Y -
   * - Z - H - = - H - X -
   *
   * @param gate The unitary gate.
   * @param hadamardGate The hadamard gate.
   * @param rewriter The used rewriter.
   */
  static mlir::LogicalResult
  swapGateWithHadamard(UnitaryInterface gate, UnitaryInterface hadamardGate,
                       mlir::PatternRewriter& rewriter) {
    const auto gateName = gate->getName().stripDialect().str();

    if (gateName == "x" || gateName == "y" || gateName == "z") {
      swapGates(gate, hadamardGate, rewriter);

      const auto qubitType = QubitType::get(rewriter.getContext());
      auto inQubits = gate.getInQubits();
      auto posCtrlInQubits = gate.getPosCtrlInQubits();
      auto negCtrlInQubits = gate.getNegCtrlInQubits();
      auto postCtrlOutQubitsType = gate.getPosCtrlOutQubits().getType();
      auto negCtrlOutQubitsType = gate.getNegCtrlOutQubits().getType();
      if (gateName == "x") {
        rewriter.replaceOpWithNewOp<ZOp>(
            gate, qubitType, postCtrlOutQubitsType, negCtrlOutQubitsType,
            mlir::DenseF64ArrayAttr{}, mlir::DenseBoolArrayAttr{},
            mlir::ValueRange{}, inQubits, posCtrlInQubits, negCtrlInQubits);
      } else if (gateName == "z") {
        rewriter.replaceOpWithNewOp<XOp>(
            gate, qubitType, postCtrlOutQubitsType, negCtrlOutQubitsType,
            mlir::DenseF64ArrayAttr{}, mlir::DenseBoolArrayAttr{},
            mlir::ValueRange{}, inQubits, posCtrlInQubits, negCtrlInQubits);
      }

      return mlir::success();
    }
    return mlir::failure();
  }

  mlir::LogicalResult
  matchAndRewrite(UnitaryInterface op,
                  mlir::PatternRewriter& rewriter) const override {

    // op needs to be in front of a hadamard gate
    const auto& users = op->getUsers();
    if (users.empty()) {
      return mlir::failure();
    }
    auto user = *users.begin();
    if (!isGateHadamardGate(*user)) {
      return mlir::failure();
    }

    auto hadamardGate = mlir::dyn_cast<UnitaryInterface>(user);
    if (!areGatesConnectedBySameQubits(op, hadamardGate)) {
      // check if gate is z gate. If so, check if all qubits between hadamard
      // and z are the same. If the target qubit of H is a ctrl in Z and vice
      // versa, move Z's target to H's target and continue with swap
      if (isGateZGate(*op)) {
        auto inQubits = hadamardGate.getAllInQubits();
        auto outQubits = op.getAllOutQubits();

        bool qubitsEqual = true;

        qubitsEqual &= inQubits.size() == outQubits.size();
        for (auto inQubit : inQubits) {
          qubitsEqual &= std::find(outQubits.begin(), outQubits.end(), inQubit) != outQubits.end();
        }

        if (qubitsEqual) {
          // If the target qubit of H is a ctrl in Z and vice versa, move Z's
          // target to H's target and continue with swap
        }
      }
      return mlir::failure();
    }

    return swapGateWithHadamard(op, hadamardGate, rewriter);
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
