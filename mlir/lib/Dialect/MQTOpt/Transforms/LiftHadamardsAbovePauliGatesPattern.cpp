
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

void swapSingleGates(UnitaryInterface firstGate, UnitaryInterface secondGate,
                     mlir::PatternRewriter& rewriter) {
  // auto measurementInput = secondGate.getInQubit();
  auto gateInput = firstGate.getInQubits()[0];
  auto secondGateInput = secondGate.getInQubits()[0];
  rewriter.replaceUsesWithIf(secondGateInput, gateInput,
                             [&](mlir::OpOperand& operand) {
                               // We only replace the single use by the
                               // measure op
                               return operand.getOwner() == secondGate;
                             });
  rewriter.replaceUsesWithIf(gateInput, secondGate.getOutQubits()[0],
                             [&](mlir::OpOperand& operand) {
                               // We only replace the single use by the
                               // predecessor
                               return operand.getOwner() == firstGate;
                             });
  rewriter.replaceUsesWithIf(secondGate.getOutQubits()[0], secondGateInput,
                             [&](mlir::OpOperand& operand) {
                               // All further uses of the measurement output now
                               // use the gate output
                               return operand.getOwner() != firstGate;
                             });
  rewriter.moveOpBefore(secondGate, firstGate);
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

    const auto& users = op->getUsers();
    if (users.empty()) {
      return mlir::failure();
    }
    auto* user = *users.begin();
    const auto userName = user->getName().stripDialect().str();

    if (!isGateHadamardGate(*user)) {
      return mlir::failure();
    }

    const auto opName = op->getName().stripDialect().str();
    for (std::unordered_set invertPair : INVERTING_GATES) {
      if (invertPair.contains(opName)) {
        bool exchangeGate = invertPair.size() == 2;

        swapSingleGates(op, mlir::dyn_cast<UnitaryInterface>(user), rewriter);

        if (exchangeGate) {
          const auto qubitType = QubitType::get(rewriter.getContext());
          if (opName == "x") {
            rewriter.replaceOpWithNewOp<ZOp>(
                op, qubitType, mlir::TypeRange{}, mlir::TypeRange{},
                mlir::DenseF64ArrayAttr{}, mlir::DenseBoolArrayAttr{},
                mlir::ValueRange{}, op.getInQubits(), mlir::ValueRange{},
                mlir::ValueRange{});
          } else {
            rewriter.replaceOpWithNewOp<XOp>(
                op, qubitType, mlir::TypeRange{}, mlir::TypeRange{},
                mlir::DenseF64ArrayAttr{}, mlir::DenseBoolArrayAttr{},
                mlir::ValueRange{}, op.getInQubits(), mlir::ValueRange{},
                mlir::ValueRange{});
          }
        }
        return mlir::success();
      }
    }

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
