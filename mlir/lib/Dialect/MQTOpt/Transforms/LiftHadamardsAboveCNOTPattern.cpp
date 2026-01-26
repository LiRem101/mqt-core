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

#include <mlir/Dialect/SparseTensor/IR/Enums.h>
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
struct LiftHadamardAboveCNOTPattern final : mlir::OpRewritePattern<MeasureOp> {

  explicit LiftHadamardAboveCNOTPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  /**
   * @brief This method swaps two qubits on a gate.
   *
   * This method swaps two qubits on a gate. Input and output are exchanged.
   *
   * @param gate The gate that the qubits belong to.
   * @param inputQubit1 The input qubit of the qubit to be exchanged with 2.
   * @param inputQubit2 The input qubit of the qubit to be exchanged with 1.
   * @param succeedingOp1 The operation succeeding gate on the corresponding
   * output of inputQubit1.
   * @param succeedingOp2 The operation succeeding gate on the corresponding
   * output of inputQubit2.
   * @param dummy A qubit that exists in the circuit but is not used by gate.
   * Has to be somewhere in the circuit before the respective gate. Needed to do
   * the exchange, but not changed.
   * @param rewriter The used rewriter.
   */
  static void swapQubits(UnitaryInterface gate, mlir::Value inputQubit1,
                         mlir::Value inputQubit2,
                         mlir::Operation* succeedingOp1,
                         mlir::Operation* succeedingOp2, mlir::Value dummy,
                         mlir::PatternRewriter& rewriter) {
    mlir::Value outputQubit1 = gate.getCorrespondingOutput(inputQubit1);
    mlir::Value outputQubit2 = gate.getCorrespondingOutput(inputQubit2);

    rewriter.replaceUsesWithIf(outputQubit1, dummy,
                               [&](mlir::OpOperand& operand) {
                                 return operand.getOwner() == gate ||
                                        operand.getOwner() == succeedingOp1 ||
                                        operand.getOwner() == succeedingOp2;
                               });
    rewriter.replaceUsesWithIf(outputQubit2, outputQubit1,
                               [&](mlir::OpOperand& operand) {
                                 return operand.getOwner() == gate ||
                                        operand.getOwner() == succeedingOp1 ||
                                        operand.getOwner() == succeedingOp2;
                               });
    rewriter.replaceUsesWithIf(dummy, outputQubit2,
                               [&](mlir::OpOperand& operand) {
                                 return operand.getOwner() == gate ||
                                        operand.getOwner() == succeedingOp1 ||
                                        operand.getOwner() == succeedingOp2;
                               });

    rewriter.replaceUsesWithIf(
        inputQubit1, dummy,
        [&](mlir::OpOperand& operand) { return operand.getOwner() == gate; });
    rewriter.replaceUsesWithIf(
        inputQubit2, inputQubit1,
        [&](mlir::OpOperand& operand) { return operand.getOwner() == gate; });
    rewriter.replaceUsesWithIf(
        dummy, inputQubit2,
        [&](mlir::OpOperand& operand) { return operand.getOwner() == gate; });
  }

  /**
   * @brief This method adds hadamrad gates before a given gate.
   *
   * @param gate The gate before which hadamard gates should be applied.
   * @param inputQubits The input qubits of gate before which hadamard gates
   * should be applied.
   * @param rewriter The used rewriter.
   *
   * @returns One of the created hadamard gates.
   */
  static HOp addHadamardGatesBeforeGate(UnitaryInterface gate,
                                        std::vector<mlir::Value> inputQubits,
                                        mlir::PatternRewriter& rewriter) {
    HOp newHOP;
    for (mlir::Value inputQubit : inputQubits) {

      std::vector<mlir::Value> inQubits{inputQubit};
      std::vector<mlir::Type> outQubits{inputQubit.getType()};

      newHOP = rewriter.create<HOp>(
          /* location = */ gate->getLoc(),
          /* out_qubits = */ outQubits,
          /* pos_ctrl_out_qubits = */ mlir::TypeRange{},
          /* neg_ctrl_out_qubits = */ mlir::TypeRange{},
          /* static_params = */ nullptr,
          /* params_mask = */ nullptr,
          /* params = */ mlir::ValueRange{},
          /* in_qubits = */ inQubits,
          /* pos_ctrl_in_qubits = */ mlir::ValueRange{},
          /* neg_ctrl_in_qubits = */ mlir::ValueRange{});

      rewriter.moveOpBefore(newHOP, gate);

      rewriter.replaceUsesWithIf(
          inputQubit, newHOP.getOutQubits().front(),
          [&](mlir::OpOperand& operand) { return operand.getOwner() == gate; });
    }
    return newHOP;
  }

  /**
   * @brief This method adds hadamrad gates after a given gate.
   *
   * @param gate The gate after which hadamard gates should be applied.
   * @param outputQubits The output qubits of gate after which hadamard gates
   * should be applied.
   * @param rewriter The used rewriter.
   *
   * @returns One of the created hadamard gates.
   */
  static HOp addHadamardGatesAfterGate(UnitaryInterface gate,
                                       std::vector<mlir::Value> outputQubits,
                                       mlir::PatternRewriter& rewriter) {
    HOp newHOp;
    for (mlir::Value outputQubit : outputQubits) {

      std::vector<mlir::Value> inQubit{outputQubit};
      std::vector<mlir::Type> outQubit{outputQubit.getType()};

      newHOp = rewriter.create<HOp>(
          /* location = */ gate->getLoc(),
          /* out_qubits = */ outQubit,
          /* pos_ctrl_out_qubits = */ mlir::TypeRange{},
          /* neg_ctrl_out_qubits = */ mlir::TypeRange{},
          /* static_params = */ nullptr,
          /* params_mask = */ nullptr,
          /* params = */ mlir::ValueRange{},
          /* in_qubits = */ inQubit,
          /* pos_ctrl_in_qubits = */ mlir::ValueRange{},
          /* neg_ctrl_in_qubits = */ mlir::ValueRange{});

      rewriter.moveOpAfter(newHOp, gate);

      rewriter.replaceUsesWithIf(
          newHOp.getInQubits().front(), newHOp.getOutQubits().front(),
          [&](mlir::OpOperand& operand) {
            return operand.getOwner() != gate && operand.getOwner() != newHOp;
          });
    }
    return newHOp;
  }

  mlir::LogicalResult
  matchAndRewrite(MeasureOp op,
                  mlir::PatternRewriter& rewriter) const override {
    // A hadamard gate needs to be in front of the measurement
    const auto qubitInMeasurement = op.getInQubit();
    auto* predecessor = qubitInMeasurement.getDefiningOp();
    auto hadamardGate = mlir::dyn_cast<UnitaryInterface>(predecessor);
    if (!hadamardGate || hadamardGate.isControlled() ||
        hadamardGate->getName().stripDialect().str() != "h") {
      return mlir::failure();
    }

    // The hadamard gate must be successor of the target of a CNOT
    auto inQubitsHadamard = hadamardGate.getInQubits();
    if (inQubitsHadamard.empty()) {
      return mlir::failure();
    }
    // We know the hadamard gate is not controlled
    auto inQubitHadamard = inQubitsHadamard.front();
    predecessor = inQubitHadamard.getDefiningOp();
    auto cnotGate = mlir::dyn_cast<UnitaryInterface>(predecessor);
    if (!cnotGate || !cnotGate.isControlled() ||
        cnotGate->getName().stripDialect().str() != "x") {
      return mlir::failure();
    }
    auto posCtrlOutQubits = cnotGate.getPosCtrlOutQubits();
    if (posCtrlOutQubits.empty()) {
      return mlir::failure();
    }

    // the hadamard gate needs to be connected to the target
    auto targetOutQubits = cnotGate.getOutQubits();
    if (targetOutQubits.empty() || targetOutQubits.front() != inQubitHadamard) {
      return mlir::failure();
    }

    // Remove the Hadamard gate
    for (auto outQubit : hadamardGate.getAllOutQubits()) {
      rewriter.replaceAllUsesWith(outQubit,
                                  hadamardGate.getCorrespondingInput(outQubit));
    }
    rewriter.eraseOp(hadamardGate);

    // Add hadamard gates to the other in and output gates of cnot
    std::vector<mlir::Value> relevantInputQubitsForHadamard{
        cnotGate.getInQubits().front(), cnotGate.getPosCtrlInQubits().front()};
    HOp newHOPBefore = addHadamardGatesBeforeGate(
        cnotGate, relevantInputQubitsForHadamard, rewriter);

    std::vector<mlir::Value> relevantOutputQubitsFOrHadamard{
        cnotGate.getPosCtrlOutQubits().front()};
    HOp newHOPAfterCtrl = addHadamardGatesAfterGate(
        cnotGate, relevantOutputQubitsFOrHadamard, rewriter);

    // Flip CNOT targets and ctrl
    swapQubits(cnotGate, cnotGate.getPosCtrlInQubits().front(),
               cnotGate.getInQubits().front(), op, newHOPAfterCtrl,
               newHOPBefore.getInQubits().front(), rewriter);

    return mlir::success();
  }
};

/**
 * @brief Populates the given pattern set with the
 * `LiftHadamardAboveCNOTPatternPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateLiftHadamardAboveCNOTPattern(mlir::RewritePatternSet& patterns) {
  patterns.add<LiftHadamardAboveCNOTPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt