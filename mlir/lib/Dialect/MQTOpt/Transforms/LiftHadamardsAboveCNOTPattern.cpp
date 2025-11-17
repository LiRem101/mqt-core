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

    // Remove the hadmard gate
    for (auto outQubit : hadamardGate.getAllOutQubits()) {
      rewriter.replaceAllUsesWith(outQubit,
                                  hadamardGate.getCorrespondingInput(outQubit));
    }
    rewriter.eraseOp(hadamardGate);

    // Add hadamard gates to the other in and output gates of cnot
    std::vector<mlir::Value> inQubitBeforeTarget{
        cnotGate.getInQubits().front()};
    std::vector<mlir::Type> outQubitBeforeTarget{
        cnotGate.getInQubits().front().getType()};
    auto newHOPBeforeTarget = rewriter.create<HOp>(
        /* location = */ cnotGate->getLoc(),
        /* out_qubits = */ outQubitBeforeTarget,
        /* pos_ctrl_out_qubits = */ mlir::TypeRange{},
        /* neg_ctrl_out_qubits = */ mlir::TypeRange{},
        /* static_params = */ nullptr,
        /* params_mask = */ nullptr,
        /* params = */ mlir::ValueRange{},
        /* in_qubits = */ inQubitBeforeTarget,
        /* pos_ctrl_in_qubits = */ mlir::ValueRange{},
        /* neg_ctrl_in_qubits = */ mlir::ValueRange{});
    rewriter.moveOpBefore(newHOPBeforeTarget, cnotGate);
    rewriter.replaceUsesWithIf(cnotGate.getInQubits().front(),
                               newHOPBeforeTarget.getOutQubits().front(),
                               [&](mlir::OpOperand& operand) {
                                 return operand.getOwner() == cnotGate;
                               });

    std::vector<mlir::Value> inQubitBeforeCtrl{
        cnotGate.getPosCtrlInQubits().front()};
    std::vector<mlir::Type> outQubitBeforeCtrl{
        cnotGate.getPosCtrlInQubits().front().getType()};
    auto newHOPBeforeCtrl = rewriter.create<HOp>(
        /* location = */ cnotGate->getLoc(),
        /* out_qubits = */ outQubitBeforeCtrl,
        /* pos_ctrl_out_qubits = */ mlir::TypeRange{},
        /* neg_ctrl_out_qubits = */ mlir::TypeRange{},
        /* static_params = */ nullptr,
        /* params_mask = */ nullptr,
        /* params = */ mlir::ValueRange{},
        /* in_qubits = */ inQubitBeforeCtrl,
        /* pos_ctrl_in_qubits = */ mlir::ValueRange{},
        /* neg_ctrl_in_qubits = */ mlir::ValueRange{});
    rewriter.moveOpBefore(newHOPBeforeCtrl, cnotGate);
    rewriter.replaceUsesWithIf(cnotGate.getPosCtrlInQubits().front(),
                               newHOPBeforeCtrl.getOutQubits().front(),
                               [&](mlir::OpOperand& operand) {
                                 return operand.getOwner() == cnotGate;
                               });

    auto cnotPosCtrlOutQubit = cnotGate.getPosCtrlOutQubits().front();
    std::vector<mlir::Value> inQubitAfterCtrl{cnotPosCtrlOutQubit};
    std::vector<mlir::Type> outQubitAfterCtrl{cnotPosCtrlOutQubit.getType()};
    auto newHOPAfterCtrl = rewriter.create<HOp>(
        /* location = */ cnotGate->getLoc(),
        /* out_qubits = */ outQubitAfterCtrl,
        /* pos_ctrl_out_qubits = */ mlir::TypeRange{},
        /* neg_ctrl_out_qubits = */ mlir::TypeRange{},
        /* static_params = */ nullptr,
        /* params_mask = */ nullptr,
        /* params = */ mlir::ValueRange{},
        /* in_qubits = */ inQubitAfterCtrl,
        /* pos_ctrl_in_qubits = */ mlir::ValueRange{},
        /* neg_ctrl_in_qubits = */ mlir::ValueRange{});
    rewriter.replaceUsesWithIf(newHOPAfterCtrl.getInQubits().front(),
                               newHOPAfterCtrl.getOutQubits().front(),
                               [&](mlir::OpOperand& operand) {
                                 return operand.getOwner() != cnotGate &&
                                        operand.getOwner() != newHOPAfterCtrl;
                               });

    // TODO: Flip CNOT targets and ctrl

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