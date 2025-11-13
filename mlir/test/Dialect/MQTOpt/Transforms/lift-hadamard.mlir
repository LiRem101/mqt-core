// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --lift-hadamards | FileCheck %s

// =================================================== Raise over one Pauli gate =========================================================================
// -----
// This test checks if hadamard gates can be lifted over one Pauli gate.
// In this example, the following liftings are being tested:
//   - Pauli-X followed by a Hadamard gate
//   - Pauli-Y followed by a Hadamard gate
//   - Pauli-Z followed by a Hadamard gate

module {
  func.func @testLiftHadamardOverPauliGate() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]] = mqtopt.z() %[[Q0_1]] : !mqtopt.Qubit
    // CHECK: %[[Q1_1:.*]] = mqtopt.h() %[[Q1_0]] : !mqtopt.Qubit
    // CHECK: %[[Q1_2:.*]] = mqtopt.x() %[[Q1_1]] : !mqtopt.Qubit
    // CHECK: %[[Q2_1:.*]] = mqtopt.h() %[[Q2_0]] : !mqtopt.Qubit
    // CHECK: %[[Q2_2:.*]] = mqtopt.y() %[[Q2_1]] : !mqtopt.Qubit

    %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.h() %q0_1 : !mqtopt.Qubit
    %q1_1 = mqtopt.z() %q1_0 : !mqtopt.Qubit
    %q1_2 = mqtopt.h() %q1_1 : !mqtopt.Qubit
    %q2_1 = mqtopt.y() %q2_0 : !mqtopt.Qubit
    %q2_2 = mqtopt.h() %q2_1 : !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_2]]
    // CHECK: mqtopt.deallocQubit %[[Q1_2]]
    // CHECK: mqtopt.deallocQubit %[[Q2_2]]
    mqtopt.deallocQubit %q0_2
    mqtopt.deallocQubit %q1_2
    mqtopt.deallocQubit %q2_2

    return
  }
}

// -----
// This test checks that Pauli gates are not lifted over hadamard gates.
// In this example, the following liftings are being tested:
//   - Hadamard gate followed by a Pauli-X gate
//   - Hadamard gate followed by a Pauli-Y gate
//   - Hadamard gate followed by a Pauli-Z gate

module {
  func.func @testDoNotLiftPauliOverHadamardGate() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]] = mqtopt.x() %[[Q0_1]] : !mqtopt.Qubit
    // CHECK: %[[Q1_1:.*]] = mqtopt.h() %[[Q1_0]] : !mqtopt.Qubit
    // CHECK: %[[Q1_2:.*]] = mqtopt.z() %[[Q1_1]] : !mqtopt.Qubit
    // CHECK: %[[Q2_1:.*]] = mqtopt.h() %[[Q2_0]] : !mqtopt.Qubit
    // CHECK: %[[Q2_2:.*]] = mqtopt.y() %[[Q2_1]] : !mqtopt.Qubit

    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.x() %q0_1 : !mqtopt.Qubit
    %q1_1 = mqtopt.h() %q1_0 : !mqtopt.Qubit
    %q1_2 = mqtopt.z() %q1_1 : !mqtopt.Qubit
    %q2_1 = mqtopt.h() %q2_0 : !mqtopt.Qubit
    %q2_2 = mqtopt.y() %q2_1 : !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_2]]
    // CHECK: mqtopt.deallocQubit %[[Q1_2]]
    // CHECK: mqtopt.deallocQubit %[[Q2_2]]
    mqtopt.deallocQubit %q0_2
    mqtopt.deallocQubit %q1_2
    mqtopt.deallocQubit %q2_2

    return
  }
}

// =================================================== Raise over multiple Pauli gate =========================================================================
// -----
// This test checks if hadamard gates can be lifted over multiple Pauli gate.
// In this example, the following liftings are being tested:
//   - Pauli-X and Pauli-Z followed by a Hadamard gate
//   - Pauli-X, Pauli-Y and Pauli-Z followed by a Hadamard gate
//   - Pauli-X, S, Pauli-X and Pauli-Y followed by a Hadamard gate

module {
  func.func @testLiftHadamardOverPauliGate() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]] = mqtopt.z() %[[Q0_1]] : !mqtopt.Qubit
    // CHECK: %[[Q0_3:.*]] = mqtopt.x() %[[Q0_2]] : !mqtopt.Qubit
    // CHECK: %[[Q1_1:.*]] = mqtopt.h() %[[Q1_0]] : !mqtopt.Qubit
    // CHECK: %[[Q1_2:.*]] = mqtopt.z() %[[Q1_1]] : !mqtopt.Qubit
    // CHECK: %[[Q1_3:.*]] = mqtopt.y() %[[Q1_2]] : !mqtopt.Qubit
    // CHECK: %[[Q1_4:.*]] = mqtopt.x() %[[Q1_3]] : !mqtopt.Qubit
    // CHECK: %[[Q2_1:.*]] = mqtopt.x() %[[Q2_0]] : !mqtopt.Qubit
    // CHECK: %[[Q2_2:.*]] = mqtopt.s() %[[Q2_1]] : !mqtopt.Qubit
    // CHECK: %[[Q2_3:.*]] = mqtopt.h() %[[Q2_2]] : !mqtopt.Qubit
    // CHECK: %[[Q2_4:.*]] = mqtopt.z() %[[Q2_3]] : !mqtopt.Qubit
    // CHECK: %[[Q2_5:.*]] = mqtopt.y() %[[Q2_4]] : !mqtopt.Qubit

    %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.z() %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.h() %q0_2 : !mqtopt.Qubit
    %q1_1 = mqtopt.x() %q1_0 : !mqtopt.Qubit
    %q1_2 = mqtopt.y() %q1_1 : !mqtopt.Qubit
    %q1_3 = mqtopt.z() %q1_2 : !mqtopt.Qubit
    %q1_4 = mqtopt.h() %q1_3 : !mqtopt.Qubit
    %q2_1 = mqtopt.x() %q2_0 : !mqtopt.Qubit
    %q2_2 = mqtopt.s() %q2_1 : !mqtopt.Qubit
    %q2_3 = mqtopt.x() %q2_2 : !mqtopt.Qubit
    %q2_4 = mqtopt.y() %q2_3 : !mqtopt.Qubit
    %q2_5 = mqtopt.h() %q2_4 : !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_3]]
    // CHECK: mqtopt.deallocQubit %[[Q1_4]]
    // CHECK: mqtopt.deallocQubit %[[Q2_5]]
    mqtopt.deallocQubit %q0_3
    mqtopt.deallocQubit %q1_4
    mqtopt.deallocQubit %q2_5

    return
  }
}

// -----
// This test checks if hadamard gates are lifted over preceeding and not over proceeding Pauli gates.
// In this example, the following liftings are being tested:
//   - Pauli-X, Hadamard and Pauli-X gates
//   - Pauli-X, Pauli-Z, Hadamard and Pauli-Z gates

module {
  func.func @testLiftHadamardOverPauliGate() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]] = mqtopt.z() %[[Q0_1]] : !mqtopt.Qubit
    // CHECK: %[[Q0_3:.*]] = mqtopt.x() %[[Q0_2]] : !mqtopt.Qubit
    // CHECK: %[[Q1_1:.*]] = mqtopt.h() %[[Q1_0]] : !mqtopt.Qubit
    // CHECK: %[[Q1_2:.*]] = mqtopt.z() %[[Q1_1]] : !mqtopt.Qubit
    // CHECK: %[[Q1_3:.*]] = mqtopt.x() %[[Q1_2]] : !mqtopt.Qubit
    // CHECK: %[[Q1_4:.*]] = mqtopt.z() %[[Q1_3]] : !mqtopt.Qubit

    %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.h() %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.x() %q0_2 : !mqtopt.Qubit
    %q1_1 = mqtopt.x() %q1_0 : !mqtopt.Qubit
    %q1_2 = mqtopt.z() %q1_1 : !mqtopt.Qubit
    %q1_3 = mqtopt.h() %q1_2 : !mqtopt.Qubit
    %q1_4 = mqtopt.z() %q1_3 : !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_3]]
    // CHECK: mqtopt.deallocQubit %[[Q1_4]]
    mqtopt.deallocQubit %q0_3
    mqtopt.deallocQubit %q1_4

    return
  }
}

// -----
// This test checks if hadamard gates are lifted if they are controlled by the same qubit as the lifted gate is.

module {
  func.func @testLiftHadamardOverPauliGateIfControlled() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.x() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]], %[[Q1_1:.*]] = mqtopt.h() %[[Q0_1]] ctrl %[[Q1_0]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q0_3:.*]], %[[Q1_2:.*]] = mqtopt.z() %[[Q0_2]] ctrl %[[Q1_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q0_4:.*]], %[[Q1_3:.*]] = mqtopt.x() %[[Q0_3]] ctrl %[[Q1_2]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

    %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit
    %q0_2, %q1_1 = mqtopt.x() %q0_1 ctrl %q1_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_3, %q1_2 = mqtopt.h() %q0_2 ctrl %q1_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_4, %q1_3 = mqtopt.x() %q0_3 ctrl %q1_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_4]]
    // CHECK: mqtopt.deallocQubit %[[Q1_3]]
    mqtopt.deallocQubit %q0_4
    mqtopt.deallocQubit %q1_3

    return
  }
}

// -----
// This test checks that a hadamard gate is not lifted if they are controlled by a different qubit than the one lifted
// gate is.

module {
  func.func @testDoNotLiftHadamardOverPauliGateIfControlledByDifferentQubits() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]], %[[Q1_1:.*]] = mqtopt.x() %[[Q0_0]] ctrl %[[Q1_0]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]], %[[Q2_1:.*]] = mqtopt.h() %[[Q0_1]] ctrl %[[Q2_0]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q0_3:.*]] = mqtopt.z() %[[Q0_2]] : !mqtopt.Qubit
    // CHECK: %[[Q0_4:.*]], %[[Q1_2:.*]] = mqtopt.h() %[[Q0_3]] ctrl %[[Q1_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_1, %q1_1 = mqtopt.x() %q0_0 ctrl %q1_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_2, %q2_1 = mqtopt.h() %q0_1 ctrl %q2_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_3 = mqtopt.z() %q0_2 : !mqtopt.Qubit
    %q0_4, %q1_2 = mqtopt.h() %q0_3 ctrl %q1_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_4]]
    // CHECK: mqtopt.deallocQubit %[[Q1_2]]
    // CHECK: mqtopt.deallocQubit %[[Q2_1]]
    mqtopt.deallocQubit %q0_4
    mqtopt.deallocQubit %q1_2
    mqtopt.deallocQubit %q2_1

    return
  }
}

// -----
// This test checks that a controlled hadamard gate is not lifted if there is another gate between the controls of
// the Pauli and the hadamard gate.
module {
  func.func @testDoNotLiftHadamardOverPauliGateIfGateBetweenControls() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]], %[[Q1_1:.*]] = mqtopt.z() %[[Q0_0]] ctrl %[[Q1_0]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q1_2:.*]] = mqtopt.s() %[[Q1_1]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]], %[[Q1_3:.*]] = mqtopt.h() %[[Q0_1]] ctrl %[[Q1_2]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_1, %q1_1 = mqtopt.z() %q0_0 ctrl %q1_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_2 = mqtopt.s() %q1_1 : !mqtopt.Qubit
    %q0_2, %q1_3 = mqtopt.h() %q0_1 ctrl %q1_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_2]]
    // CHECK: mqtopt.deallocQubit %[[Q1_3]]
    mqtopt.deallocQubit %q0_2
    mqtopt.deallocQubit %q1_3

    return
  }
}

// -----
// This test checks that a hadamard gate is not lifted if they do not share all controls with the Pauli gate.
module {
  func.func @testDoNotLiftHadamardOverPauliGateWithDifferentControls() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]], %[[Q12_1:.*]]:2 = mqtopt.z() %[[Q0_0]] ctrl %[[Q1_0]], %[[Q2_0]] : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]], %[[Q1_2:.*]] = mqtopt.h() %[[Q0_1]] ctrl %[[Q12_1]]#0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_1, %q1_1, %q2_1 = mqtopt.z() %q0_0 ctrl %q1_0, %q2_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    %q0_2, %q1_2 = mqtopt.h() %q0_1 ctrl %q1_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_2]]
    // CHECK: mqtopt.deallocQubit %[[Q1_2]]
    // CHECK: mqtopt.deallocQubit %[[Q12_1]]#1
    mqtopt.deallocQubit %q0_2
    mqtopt.deallocQubit %q1_2
    mqtopt.deallocQubit %q2_1

    return
  }
}

// -----
// This test checks that a hadamard gate can be lifted over a controlled Pauli Z gate even if the targets are at
// different places.
module {
  func.func @testDoLiftHadamardOverControlledPauliZ() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    // CHECK: %[[Q1_1:.*]], %[[Q02_1:.*]]:2 = mqtopt.h() %[[Q1_0]] ctrl %[[Q0_0]], %[[Q2_0]] : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[Q1_2:.*]], %[[Q02_2:.*]]:2 = mqtopt.x() %[[Q1_1]] ctrl %[[Q02_1]]#0, %[[Q02_1]]#1 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[Q0_3:.*]], %[[Q12_3:.*]]:2 = mqtopt.z() %[[Q02_2]]#0 nctrl %[[Q1_2]], %[[Q02_2]]#1 : !mqtopt.Qubit nctrl !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[Q0_4:.*]], %[[Q12_4:.*]]:2 = mqtopt.h() %[[Q12_3]]#0 nctrl %[[Q0_3]], %[[Q12_3]]#1 : !mqtopt.Qubit nctrl !mqtopt.Qubit, !mqtopt.Qubit
    %q0_1, %q1_1, %q2_1 = mqtopt.z() %q0_0 ctrl %q1_0, %q2_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    %q1_2, %q0_2, %q2_2 = mqtopt.h() %q1_1 ctrl %q0_1, %q2_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    %q0_3, %q1_3, %q2_3 = mqtopt.z() %q0_2 nctrl %q1_2, %q2_2 : !mqtopt.Qubit nctrl !mqtopt.Qubit, !mqtopt.Qubit
    %q1_4, %q0_4, %q2_4 = mqtopt.h() %q1_3 nctrl %q0_3, %q2_3 : !mqtopt.Qubit nctrl !mqtopt.Qubit, !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q12_4]]#0
    // CHECK: mqtopt.deallocQubit %[[Q0_4]]
    // CHECK: mqtopt.deallocQubit %[[Q12_4]]#1
    mqtopt.deallocQubit %q0_4
    mqtopt.deallocQubit %q1_4
    mqtopt.deallocQubit %q2_4

    return
  }
}

// -----
// This test checks that a hadamard gate is lifted over a Pauli gate if the negatively and positively controlled gates
// are exactly equal.

module {
  func.func @testLiftHadamardOverPauliGateIfControlsFit() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]], %[[Q1_1:.*]], %[[Q2_1:.*]] = mqtopt.x() %[[Q0_0]] ctrl %[[Q1_0]] nctrl %[[Q2_0]] : !mqtopt.Qubit ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]], %[[Q2_2:.*]], %[[Q1_2:.*]] = mqtopt.h() %[[Q0_1]] ctrl %[[Q2_1]] nctrl %[[Q1_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit
    // CHECK: %[[Q0_3:.*]], %[[Q1_3:.*]], %[[Q2_3:.*]] = mqtopt.h() %[[Q0_2]] ctrl %[[Q1_2]] nctrl %[[Q2_2]] : !mqtopt.Qubit ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit
    // CHECK: %[[Q0_4:.*]], %[[Q1_4:.*]], %[[Q2_4:.*]] = mqtopt.x() %[[Q0_3]] ctrl %[[Q1_3]] nctrl %[[Q2_3]] : !mqtopt.Qubit ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit
    %q0_1, %q1_1, %q2_1 = mqtopt.x() %q0_0 ctrl %q1_0 nctrl %q2_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit
    %q0_2, %q2_2, %q1_2 = mqtopt.h() %q0_1 ctrl %q2_1 nctrl %q1_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit
    %q0_3, %q1_3, %q2_3 = mqtopt.z() %q0_2 ctrl %q1_2 nctrl %q2_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit
    %q0_4, %q1_4, %q2_4 = mqtopt.h() %q0_3 ctrl %q1_3 nctrl %q2_3 : !mqtopt.Qubit ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_4]]
    // CHECK: mqtopt.deallocQubit %[[Q1_4]]
    // CHECK: mqtopt.deallocQubit %[[Q2_4]]
    mqtopt.deallocQubit %q0_4
    mqtopt.deallocQubit %q1_4
    mqtopt.deallocQubit %q2_4

    return
  }
}
