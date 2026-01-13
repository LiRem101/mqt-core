// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --constant-propagation | FileCheck %s

// -----
// This test checks if CNOTs or the controls of CNOTs are removed if we can classically determine the ctrls value.
module {
  func.func @testReducePosCtrls() attributes {passthrough = ["entry_point"]} {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q3_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit
    %q3_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]] = mqtopt.x() %[[Q0_1]] : !mqtopt.Qubit
    // CHECK: %[[Q0_3:.*]] = mqtopt.h() %[[Q0_2]] : !mqtopt.Qubit

    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.x() %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.h() %q0_2 : !mqtopt.Qubit
    %q1_1, %q0_4 = mqtopt.x() %q1_0 ctrl %q0_3 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // CHECK: %[[Q2_1:.*]] = mqtopt.h() %[[Q2_0]] : !mqtopt.Qubit
    // CHECK: %[[Q2_2:.*]] = mqtopt.z() %[[Q2_1]] : !mqtopt.Qubit
    // CHECK: %[[Q2_3:.*]] = mqtopt.h() %[[Q2_2]] : !mqtopt.Qubit
    // CHECK: %[[Q3_1:.*]] = mqtopt.rx(static [-3.926991e-01]) %[[Q3_0]] : !mqtopt.Qubit
    // CHECK: %[[c0:.*]] = arith.constant 3.000000e-01 : f64
    // CHECK: %[[Q3_2:.*]] = mqtopt.ry(%[[c0]]) %[[Q3_1]] : !mqtopt.Qubit

    %q2_1 = mqtopt.h() %q2_0 : !mqtopt.Qubit
    %q2_2 = mqtopt.z() %q2_1 : !mqtopt.Qubit
    %q2_3 = mqtopt.h() %q2_2 : !mqtopt.Qubit
    %q3_1, %q2_4 = mqtopt.rx(static [-3.926991e-01]) %q3_0 ctrl %q2_3 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %c0_f64 = arith.constant 3.000000e-01 : f64
    %q3_2, %q2_5 = mqtopt.ry(%c0_f64) %q3_1 ctrl %q2_4 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_3]]
    // CHECK: mqtopt.deallocQubit %[[Q1_0]]
    // CHECK: mqtopt.deallocQubit %[[Q2_3]]
    // CHECK: mqtopt.deallocQubit %[[Q3_2]]
    mqtopt.deallocQubit %q0_4
    mqtopt.deallocQubit %q1_1
    mqtopt.deallocQubit %q2_5
    mqtopt.deallocQubit %q3_2

    return
  }
}

// -----
// This test checks if CNOTs or the controls of CNOTs are removed if we can classically determine the neg ctrls value.
module {
  func.func @testReduceNegCtrls() attributes {passthrough = ["entry_point"]} {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q3_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit
    %q3_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]] = mqtopt.x() %[[Q0_1]] : !mqtopt.Qubit
    // CHECK: %[[Q0_3:.*]] = mqtopt.h() %[[Q0_2]] : !mqtopt.Qubit
    // CHECK: %[[Q1_1:.*]] = mqtopt.x() %[[Q1_0]] : !mqtopt.Qubit

    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.x() %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.h() %q0_2 : !mqtopt.Qubit
    %q1_1, %q0_4 = mqtopt.x() %q1_0 nctrl %q0_3 : !mqtopt.Qubit nctrl !mqtopt.Qubit

    // CHECK: %[[Q2_1:.*]] = mqtopt.h() %[[Q2_0]] : !mqtopt.Qubit
    // CHECK: %[[Q2_2:.*]] = mqtopt.z() %[[Q2_1]] : !mqtopt.Qubit
    // CHECK: %[[Q2_3:.*]] = mqtopt.h() %[[Q2_2]] : !mqtopt.Qubit

    %q2_1 = mqtopt.h() %q2_0 : !mqtopt.Qubit
    %q2_2 = mqtopt.z() %q2_1 : !mqtopt.Qubit
    %q2_3 = mqtopt.h() %q2_2 : !mqtopt.Qubit
    %q3_1, %q2_4 = mqtopt.x() %q3_0 nctrl %q2_3 : !mqtopt.Qubit nctrl !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_3]]
    // CHECK: mqtopt.deallocQubit %[[Q1_1]]
    // CHECK: mqtopt.deallocQubit %[[Q2_3]]
    // CHECK: mqtopt.deallocQubit %[[Q3_0]]
    mqtopt.deallocQubit %q0_4
    mqtopt.deallocQubit %q1_1
    mqtopt.deallocQubit %q2_4
    mqtopt.deallocQubit %q3_1

    return
  }
}

// -----
// This test checks that CNOTs are not changed if the target is not in |0> or |1>.
module {
  func.func @testDontRemoveIfTargetInSuperposition() attributes {passthrough = ["entry_point"]} {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q1_1:.*]], %[[Q0_2:.*]] = mqtopt.x() %[[Q1_0]] ctrl %[[Q0_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q1_1, %q0_2 = mqtopt.x() %q1_0 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_2]]
    // CHECK: mqtopt.deallocQubit %[[Q1_1]]
    mqtopt.deallocQubit %q0_2
    mqtopt.deallocQubit %q1_1

    return
  }
}

// -----
// This test checks that implied Qubits are removed from a controlled gate.
module {
  func.func @testRemoveImpliedQubits() attributes {passthrough = ["entry_point"]} {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q3_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit
    %q3_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q1_1:.*]] = mqtopt.h() %[[Q1_0]] : !mqtopt.Qubit
    // CHECK: %[[Q2_1:.*]], %[[Q12_2:.*]]:2 = mqtopt.x() %[[Q2_0]] ctrl %[[Q0_1]], %[[Q1_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[Q3_1:.*]], %[[Q2_2:.*]] = mqtopt.x() %[[Q3_0]] ctrl %[[Q2_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q1_1 = mqtopt.h() %q1_0 : !mqtopt.Qubit
    %q2_1, %q0_2, %q1_2 = mqtopt.x() %q2_0 ctrl %q0_1, %q1_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    %q3_1, %q0_3, %q1_3, %q2_2 = mqtopt.x() %q3_0 ctrl %q0_2, %q1_2, %q2_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit, !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q12_2]]#0
    // CHECK: mqtopt.deallocQubit %[[Q12_2]]#1
    // CHECK: mqtopt.deallocQubit %[[Q2_2]]
    // CHECK: mqtopt.deallocQubit %[[Q3_1]]
    mqtopt.deallocQubit %q0_3
    mqtopt.deallocQubit %q1_3
    mqtopt.deallocQubit %q2_2
    mqtopt.deallocQubit %q3_1

    return
  }
}

// -----
// This test checks that gates whose quantum controls cannot be satisfied are removed.
module {
  func.func @testUnsatisfiableQuantumCombination() attributes {passthrough = ["entry_point"]} {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q1_1:.*]] = mqtopt.x() %[[Q1_0]] : !mqtopt.Qubit
    // CHECK: %[[Q1_2:.*]], %[[Q0_2:.*]] = mqtopt.x() %[[Q1_1]] ctrl %[[Q0_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q1_1 = mqtopt.x() %q1_0 : !mqtopt.Qubit
    %q1_2, %q0_2 = mqtopt.x() %q1_1 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q2_1, %q0_3, %q1_3 = mqtopt.s() %q2_0 ctrl %q0_2, %q1_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_2]]
    // CHECK: mqtopt.deallocQubit %[[Q1_2]]
    // CHECK: mqtopt.deallocQubit %[[Q3_0]]
    mqtopt.deallocQubit %q0_3
    mqtopt.deallocQubit %q1_3
    mqtopt.deallocQubit %q2_1

    return
  }
}

// -----
// This test checks that gates whose quantum and classical controls cannot be satisfied are removed.
module {
  func.func @testUnsatisfiableHybridCombination() attributes {passthrough = ["entry_point"]} {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q1_1:.*]] = mqtopt.x() %[[Q1_0]] : !mqtopt.Qubit
    // CHECK: %[[Q1_2:.*]], %[[Q0_2:.*]] = mqtopt.x() %[[Q1_1]] ctrl %[[Q0_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q0_3:.*]], %[[c0:.*]] = mqtopt.measure %[[Q0_2]]
    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q1_1 = mqtopt.x() %q1_0 : !mqtopt.Qubit
    %q1_2, %q0_2 = mqtopt.x() %q1_1 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_3, %c0 = mqtopt.measure %q0_2
    %q0_4, %q1_3 = scf.if %c0 -> (!mqtopt.Qubit, !mqtopt.Qubit) {
        %q0_3_if, %q1_2_if = mqtopt.x() %q0_3 ctrl %q1_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        scf.yield %q0_3_if, %q1_2_if  : !mqtopt.Qubit, !mqtopt.Qubit
    } else {
        scf.yield %q0_3, %q1_2 : !mqtopt.Qubit, !mqtopt.Qubit
    }

    // CHECK: mqtopt.deallocQubit %[[Q0_3]]
    // CHECK: mqtopt.deallocQubit %[[Q1_2]]
    mqtopt.deallocQubit %q0_4
    mqtopt.deallocQubit %q1_3

    return
  }
}

// -----
// This test checks that gates are unconditionally applied if the bit they depend on is always zero.
module {
  func.func @testRemoveClassicalConditionalIfItsZero() attributes {passthrough = ["entry_point"]} {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]] = mqtopt.h() %[[Q0_1]] : !mqtopt.Qubit
    // CHECK: %[[Q0_3:.*]], %[[c0:.*]] = mqtopt.measure %[[Q0_2]]
    // CHECK: %[[Q0_4:.*]] = mqtopt.h() %[[Q0_3]] : !mqtopt.Qubit
    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.h() %q0_1 : !mqtopt.Qubit
    %q0_3, %c0 = mqtopt.measure %q0_2
    %q0_4 = scf.if %c0 -> (!mqtopt.Qubit) {
        %q0_3_if = mqtopt.x() %q0_3 : !mqtopt.Qubit
        scf.yield %q0_3_if  : !mqtopt.Qubit
    } else {
        %q0_3_else = mqtopt.h() %q0_3 : !mqtopt.Qubit
        scf.yield %q0_3_else : !mqtopt.Qubit
    }

    // CHECK: mqtopt.deallocQubit %[[Q0_4]]
    mqtopt.deallocQubit %q0_4

    return
  }
}

// -----
// This test checks that gates are unconditionally applied if the bit they depend on is always one.
module {
  func.func @testRemoveClassicalConditionalIfItsOne() attributes {passthrough = ["entry_point"]} {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.x() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]], %[[c0:.*]] = mqtopt.measure %[[Q0_1]]
    // CHECK: %[[Q0_3:.*]] = mqtopt.x() %[[Q0_2]] : !mqtopt.Qubit
    %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit
    %q0_2, %c0 = mqtopt.measure %q0_1
    %q0_3 = scf.if %c0 -> (!mqtopt.Qubit) {
        %q0_2_if = mqtopt.x() %q0_2 : !mqtopt.Qubit
        scf.yield %q0_2_if  : !mqtopt.Qubit
    } else {
        %q0_2_else = mqtopt.h() %q0_2 : !mqtopt.Qubit
        scf.yield %q0_2_else : !mqtopt.Qubit
    }

    // CHECK: mqtopt.deallocQubit %[[Q0_3]]
    mqtopt.deallocQubit %q0_3

    return
  }
}

// -----
// This test checks that conditionals are not changed if we cannot tell the bits value.
module {
  func.func @testDoNotRemoveClassicalConditional() attributes {passthrough = ["entry_point"]} {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]], %[[c0:.*]] = mqtopt.measure %[[Q0_1]]
    // CHECK: %[[Q0_3:.*]] = scf.if %[[c0]] -> (!mqtopt.Qubit) {
    // CHECK:     %[[Q0_2_if:.*]] = mqtopt.x() %[[Q0_2]] : !mqtopt.Qubit
    // CHECK:     scf.yield %[[Q0_2_if]] : !mqtopt.Qubit
    // CHECK: } else {
    // CHECK:     %[[Q0_2_else:.*]] = mqtopt.h() %[[Q0_2]] : !mqtopt.Qubit
    // CHECK:     scf.yield %[[Q0_2_else]] : !mqtopt.Qubit
    // CHECK: }
    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q0_2, %c0 = mqtopt.measure %q0_1
    %q0_3 = scf.if %c0 -> (!mqtopt.Qubit) {
        %q0_2_if = mqtopt.x() %q0_2 : !mqtopt.Qubit
        scf.yield %q0_2_if  : !mqtopt.Qubit
    } else {
        %q0_2_else = mqtopt.h() %q0_2 : !mqtopt.Qubit
        scf.yield %q0_2_else : !mqtopt.Qubit
    }

    // CHECK: mqtopt.deallocQubit %[[Q0_3]]
    mqtopt.deallocQubit %q0_3

    return
  }
}

// -----
// This test checks that a quantum conditional is replaced by a classical if a qubit and a classical bit are equivalent.
module {
  func.func @testEquivalentPositiveClassicalAndQuantumControl() attributes {passthrough = ["entry_point"]} {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q1_1:.*]], %[[Q0_2:.*]] = mqtopt.x() %[[Q1_0]] ctrl %[[Q0_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q0_3:.*]], %[[c0:.*]] = mqtopt.measure %[[Q0_2]]
    // CHECK: %[[Q2_1:.*]] = scf.if %[[c0]] -> (!mqtopt.Qubit) {
    // CHECK:     %[[Q2_0_if:.*]] = mqtopt.x() %[[Q2_0]] : !mqtopt.Qubit
    // CHECK:     scf.yield %[[Q2_0_if]] : !mqtopt.Qubit
    // CHECK: } else {
    // CHECK:     scf.yield %[[Q2_0]] : !mqtopt.Qubit
    // CHECK: }
    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q1_1, %q0_2 = mqtopt.x() %q1_0 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_3, %c0 = mqtopt.measure %q0_2
    %q2_1, %q1_2 = mqtopt.x() %q2_0 ctrl %q1_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_3]]
    // CHECK: mqtopt.deallocQubit %[[Q1_1]]
    // CHECK: mqtopt.deallocQubit %[[Q2_1]]
    mqtopt.deallocQubit %q0_3
    mqtopt.deallocQubit %q1_2
    mqtopt.deallocQubit %q2_1

    return
  }
}

// -----
// This test checks that multiple quantum conditionals are replaced by a classical if a qubit and a classical bit are
// equivalent.
module {
  func.func @testEquivalentClassicalAndQuantumControl() attributes {passthrough = ["entry_point"]} {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q1_1:.*]], %[[Q0_2:.*]] = mqtopt.x() %[[Q1_0]] ctrl %[[Q0_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q0_3:.*]], %[[c0:.*]] = mqtopt.measure %[[Q0_2]]
    // CHECK: %[[Q2_1:.*]] = scf.if %[[c0]] -> (!mqtopt.Qubit) {
    // CHECK:     %[[Q2_0_if:.*]] = mqtopt.x() %[[Q2_0]] : !mqtopt.Qubit
    // CHECK:     scf.yield %[[Q2_0_if]] : !mqtopt.Qubit
    // CHECK: } else {
    // CHECK:     %[[Q2_0_else:.*]] = mqtopt.y() %[[Q2_0]] : !mqtopt.Qubit
    // CHECK:     scf.yield %[[Q2_0_else]] : !mqtopt.Qubit
    // CHECK: }
    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q1_1, %q0_2 = mqtopt.x() %q1_0 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_3, %c0 = mqtopt.measure %q0_2
    %q2_1, %q1_2 = mqtopt.x() %q2_0 ctrl %q1_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q2_2, %q1_3 = mqtopt.y() %q2_1 nctrl %q1_2 : !mqtopt.Qubit nctrl !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_3]]
    // CHECK: mqtopt.deallocQubit %[[Q1_1]]
    // CHECK: mqtopt.deallocQubit %[[Q2_1]]
    mqtopt.deallocQubit %q0_3
    mqtopt.deallocQubit %q1_3
    mqtopt.deallocQubit %q2_2

    return
  }
}

// -----
// This test checks if a classical control is removed if the quantum control implies the classical one.
module {
  func.func @testQuantumImpliesClassical() attributes {passthrough = ["entry_point"]} {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]], %[[c0:.*]] = mqtopt.measure %[[Q0_1]]
    // CHECK: %[[Q1_1:.*]], %[[Q0_3:.*]] = mqtopt.h() %[[Q1_0]] ctrl %[[Q0_2]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q0_4:.*]], %[[Q1_2:.*]] = mqtopt.x() %[[Q0_3]] ctrl %[[Q1_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q0_2, %c0 = mqtopt.measure %q0_1
    %q1_1, %q0_3 = mqtopt.h() %q1_0 ctrl %q0_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_4, %q1_2 = scf.if %c0 -> (!mqtopt.Qubit, !mqtopt.Qubit) {
        %q0_3_if, %q1_1_if = mqtopt.x() %q0_3 ctrl %q1_1, : !mqtopt.Qubit ctrl !mqtopt.Qubit
        scf.yield %q0_3_if, %q1_1_if  : !mqtopt.Qubit, !mqtopt.Qubit
    } else {
        scf.yield %q0_3, %q1_1 : !mqtopt.Qubit, !mqtopt.Qubit
    }

    // CHECK: mqtopt.deallocQubit %[[Q0_4]]
    // CHECK: mqtopt.deallocQubit %[[Q1_2]]
    mqtopt.deallocQubit %q0_4
    mqtopt.deallocQubit %q1_2

    return
  }
}

// -----
// This test checks if a quantum control is removed if the classical control implies the quantum one.
module {
  func.func @testClassicalImpliesQuantum() attributes {passthrough = ["entry_point"]} {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]], %[[c0:.*]] = mqtopt.measure %[[Q0_1]]
    // CHECK: %[[Q1_1:.*]], %[[Q0_3:.*]] = mqtopt.x() %[[Q1_0]] nctrl %[[Q0_2]] : !mqtopt.Qubit nctrl !mqtopt.Qubit
    // CHECK: %[[Q0_4:.*]], %[[Q1_2:.*]] = mqtopt.h() %[[Q0_3]] ctrl %[[Q1_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q0_5:.*]], %[[Q1_3:.*]] = scf.if %[[c0]] -> (!mqtopt.Qubit, !mqtopt.Qubit) {
    // CHECK:     %[[Q1_2_if:.*]] = mqtopt.x() %[[Q1_2]] : !mqtopt.Qubit
    // CHECK:     scf.yield %[[Q0_4]], %[[Q1_2_if]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: } else {
    // CHECK:     scf.yield %[[Q0_4]], %[[Q1_2]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: }
    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q0_2, %c0 = mqtopt.measure %q0_1
    %q1_1, %q0_3 = mqtopt.x() %q1_0 nctrl %q0_2 : !mqtopt.Qubit nctrl !mqtopt.Qubit
    %q0_4, %q1_2 = mqtopt.h() %q0_3 ctrl %q1_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_5, %q1_3 = scf.if %c0 -> (!mqtopt.Qubit, !mqtopt.Qubit) {
        %q1_2_if, %q0_4_if = mqtopt.x() %q1_2 ctrl %q0_4 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        scf.yield %q0_4_if, %q1_2_if  : !mqtopt.Qubit, !mqtopt.Qubit
    } else {
        scf.yield %q0_4, %q1_2 : !mqtopt.Qubit, !mqtopt.Qubit
    }

    // CHECK: mqtopt.deallocQubit %[[Q0_5]]
    // CHECK: mqtopt.deallocQubit %[[Q1_3]]
    mqtopt.deallocQubit %q0_5
    mqtopt.deallocQubit %q1_3

    return
  }
}

// -----
// This test checks if a classical neg control is removed if it is implied by a quantum control.
module {
  func.func @testQuantumImpliesNegClassical() attributes {passthrough = ["entry_point"]} {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]], %[[c0:.*]] = mqtopt.measure %[[Q0_1]]
    // CHECK: %[[Q0_3:.*]] = mqtopt.x() %[[Q0_2]] : !mqtopt.Qubit
    // CHECK: %[[Q1_1:.*]], %[[Q0_4:.*]] = mqtopt.h() %[[Q1_0]] ctrl %[[Q0_3]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q0_5:.*]], %[[Q1_2:.*]] = mqtopt.x() %[[Q0_4]] ctrl %[[Q1_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q0_2, %c0 = mqtopt.measure %q0_1
    %q0_3 = mqtopt.x() %q0_2 : !mqtopt.Qubit
    %q1_1, %q0_4 = mqtopt.h() %q1_0 ctrl %q0_3 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_5, %q1_2 = scf.if %c0 -> (!mqtopt.Qubit, !mqtopt.Qubit) {
        scf.yield %q0_4, %q1_1  : !mqtopt.Qubit, !mqtopt.Qubit
    } else {
        %q0_4_else, %q1_1_else = mqtopt.x() %q0_4 ctrl %q1_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        scf.yield %q0_4_else, %q1_1_else : !mqtopt.Qubit, !mqtopt.Qubit
    }

    // CHECK: mqtopt.deallocQubit %[[Q0_5]]
    // CHECK: mqtopt.deallocQubit %[[Q1_2]]
    mqtopt.deallocQubit %q0_5
    mqtopt.deallocQubit %q1_2

    return
  }
}

// -----
// This test checks if a quantum neg control is removed if it is implied by a classical control.
module {
  func.func @testClassicalImpliesNegQuantum() attributes {passthrough = ["entry_point"]} {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]], %[[c0:.*]] = mqtopt.measure %[[Q0_1]]
    // CHECK: %[[Q0_3:.*]] = mqtopt.x() %[[Q0_2]] : !mqtopt.Qubit
    // CHECK: %[[Q1_1:.*]], %[[Q0_4:.*]] = mqtopt.h() %[[Q1_0]] ctrl %[[Q0_3]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q0_5:.*]], %[[Q1_2:.*]] = scf.if %[[c0]] -> (!mqtopt.Qubit) {
    // CHECK:     %[[Q0_4_if:.*]] = mqtopt.x() %[[Q0_4]] : !mqtopt.Qubit
    // CHECK:     scf.yield %[[Q0_4_if]], %[[Q1_1]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: } else {
    // CHECK:     scf.yield %[[Q0_4]], %[[Q1_1]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: }
    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q0_2, %c0 = mqtopt.measure %q0_1
    %q0_3 = mqtopt.x() %q0_2 : !mqtopt.Qubit
    %q1_1, %q0_4 = mqtopt.h() %q1_0 ctrl %q0_3 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_5, %q1_2 = scf.if %c0 -> (!mqtopt.Qubit, !mqtopt.Qubit) {
        %q0_4_if, %q1_1_if = mqtopt.x() %q0_4 nctrl %q1_1 : !mqtopt.Qubit nctrl !mqtopt.Qubit
        scf.yield %q0_4_if, %q1_1_if  : !mqtopt.Qubit, !mqtopt.Qubit
    } else {
        scf.yield %q0_4, %q1_1 : !mqtopt.Qubit, !mqtopt.Qubit
    }

    // CHECK: mqtopt.deallocQubit %[[Q0_5]]
    // CHECK: mqtopt.deallocQubit %[[Q1_2]]
    mqtopt.deallocQubit %q0_5
    mqtopt.deallocQubit %q1_2

    return
  }
}

// -----
// This test checks if a phase gate is removed if it only adds a global phase.
module {
  func.func @testRemoveSingleQubitPhaseGate() attributes {passthrough = ["entry_point"]} {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]] = mqtopt.z() %[[Q0_1]] : !mqtopt.Qubit
    // CHECK: %[[Q0_3:.*]] = mqtopt.h() %[[Q0_2]] : !mqtopt.Qubit
    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.z() %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.h() %q0_2 : !mtopt.Qubit
    %q0_4 = mqtopt.z() %q0_3 : !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_3]]
    mqtopt.deallocQubit %q0_4

    return
  }
}

// -----
// This test checks if a multi-qubit phase gate is removed if it only adds a global phase.
module {
  func.func @testRemoveMultiQubitPhaseGate() attributes {passthrough = ["entry_point"]} {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q1_1:.*]], %[[Q0_2:.*]] = mqtopt.x() %[[Q1_0]] ctrl %[[Q0_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q0_3:.*]], %[[c0:.*]] = mqtopt.measure %[[Q0_2]]
    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q1_1, %q0_2 = mqtopt.x() %q1_0 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_3, %c0 = mqtopt.measure %q0_2
    %q1_2, %q0_4 = mqtopt.z() %q1_1 ctrl %q0_3 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_3]]
    // CHECK: mqtopt.deallocQubit %[[Q1_1]]
    mqtopt.deallocQubit %q0_4
    mqtopt.deallocQubit %q1_2

    return
  }
}
