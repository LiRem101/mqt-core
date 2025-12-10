/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/Transforms/ConstantPropagation/RewriteChecker.hpp"
#include "mlir/Dialect/MQTOpt/Transforms/ConstantPropagation/UnionTable.hpp"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <mlir/Dialect/MQTOpt/Transforms/Passes.h>

using namespace mqt::ir::opt::qcp;

class QubitStateTest : public ::testing::Test {
protected:
  void SetUp() override {}

  void TearDown() override {}
};

// ##################################################
// # Helper functions
// ##################################################

// ##################################################
// # Basic tests
// ##################################################

TEST_F(QubitStateTest, ApplyHGate) {
  QubitState qState = QubitState(1, 4);
  QubitState nextState = qState.propagateGate(qc::H, {0});

  EXPECT_THAT(nextState.toString(),
              testing::HasSubstr("|0> -> 0.71, |1> -> 0.71"));
}

TEST_F(QubitStateTest, ApplyHGateToThirdQubit) {
  QubitState qState = QubitState(4, 4);
  auto nextState = qState.propagateGate(qc::H, {2});

  EXPECT_THAT(nextState.toString(),
              testing::HasSubstr("|0000> -> 0.71, |0100> -> 0.71"));
}

TEST_F(QubitStateTest, ApplyHHGateToThirdQubit) {
  QubitState qState = QubitState(4, 4);
  auto nextState = qState.propagateGate(qc::H, {2});
  nextState = qState.propagateGate(qc::H, {2});

  EXPECT_THAT(nextState.toString(), testing::HasSubstr("|0000> -> 1"));
}

TEST_F(QubitStateTest, ApplyHZGateToThirdQubit) {
  QubitState qState = QubitState(4, 4);
  qState.propagateGate(qc::H, {2});
  qState.propagateGate(qc::Z, {2});

  EXPECT_THAT(qState.toString(),
              testing::HasSubstr("|0000> -> 0.71, |0100> -> -0.71"));
}

TEST_F(QubitStateTest, ApplyHZHGateToThirdQubit) {
  QubitState qState = QubitState(4, 4);
  qState.propagateGate(qc::H, {2});
  qState.propagateGate(qc::Z, {2});
  qState.propagateGate(qc::H, {2});

  EXPECT_THAT(qState.toString(), testing::HasSubstr("|0100> -> 1"));
}

TEST_F(QubitStateTest, ApplyHGatesToTwoQubits) {
  QubitState qState = QubitState(4, 4);
  qState.propagateGate(qc::H, {2});
  qState.propagateGate(qc::X, {0});

  EXPECT_THAT(qState.toString(),
              testing::HasSubstr("|0001> -> 0.71, |0101> -> 0.71"));
}

TEST_F(QubitStateTest, ApplyParametrizedGateToThirdQubit) {
  QubitState qState = QubitState(4, 4);
  qState.propagateGate(qc::H, {2});
  qState.propagateGate(qc::U, {2}, {}, {}, {1, 0.5, 2});

  EXPECT_THAT(
      qState.toString(),
      testing::HasSubstr("|0000> -> 0.76 - i0.31, |0100> -> -0.20 + i0.53"));
}

TEST_F(QubitStateTest, ApplyTwoQubitGate) {
  QubitState qState = QubitState(4, 4);
  qState.propagateGate(qc::H, {1});
  qState.propagateGate(qc::S, {1});
  qState.propagateGate(qc::H, {2});
  qState.propagateGate(qc::Tdg, {2});
  qState.propagateGate(qc::Peres, {2, 1}, {}, {}, {});

  EXPECT_THAT(
      qState.toString(),
      testing::HasSubstr("|0000> -> 0.35 - i0.35, |0010> -> 0.35 + i0.35, "
                         "|0100> -> 0.00 + i0.50, |0110> -> 0.50"));
}

TEST_F(QubitStateTest, ApplyTwoQubitGateReversedOrd) {
  QubitState qState = QubitState(4, 4);
  qState.propagateGate(qc::H, {1});
  qState.propagateGate(qc::S, {1});
  qState.propagateGate(qc::H, {2});
  qState.propagateGate(qc::Tdg, {2});
  qState.propagateGate(qc::Peres, {1, 2}, {}, {}, {});

  EXPECT_THAT(
      qState.toString(),
      testing::HasSubstr("|0000> -> 0.00 + i0.50, |0010> -> 0.35 - i0.35, "
                         "|0100> -> 0.35 + i0.35, |0110> -> 0.50"));
}