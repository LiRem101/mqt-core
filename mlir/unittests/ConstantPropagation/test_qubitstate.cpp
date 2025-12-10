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
              testing::HasSubstr("|0> -> 0.70, |1> -> 0.70"));
}

TEST_F(QubitStateTest, ApplyHGateToThirdQubit) {
  QubitState qState = QubitState(4, 4);
  auto nextState = qState.propagateGate(qc::H, {2});

  EXPECT_THAT(nextState.toString(),
              testing::HasSubstr("|0000> -> 0.70, |0100> -> 0.70"));
}

TEST_F(QubitStateTest, ApplyHHGateToThirdQubit) {
  QubitState qState = QubitState(4, 4);
  auto nextState = qState.propagateGate(qc::H, {2});
  nextState = qState.propagateGate(qc::H, {2});

  EXPECT_THAT(nextState.toString(), testing::HasSubstr("|0000> -> 1"));
}
