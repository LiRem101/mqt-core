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
  qState.propagateGate(qc::H, {0});

  EXPECT_THAT(qState.toString(),
              testing::HasSubstr("|0> -> 0.71, |1> -> 0.71"));
}

TEST_F(QubitStateTest, ApplyHGateToThirdQubit) {
  QubitState qState = QubitState(4, 4);
  qState.propagateGate(qc::H, {2});

  EXPECT_THAT(qState.toString(),
              testing::HasSubstr("|0000> -> 0.71, |0100> -> 0.71"));
}

TEST_F(QubitStateTest, ApplyHHGateToThirdQubit) {
  QubitState qState = QubitState(4, 4);
  qState.propagateGate(qc::H, {2});
  qState.propagateGate(qc::H, {2});

  EXPECT_THAT(qState.toString(), testing::HasSubstr("|0000> -> 1"));
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
      testing::HasSubstr("|0000> -> 0.35 + i0.35, |0010> -> 0.35 - i0.35, "
                         "|0100> -> 0.50, |0110> -> 0.00 + i0.50"));
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
      testing::HasSubstr("|0000> -> 0.35 + i0.35, |0010> -> 0.50, "
                         "|0100> -> 0.00 + i0.50, |0110> -> 0.35 - i0.35"));
}

TEST_F(QubitStateTest, ApplySwapGate) {
  QubitState qState = QubitState(4, 4);
  qState.propagateGate(qc::H, {1});
  qState.propagateGate(qc::SWAP, {1, 3});

  EXPECT_THAT(qState.toString(),
              testing::HasSubstr("|0000> -> 0.71, |1000> -> 0.71"));
}

TEST_F(QubitStateTest, ApplyControlledGate) {
  QubitState qState = QubitState(4, 4);
  qState.propagateGate(qc::H, {1});
  qState.propagateGate(qc::X, {3});
  qState.propagateGate(qc::X, {3}, {1});

  EXPECT_THAT(qState.toString(),
              testing::HasSubstr("|0010> -> 0.71, |1000> -> 0.71"));
}

TEST_F(QubitStateTest, ApplyNegControlledGate) {
  QubitState qState = QubitState(4, 4);
  qState.propagateGate(qc::H, {1});
  qState.propagateGate(qc::X, {3});
  qState.propagateGate(qc::X, {3}, {}, {1});

  EXPECT_THAT(qState.toString(),
              testing::HasSubstr("|0000> -> 0.71, |1010> -> 0.71"));
}

TEST_F(QubitStateTest, ApplyPosNegControlledGate) {
  QubitState qState = QubitState(4, 8);
  qState.propagateGate(qc::H, {0});
  qState.propagateGate(qc::H, {1});
  qState.propagateGate(qc::H, {2});
  qState.propagateGate(qc::X, {3}, {0, 1}, {2});

  EXPECT_THAT(
      qState.toString(),
      testing::HasSubstr("|0000> -> 0.35, |0001> -> 0.35, |0010> -> 0.35, "
                         "|0100> -> 0.35, |0101> -> 0.35, |0110> -> 0.35, "
                         "|0111> -> 0.35, |1011> -> 0.35"));
}

TEST_F(QubitStateTest, ApplyControlledTwoQubitGate) {
  QubitState qState = QubitState(4, 4);
  qState.propagateGate(qc::H, {3});
  qState.propagateGate(qc::H, {2});
  qState.propagateGate(qc::SWAP, {2, 1}, {3}, {}, {});

  EXPECT_THAT(
      qState.toString(),
      testing::HasSubstr(
          "|0000> -> 0.50, |0100> -> 0.50, |1000> -> 0.50, |1010> -> 0.50"));
}

TEST_F(QubitStateTest, propagateGateCheckErrorIfTwoManyAmplitudesAreNonzero) {
  QubitState qState = QubitState(4, 2);
  qState.propagateGate(qc::H, {3});
  qState.propagateGate(qc::X, {2}, {3});

  EXPECT_THROW(qState.propagateGate(qc::H, {2});, std::domain_error);
}

TEST_F(QubitStateTest, doMeasurementWithOneResult) {
  QubitState qState = QubitState(1, 2);
  qState.propagateGate(qc::X, {0});
  std::map<MeasurementResult, QubitState> const res = qState.measureQubit(0);

  EXPECT_TRUE(size(res) == 1);
  for (auto const& [key, value] : res) {
    EXPECT_TRUE(qState == value);
    EXPECT_TRUE(key.measurementResult == 1);
    EXPECT_DOUBLE_EQ(key.probability, 1.00);
  }
}

TEST_F(QubitStateTest, doMeasurementWithTwoResults) {
  QubitState qState = QubitState(2, 2);
  qState.propagateGate(qc::H, {0});
  qState.propagateGate(qc::X, {1}, {0});
  std::map<MeasurementResult, QubitState> const res = qState.measureQubit(0);

  QubitState zeroReference = QubitState(0, 0);
  QubitState oneReference = QubitState(0, 0);
  oneReference.propagateGate(qc::X, {1});
  oneReference.propagateGate(qc::X, {0});

  unsigned int checkCombination = 0;

  EXPECT_TRUE(size(res) == 2);
  for (auto const& [key, value] : res) {
    if (key.measurementResult == 0) {
      EXPECT_TRUE(zeroReference == value);
      checkCombination++;
    } else if (key.measurementResult == 1) {
      EXPECT_TRUE(oneReference == value);
      checkCombination += 2;
    }
    EXPECT_DOUBLE_EQ(key.probability, 0.50);
  }
  EXPECT_EQ(checkCombination, 3);
}