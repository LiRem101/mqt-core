/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
  auto value = res.at(1);
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/Transforms/ConstantPropagation/RewriteChecker.hpp"
#include "mlir/Dialect/MQTOpt/Transforms/ConstantPropagation/UnionTable.hpp"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <mlir/Dialect/MQTOpt/Transforms/Passes.h>

using namespace mqt::ir::opt::qcp;

class HybridStateTest : public ::testing::Test {
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

TEST_F(HybridStateTest, ApplyHGate) {
  HybridState hState = HybridState(1, 4, 2);
  hState.propagateGate(qc::H, {0});

  EXPECT_THAT(hState.toString(),
              testing::HasSubstr("{|0> -> 0.71, |1> -> 0.71}: p = 1.00;"));
}

TEST_F(HybridStateTest, ApplyHGateToThirdQubit) {
  HybridState hState = HybridState(4, 4, 2, {false, true}, 0.5);
  hState.propagateGate(qc::H, {2});

  EXPECT_THAT(
      hState.toString(),
      testing::HasSubstr("{|0000> -> 0.71, |0100> -> 0.71}: 10, p = 0.50;"));
}

TEST_F(HybridStateTest, ApplyParametrizedGateToThirdQubit) {
  HybridState hState = HybridState(4, 4, 2, {true}, 0.75);
  hState.propagateGate(qc::H, {2});
  hState.propagateGate(qc::U, {2}, {}, {}, {}, {}, {1, 0.5, 2});

  EXPECT_THAT(
      hState.toString(),
      testing::HasSubstr(
          "{|0000> -> 0.76 - i0.31, |0100> -> -0.20 + i0.53}: 1, p = 0.75;"));
}

TEST_F(HybridStateTest, ApplyQuantumControlledGate) {
  HybridState hState = HybridState(4, 4, 2);
  hState.propagateGate(qc::H, {1});
  hState.propagateGate(qc::X, {3});
  hState.propagateGate(qc::X, {3}, {1});

  EXPECT_THAT(
      hState.toString(),
      testing::HasSubstr("{|0010> -> 0.71, |1000> -> 0.71}: p = 1.00;"));
}

TEST_F(HybridStateTest, ApplyQuantumNegControlledGate) {
  HybridState hState = HybridState(4, 4, 2);
  hState.propagateGate(qc::H, {1});
  hState.propagateGate(qc::X, {3});
  hState.propagateGate(qc::X, {3}, {}, {1});

  EXPECT_THAT(
      hState.toString(),
      testing::HasSubstr("{|0000> -> 0.71, |1010> -> 0.71}: p = 1.00;"));
}

TEST_F(HybridStateTest, ApplyClassicalControlledGateThatsFalse) {
  HybridState hState = HybridState(4, 4, 2, {true, false}, 0.5);
  hState.propagateGate(qc::H, {1});
  hState.propagateGate(qc::X, {3});
  hState.propagateGate(qc::X, {3}, {1}, {}, {1});

  EXPECT_THAT(
      hState.toString(),
      testing::HasSubstr("{|1000> -> 0.71, |1010> -> 0.71}: 01, p = 0.50;"));
}

TEST_F(HybridStateTest, ApplyClassicalControlledGateThatsTrue) {
  HybridState hState = HybridState(4, 4, 2, {true, false}, 0.5);
  hState.propagateGate(qc::H, {1});
  hState.propagateGate(qc::X, {3});
  hState.propagateGate(qc::X, {3}, {1}, {}, {0});

  EXPECT_THAT(
      hState.toString(),
      testing::HasSubstr("{|0010> -> 0.71, |1000> -> 0.71}: 01, p = 0.50;"));
}

TEST_F(HybridStateTest, ApplyClassicalMegControlledGateThatsFalse) {
  HybridState hState = HybridState(4, 4, 2, {true, false}, 0.5);
  hState.propagateGate(qc::H, {1});
  hState.propagateGate(qc::X, {3});
  hState.propagateGate(qc::X, {3}, {1}, {}, {}, {0});

  EXPECT_THAT(
      hState.toString(),
      testing::HasSubstr("{|1000> -> 0.71, |1010> -> 0.71}: 01, p = 0.50;"));
}

TEST_F(HybridStateTest, ApplyClassicalNegControlledGateThatsTrue) {
  HybridState hState = HybridState(4, 4, 2, {true, false}, 0.5);
  hState.propagateGate(qc::H, {1});
  hState.propagateGate(qc::X, {3});
  hState.propagateGate(qc::X, {3}, {1}, {}, {}, {1});

  EXPECT_THAT(
      hState.toString(),
      testing::HasSubstr("{|0010> -> 0.71, |1000> -> 0.71}: 01, p = 0.50;"));
}

TEST_F(HybridStateTest, ApplyPosNegControlledClassicalGateFalse) {
  HybridState hState = HybridState(4, 4, 2, {true, true}, 0.5);
  hState.propagateGate(qc::H, {1});
  hState.propagateGate(qc::X, {3});
  hState.propagateGate(qc::X, {3}, {1}, {}, {1}, {0});

  EXPECT_THAT(
      hState.toString(),
      testing::HasSubstr("{|1000> -> 0.71, |1010> -> 0.71}: 11, p = 0.50;"));
}

TEST_F(HybridStateTest, ApplyPosNegControlledClassicalGateTrue) {
  HybridState hState = HybridState(4, 4, 2, {true, false}, 0.5);
  hState.propagateGate(qc::H, {1});
  hState.propagateGate(qc::X, {3});
  hState.propagateGate(qc::X, {3}, {1}, {}, {0}, {1});

  EXPECT_THAT(
      hState.toString(),
      testing::HasSubstr("{|0010> -> 0.71, |1000> -> 0.71}: 01, p = 0.50;"));
}

TEST_F(HybridStateTest, ApplyControlledTwoBitGate) {
  HybridState hState = HybridState(4, 4, 4, {true, false, true}, 0.5);
  hState.propagateGate(qc::H, {1});
  hState.propagateGate(qc::X, {3});
  hState.propagateGate(qc::X, {3}, {1}, {}, {0, 2}, {1});

  EXPECT_THAT(
      hState.toString(),
      testing::HasSubstr("{|0010> -> 0.71, |1000> -> 0.71}: 101, p = 0.50;"));
}

TEST_F(HybridStateTest, handleErrorIfTwoManyAmplitudesAreNonzero) {
  HybridState hState = HybridState(4, 2, 2, {true}, 0.2);
  hState.propagateGate(qc::H, {3});
  hState.propagateGate(qc::X, {2}, {3});
  hState.propagateGate(qc::H, {2}); // error occures here
  hState.propagateGate(qc::H, {0}); // should leave QubitState in TOP

  EXPECT_THAT(hState.toString(), testing::HasSubstr("{TOP}: 1, p = 0.20;"));
}

TEST_F(HybridStateTest, doMeasurementWithOneResult) {
  HybridState hState = HybridState(1, 2, 2, {false}, 0.4);
  hState.propagateGate(qc::X, {0});
  std::vector<HybridState> const res = hState.propagateMeasurement(0, 0);

  EXPECT_TRUE(size(res) == 1);
  EXPECT_THAT(res.at(0).toString(),
              testing::HasSubstr("{|1> -> 1.00}: 01, p = 0.40;"));
}

TEST_F(HybridStateTest, doMeasurementWithTwoResults) {
  HybridState hState = HybridState(2, 2, 2, {false}, 0.4);
  hState.propagateGate(qc::H, {0});
  hState.propagateGate(qc::X, {1}, {0});
  std::vector<HybridState> const res = hState.propagateMeasurement(0, 0);

  EXPECT_TRUE(size(res) == 2);
  std::string resString = res.at(0).toString() + res.at(1).toString();
  EXPECT_THAT(resString, testing::HasSubstr("{|00> -> 1.00}: 00, p = 0.20;"));
  EXPECT_THAT(resString, testing::HasSubstr("{|11> -> 1.00}: 01, p = 0.20;"));
}

TEST_F(HybridStateTest, doMeasurementAndGetToTop) {
  HybridState hState = HybridState(2, 2, 1, {false}, 0.4);
  hState.propagateGate(qc::H, {0});
  hState.propagateGate(qc::X, {1}, {0});
  EXPECT_THROW(hState.propagateMeasurement(0, 0);, std::domain_error);
}

TEST_F(HybridStateTest, doMeasurementOnTop) {
  HybridState hState = HybridState(4, 2, 2, {true}, 0.2);
  hState.propagateGate(qc::H, {3});
  hState.propagateGate(qc::X, {2}, {3});
  hState.propagateGate(qc::H, {2}); // qState enters TOP

  EXPECT_THROW(hState.propagateMeasurement(0, 0);, std::domain_error);
}

TEST_F(HybridStateTest, unifyTwoHybridStates) {
  HybridState hState1 = HybridState(3, 10, 4, {true, false}, 0.5);
  hState1.propagateGate(qc::H, {2});
  hState1.propagateGate(qc::X, {1}, {2});
  hState1.propagateGate(qc::X, {0}, {1});

  HybridState hState2 = HybridState(2, 10, 4, {false}, 0.2);
  hState2.propagateGate(qc::H, {1});
  hState2.propagateGate(qc::X, {0}, {1});

  HybridState unified = hState1.unify(hState2, {1, 3}, {1});

  EXPECT_THAT(
      unified.toString(),
      testing::HasSubstr("{|00000> -> 0.50, |01010> -> 0.50, "
                         "|10101> -> 0.50, |11111> -> 0.50}: 100, p = 0.10;"));
}

TEST_F(HybridStateTest, unifyTooLargeQuantumStates) {
  HybridState hState1 = HybridState(3, 3, 4, {true, false}, 0.5);
  hState1.propagateGate(qc::H, {2});
  hState1.propagateGate(qc::X, {1}, {2});
  hState1.propagateGate(qc::X, {0}, {1});

  HybridState hState2 = HybridState(2, 3, 4, {false}, 0.2);
  hState2.propagateGate(qc::H, {1});
  hState2.propagateGate(qc::X, {0}, {1});

  HybridState unified = hState1.unify(hState2, {1, 3}, {1});

  EXPECT_THAT(unified.toString(), testing::HasSubstr("{TOP}: 100, p = 0.10;"));
}

TEST_F(HybridStateTest, unifyTooLargeHybridStates) {
  HybridState hState1 = HybridState(3, 3, 2, {true, false}, 0.5);
  hState1.propagateGate(qc::H, {2});
  hState1.propagateGate(qc::X, {1}, {2});
  hState1.propagateGate(qc::X, {0}, {1});

  HybridState hState2 = HybridState(2, 3, 2, {false}, 0.2);
  hState2.propagateGate(qc::H, {1});
  hState2.propagateGate(qc::X, {0}, {1});

  EXPECT_THROW(hState1.unify(hState2, {1, 3}, {1});, std::domain_error);
}