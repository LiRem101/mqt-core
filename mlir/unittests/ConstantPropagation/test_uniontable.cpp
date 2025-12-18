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

// ##################################################
// # Helper functions
// ##################################################

// ##################################################
// # Basic tests
// ##################################################

TEST(SimpleUnionTableTest, propagateQubitAlloc) {
  UnionTable ut = UnionTable(4, 2);
  unsigned int const qubitIndex0 = ut.propagateQubitAlloc();
  unsigned int const qubitIndex1 = ut.propagateQubitAlloc();

  EXPECT_EQ(qubitIndex0, 0);
  EXPECT_EQ(qubitIndex1, 1);
  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 0, HybridStates: {{|0> -> 1.00}: p = 1.00;}"));
  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 1, HybridStates: {{|0> -> 1.00}: p = 1.00;}"));
}

TEST(SimpleUnionTableTest, doMeasurementAndGetToTop) {
  UnionTable ut = UnionTable(4, 0);
  ut.propagateQubitAlloc();
  ut.propagateQubitAlloc();
  ut.propagateGate(qc::H, {0});
  ut.propagateMeasurement(0, 0);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 0, Bits: 0, HybridStates: {TOP}"));
}

TEST(SimpleUnionTableTest, doMeasurementOnTop) {
  UnionTable ut = UnionTable(2, 1);
  ut.propagateQubitAlloc();
  ut.propagateQubitAlloc();
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});
  ut.propagateGate(qc::H, {1}); // qState enters TOP
  ut.propagateMeasurement(0, 0);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 10, Bits: 0, HybridStates: {TOP}"));
}

TEST(SimpleUnionTableTest, doResetOnTop) {
  UnionTable ut = UnionTable(2, 1);
  ut.propagateQubitAlloc();
  ut.propagateQubitAlloc();
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});
  ut.propagateGate(qc::H, {1}); // qState enters TOP
  ut.propagateReset(0);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 10, HybridStates: {TOP}"));
}

TEST(SimpleUnionTableTest, unifyTooLargeHybridStates) {
  UnionTable ut = UnionTable(4, 0);
  ut.propagateQubitAlloc();
  ut.propagateQubitAlloc();
  ut.propagateQubitAlloc();
  ut.propagateGate(qc::H, {0});
  ut.propagateMeasurement(0, 0);
  ut.propagateGate(qc::X, {1}, {0});

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 10, Bits: 0, HybridStates: {TOP}"));
  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 2, HybridStates: {{|0> -> 1.00}: p = 1.00;}"));
}

class UnionTableTest : public ::testing::Test {
protected:
  UnionTable ut = UnionTable(4, 4);

  void SetUp() override {
    ut.propagateQubitAlloc();
    ut.propagateQubitAlloc();
    ut.propagateQubitAlloc();
    ut.propagateQubitAlloc();
  }

  void TearDown() override {}
};

TEST_F(UnionTableTest, ApplyHGate) {
  ut.propagateGate(qc::H, {0});

  EXPECT_THAT(
      ut.toString(),
      testing::HasSubstr(
          "Qubits: 0, HybridStates: {{|0> -> 0.71, |1> -> 0.71}: p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyHGateToThirdQubit) {
  ut.propagateGate(qc::H, {2});

  EXPECT_THAT(
      ut.toString(),
      testing::HasSubstr(
          "Qubits: 2, HybridStates: {{|0> -> 0.71, |1> -> 0.71}: p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyParametrizedGateToThirdQubit) {
  ut.propagateGate(qc::H, {2});
  ut.propagateGate(qc::U, {2}, {}, {}, {}, {}, {1, 0.5, 2});

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 2, HybridStates: {{|0> -> 0.76 - "
                                 "i0.31, |1> -> -0.20 + i0.53}: p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyQuantumControlledGate) {
  ut.propagateGate(qc::H, {1});
  ut.propagateGate(qc::X, {3});
  ut.propagateGate(qc::X, {3}, {1});

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 0, HybridStates: {{|0> -> 1.00}: p = 1.00;}"));
  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 2, HybridStates: {{|0> -> 1.00}: p = 1.00;}"));
  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 31, HybridStates: {{|01> -> 0.71, "
                                 "|10> -> 0.71}: p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyQuantumNegControlledGate) {
  ut.propagateGate(qc::H, {1});
  ut.propagateGate(qc::X, {3});
  ut.propagateGate(qc::X, {3}, {}, {1});

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 0, HybridStates: {{|0> -> 1.00}: p = 1.00;}"));
  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 2, HybridStates: {{|0> -> 1.00}: p = 1.00;}"));
  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 31, HybridStates: {{|00> -> 0.71, "
                                 "|11> -> 0.71}: p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyClassicalControlledGateThatsFalse) {
  ut.propagateBitDef(true);
  ut.propagateBitDef(false);
  ut.propagateGate(qc::H, {1});
  ut.propagateGate(qc::X, {3});
  ut.propagateGate(qc::X, {3}, {1}, {}, {1});

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 31, Bits: 1, HybridStates: {{|10> "
                                 "-> 0.71, |11> -> 0.71}: 0, p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyClassicalControlledGateThatsTrue) {
  ut.propagateBitDef(true);
  ut.propagateBitDef(false);
  ut.propagateGate(qc::H, {1});
  ut.propagateGate(qc::X, {3});
  ut.propagateGate(qc::X, {3}, {1}, {}, {0});

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 31, Bits: 0, HybridStates: {{|01> "
                                 "-> 0.71, |10> -> 0.71}: 1, p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyClassicalNegControlledGateThatsFalse) {
  ut.propagateBitDef(true);
  ut.propagateBitDef(false);
  ut.propagateGate(qc::H, {1});
  ut.propagateGate(qc::X, {3});
  ut.propagateGate(qc::X, {3}, {1}, {}, {}, {0});

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 31, Bits: 0, HybridStates: {{|10> "
                                 "-> 0.71, |11> -> 0.71}: 1, p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyClassicalNegControlledGateThatsTrue) {
  ut.propagateBitDef(true);
  ut.propagateBitDef(false);
  ut.propagateGate(qc::H, {1});
  ut.propagateGate(qc::X, {3});
  ut.propagateGate(qc::X, {3}, {1}, {}, {}, {1});

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 31, Bits: 1, HybridStates: {{|01> "
                                 "-> 0.71, |10> -> 0.71}: 0, p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyPosNegControlledClassicalGateFalse) {
  ut.propagateBitDef(true);
  ut.propagateBitDef(true);
  ut.propagateGate(qc::H, {1});
  ut.propagateGate(qc::X, {3});
  ut.propagateGate(qc::X, {3}, {1}, {}, {1}, {0});

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 31, Bits: 10, HybridStates: {{|10> "
                                 "-> 0.71, |11> -> 0.71}: 11, p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyPosNegControlledClassicalGateTrue) {
  ut.propagateBitDef(true);
  ut.propagateBitDef(false);
  ut.propagateGate(qc::H, {1});
  ut.propagateGate(qc::X, {3});
  ut.propagateGate(qc::X, {3}, {1}, {}, {0}, {1});

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 31, Bits: 10, HybridStates: {{|01> "
                                 "-> 0.71, |10> -> 0.71}: 01, p = 1.00;}"));
}

TEST_F(UnionTableTest, ApplyControlledTwoBitGate) {
  ut.propagateBitDef(true);
  ut.propagateBitDef(false);
  ut.propagateBitDef(true);
  ut.propagateGate(qc::H, {1});
  ut.propagateGate(qc::X, {3});
  ut.propagateGate(qc::X, {3}, {1}, {}, {0, 2}, {1});

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 31, Bits: 210, HybridStates: {{|01> "
                                 "-> 0.71, |10> -> 0.71}: 101, p = 1.00;}"));
}

TEST_F(UnionTableTest, doMeasurementWithOneResult) {
  ut.propagateGate(qc::X, {0});
  ut.propagateMeasurement(0, 0);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 0, Bits: 0, HybridStates: {{|1> "
                                 "-> 1.00}: 1, p = 1.00;}"));
}

TEST_F(UnionTableTest, doMeasurementWithTwoResults) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});
  ut.propagateMeasurement(0, 0);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr(
                  "Qubits: 10, Bits: 0, HybridStates: {{|00> "
                  "-> 1.00}: 0, p = 0.50; {|11> -> 1.00}: 1, p = 0.50;}"));
}

TEST_F(UnionTableTest, doResetWithOneResult) {
  ut.propagateGate(qc::X, {0});
  ut.propagateReset(0);

  EXPECT_THAT(ut.toString(),
              testing::HasSubstr("Qubits: 0, HybridStates: {{|0> "
                                 "-> 1.00}: p = 1.00;}"));
}

TEST_F(UnionTableTest, doResetWithTwoResults) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});
  ut.propagateReset(0);

  EXPECT_THAT(
      ut.toString(),
      testing::HasSubstr("Qubits: 10, HybridStates: {{|10> "
                         "-> 1.00}: p = 0.50; {|00> -> 1.00}: p = 0.50;}"));
}

class UnionTablePropertiesTest : public ::testing::Test {
protected:
  UnionTable ut = UnionTable(3, 2);

  void SetUp() override {
    ut.propagateQubitAlloc();
    ut.propagateQubitAlloc();
    ut.propagateQubitAlloc();
  }

  void TearDown() override {}
};

TEST_F(UnionTablePropertiesTest, alwaysZeroOneAreFalse) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});
  ut.propagateGate(qc::X, {2}, {1});
  ut.propagateGate(qc::X, {0});
  ut.propagateMeasurement(1, 0);
  ut.propagateGate(qc::H, {0}, {}, {}, {0});

  EXPECT_FALSE(ut.isQubitAlwaysZero(2));
  EXPECT_FALSE(ut.isQubitAlwaysOne(0));
}

TEST_F(UnionTablePropertiesTest, alwaysZeroIsTrue) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});
  ut.propagateGate(qc::X, {2}, {1});
  ut.propagateGate(qc::X, {0});
  ut.propagateMeasurement(1, 0);
  ut.propagateGate(qc::X, {2}, {}, {}, {0});
  ut.propagateGate(qc::H, {0}, {}, {}, {0});

  EXPECT_TRUE(ut.isQubitAlwaysZero(2));
}

TEST_F(UnionTablePropertiesTest, alwaysOneIsTrue) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});
  ut.propagateGate(qc::X, {2}, {1});
  ut.propagateGate(qc::X, {0});
  ut.propagateMeasurement(2, 0);
  ut.propagateGate(qc::H, {0}, {}, {}, {0});
  ut.propagateGate(qc::Z, {0}, {1, 2});
  ut.propagateGate(qc::H, {0}, {}, {}, {0});

  EXPECT_TRUE(ut.isQubitAlwaysOne(0));
}

TEST_F(UnionTablePropertiesTest, bitAlwaysZeroIsTrueOneIsFalse) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});
  ut.propagateMeasurement(0, 0);
  ut.propagateGate(qc::X, {1}, {}, {}, {0});
  ut.propagateMeasurement(1, 1);

  EXPECT_FALSE(ut.isBitAlwaysOne(0));
  EXPECT_TRUE(ut.isBitAlwaysZero(1));
}

TEST_F(UnionTablePropertiesTest, bitAlwaysZeroIsFalseOneIsTrue) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});
  ut.propagateMeasurement(0, 0);
  ut.propagateGate(qc::X, {1}, {}, {}, {}, {0});
  ut.propagateMeasurement(1, 1);

  EXPECT_TRUE(ut.isBitAlwaysOne(1));
  EXPECT_FALSE(ut.isBitAlwaysZero(0));
}

TEST_F(UnionTablePropertiesTest, testAllTop) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});
  ut.propagateGate(qc::X, {2}, {1});
  ut.propagateMeasurement(2, 0);
  ut.propagateGate(qc::H, {0});
  EXPECT_FALSE(ut.allTop());
  ut.propagateGate(qc::H, {1});
  EXPECT_FALSE(ut.allTop());
  ut.propagateMeasurement(0, 1);
  EXPECT_TRUE(ut.allTop());
}

TEST_F(UnionTablePropertiesTest, testHasAlwaysZeroAmplitude) {
  ut.propagateGate(qc::X, {2});
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});
  ut.propagateMeasurement(1, 0);
  ut.propagateGate(qc::H, {0}, {}, {}, {0});
  EXPECT_FALSE(ut.hasAlwaysZeroAmplitude({0, 1, 2}, 0));
  EXPECT_FALSE(ut.hasAlwaysZeroAmplitude({0, 1, 2}, 2));
  EXPECT_FALSE(ut.hasAlwaysZeroAmplitude({0, 1, 2}, 3));
  EXPECT_FALSE(ut.hasAlwaysZeroAmplitude({0, 1, 2}, 4));
  EXPECT_FALSE(ut.hasAlwaysZeroAmplitude({0, 1, 2}, 5));
  EXPECT_FALSE(ut.hasAlwaysZeroAmplitude({0, 1, 2}, 6));
  EXPECT_FALSE(ut.hasAlwaysZeroAmplitude({0, 1, 2}, 7));
  EXPECT_TRUE(ut.hasAlwaysZeroAmplitude({0, 1}, 1));
  EXPECT_TRUE(ut.hasAlwaysZeroAmplitude({0, 1, 2}, 1));
}

class SmallUnionTableTest : public ::testing::Test {
protected:
  UnionTable ut = UnionTable(2, 2);

  void SetUp() override {
    ut.propagateQubitAlloc();
    ut.propagateQubitAlloc();
    ut.propagateQubitAlloc();
    ut.propagateQubitAlloc();
  }

  void TearDown() override {}
};

TEST_F(SmallUnionTableTest, handleErrorIfTwoManyAmplitudesAreNonzero) {
  ut.propagateGate(qc::H, {3});
  ut.propagateGate(qc::X, {2}, {3});
  ut.propagateGate(qc::H, {2});
  ut.propagateGate(qc::H, {2});

  EXPECT_THAT(
      ut.toString(),
      testing::HasSubstr("Qubits: 32, HybridStates: {{TOP}: p = 1.00;}"));
}

TEST_F(SmallUnionTableTest, applyGatesOnPartiallyTopQState) {
  ut.propagateGate(qc::H, {2});
  ut.propagateGate(qc::H, {3});
  ut.propagateGate(qc::X, {2}, {3}); // Qubit 2 and 3 enter TOP
  ut.propagateGate(qc::X, {1}, {2});

  EXPECT_THAT(
      ut.toString(),
      testing::HasSubstr("Qubits: 321, HybridStates: {{TOP}: p = 1.00;}"));
}
