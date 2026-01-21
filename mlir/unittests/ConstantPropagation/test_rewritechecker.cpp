/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/OpType.hpp"
#include "mlir/Dialect/MQTOpt/Transforms/ConstantPropagation/RewriteChecker.hpp"
#include "mlir/Dialect/MQTOpt/Transforms/ConstantPropagation/UnionTable.hpp"

#include <gtest/gtest.h>
#include <optional>
#include <utility>

using namespace mqt::ir::opt::qcp;

class RewriteCheckerTest : public testing::Test {
protected:
  UnionTable ut;

  RewriteCheckerTest() : ut(UnionTable(8, 4)) {}

  void SetUp() override {
    ut.propagateQubitAlloc();
    ut.propagateQubitAlloc();
  }

  void TearDown() override {}
};

TEST_F(RewriteCheckerTest, FindEquivalentBit) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});
  ut.propagateMeasurement(1, 0);
  const std::optional<std::pair<unsigned int, bool>> result =
      RewriteChecker::getEquivalentBit(ut, 1);
  ASSERT_TRUE(result.has_value());
  auto [bitIndex, bitValue] = result.value();
  ASSERT_TRUE(bitIndex == 0);
  ASSERT_TRUE(bitValue);
}

TEST_F(RewriteCheckerTest, FindEquivalentReversedBit) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});
  ut.propagateGate(qc::X, {0});
  ut.propagateMeasurement(0, 0);
  const std::optional<std::pair<unsigned int, bool>> result =
      RewriteChecker::getEquivalentBit(ut, 1);
  ASSERT_TRUE(result.has_value());
  auto [bitIndex, bitValue] = result.value();
  ASSERT_TRUE(bitIndex == 0);
  ASSERT_FALSE(bitValue);
}

TEST_F(RewriteCheckerTest, FindNoEquivalentBit) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});
  ut.propagateGate(qc::H, {0});
  ut.propagateMeasurement(1, 0);
  const std::optional<std::pair<unsigned int, bool>> result =
      RewriteChecker::getEquivalentBit(ut, 0);
  ASSERT_FALSE(result.has_value());
}

TEST_F(RewriteCheckerTest, ZeroIsAlwaysAntecedent) {
  ut.propagateMeasurement(0, 0);
  ut.propagateGate(qc::H, {1});
  auto [antecedentQubits, antecedentBits] =
      RewriteChecker::getAntecedentsOfQubit(ut, 1, false, {0}, {}, {0}, {});
  ASSERT_EQ(antecedentQubits.size(), 1);
  ASSERT_EQ(antecedentBits.size(), 1);
  ASSERT_EQ(*antecedentQubits.begin(), 0);
  ASSERT_EQ(*antecedentBits.begin(), 0);
}

TEST_F(RewriteCheckerTest, ImplyingQubitA) {
  ut.propagateGate(qc::H, {0});
  ut.propagateMeasurement(0, 0);
  ut.propagateGate(qc::H, {1}, {0});
  auto [antecedentQubits, antecedentBits] =
      RewriteChecker::getAntecedentsOfQubit(ut, 1, true, {}, {0}, {}, {0});
  auto [antecedentQubitsEmptyA, antecedentBitsEmptyA] =
      RewriteChecker::getAntecedentsOfQubit(ut, 1, false, {0}, {}, {0}, {});
  auto [antecedentQubitsEmptyB, antecedentBitsEmptyB] =
      RewriteChecker::getAntecedentsOfQubit(ut, 1, false, {}, {0}, {}, {0});
  auto [antecedentQubitsEmptyC, antecedentBitsEmptyC] =
      RewriteChecker::getAntecedentsOfQubit(ut, 1, true, {0}, {}, {0}, {});

  ASSERT_EQ(antecedentQubits.size(), 1);
  ASSERT_EQ(antecedentBits.size(), 1);
  ASSERT_EQ(*antecedentQubits.begin(), 0);
  ASSERT_EQ(*antecedentBits.begin(), 0);
  ASSERT_TRUE(antecedentQubitsEmptyA.empty());
  ASSERT_TRUE(antecedentBitsEmptyA.empty());
  ASSERT_TRUE(antecedentQubitsEmptyB.empty());
  ASSERT_TRUE(antecedentBitsEmptyB.empty());
  ASSERT_TRUE(antecedentQubitsEmptyC.empty());
  ASSERT_TRUE(antecedentBitsEmptyC.empty());
}

TEST_F(RewriteCheckerTest, ImplyingBitA) {
  ut.propagateGate(qc::H, {0});
  ut.propagateMeasurement(0, 0);
  ut.propagateGate(qc::H, {1}, {0});
  auto [antecedentQubits, antecedentBits] =
      RewriteChecker::getAntecedentsOfBit(ut, 0, false, {1}, {}, {}, {});
  auto [antecedentQubitsEmptyA, antecedentBitsEmptyA] =
      RewriteChecker::getAntecedentsOfBit(ut, 0, true, {1}, {}, {}, {});
  auto [antecedentQubitsEmptyB, antecedentBitsEmptyB] =
      RewriteChecker::getAntecedentsOfBit(ut, 0, false, {}, {1}, {}, {});
  auto [antecedentQubitsEmptyC, antecedentBitsEmptyC] =
      RewriteChecker::getAntecedentsOfBit(ut, 0, true, {}, {1}, {}, {});

  ASSERT_EQ(antecedentQubits.size(), 1);
  ASSERT_TRUE(antecedentBits.empty());
  ASSERT_EQ(*antecedentQubits.begin(), 1);
  ASSERT_TRUE(antecedentQubitsEmptyA.empty());
  ASSERT_TRUE(antecedentBitsEmptyA.empty());
  ASSERT_TRUE(antecedentQubitsEmptyB.empty());
  ASSERT_TRUE(antecedentBitsEmptyB.empty());
  ASSERT_TRUE(antecedentQubitsEmptyC.empty());
  ASSERT_TRUE(antecedentBitsEmptyC.empty());
}

TEST_F(RewriteCheckerTest, ImplyingQubitB) {
  ut.propagateGate(qc::H, {0});
  ut.propagateMeasurement(0, 0);
  ut.propagateGate(qc::H, {1}, {0});
  ut.propagateGate(qc::H, {0}, {}, {}, {}, {0});
  auto [antecedentQubits, antecedentBits] =
      RewriteChecker::getAntecedentsOfQubit(ut, 0, false, {}, {}, {0}, {});
  auto [antecedentQubitsEmptyA, antecedentBitsEmptyA] =
      RewriteChecker::getAntecedentsOfQubit(ut, 0, true, {}, {}, {0}, {});
  auto [antecedentQubitsEmptyB, antecedentBitsEmptyB] =
      RewriteChecker::getAntecedentsOfQubit(ut, 0, false, {}, {}, {}, {0});
  auto [antecedentQubitsEmptyC, antecedentBitsEmptyC] =
      RewriteChecker::getAntecedentsOfQubit(ut, 0, true, {}, {}, {}, {0});

  ASSERT_TRUE(antecedentQubits.empty());
  ASSERT_EQ(antecedentBits.size(), 1);
  ASSERT_EQ(*antecedentBits.begin(), 0);
  ASSERT_TRUE(antecedentQubitsEmptyA.empty());
  ASSERT_TRUE(antecedentBitsEmptyA.empty());
  ASSERT_TRUE(antecedentQubitsEmptyB.empty());
  ASSERT_TRUE(antecedentBitsEmptyB.empty());
  ASSERT_TRUE(antecedentQubitsEmptyC.empty());
  ASSERT_TRUE(antecedentBitsEmptyC.empty());
}

TEST_F(RewriteCheckerTest, ImplyingBitB) {
  ut.propagateGate(qc::H, {0});
  ut.propagateMeasurement(0, 0);
  ut.propagateGate(qc::X, {1}, {}, {0});
  ut.propagateGate(qc::H, {1}, {0});
  ut.propagateGate(qc::H, {0}, {}, {}, {}, {0});
  auto [antecedentQubits, antecedentBits] =
      RewriteChecker::getAntecedentsOfBit(ut, 0, true, {}, {0}, {}, {});
  auto [antecedentQubitsEmptyA, antecedentBitsEmptyA] =
      RewriteChecker::getAntecedentsOfBit(ut, 0, false, {0}, {}, {}, {});
  auto [antecedentQubitsEmptyB, antecedentBitsEmptyB] =
      RewriteChecker::getAntecedentsOfBit(ut, 0, false, {}, {0}, {}, {});
  auto [antecedentQubitsEmptyC, antecedentBitsEmptyC] =
      RewriteChecker::getAntecedentsOfBit(ut, 0, true, {0}, {}, {}, {});

  ASSERT_EQ(antecedentQubits.size(), 1);
  ASSERT_TRUE(antecedentBits.empty());
  ASSERT_EQ(*antecedentQubits.begin(), 0);
  ASSERT_TRUE(antecedentQubitsEmptyA.empty());
  ASSERT_TRUE(antecedentBitsEmptyA.empty());
  ASSERT_TRUE(antecedentQubitsEmptyB.empty());
  ASSERT_TRUE(antecedentBitsEmptyB.empty());
  ASSERT_TRUE(antecedentQubitsEmptyC.empty());
  ASSERT_TRUE(antecedentBitsEmptyC.empty());
}

TEST_F(RewriteCheckerTest, ImplyingQubitC) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1});
  ut.propagateMeasurement(0, 0);
  ut.propagateGate(qc::H, {1}, {0});
  auto [antecedentQubits, antecedentBits] =
      RewriteChecker::getAntecedentsOfQubit(ut, 1, false, {}, {}, {}, {0});
  auto [antecedentQubitsEmptyA, antecedentBitsEmptyA] =
      RewriteChecker::getAntecedentsOfQubit(ut, 1, false, {}, {}, {0}, {});
  auto [antecedentQubitsEmptyB, antecedentBitsEmptyB] =
      RewriteChecker::getAntecedentsOfQubit(ut, 1, false, {}, {}, {0}, {});
  auto [antecedentQubitsEmptyC, antecedentBitsEmptyC] =
      RewriteChecker::getAntecedentsOfQubit(ut, 1, true, {}, {}, {}, {0});

  ASSERT_TRUE(antecedentQubits.empty());
  ASSERT_EQ(antecedentBits.size(), 1);
  ASSERT_EQ(*antecedentBits.begin(), 0);
  ASSERT_TRUE(antecedentQubitsEmptyA.empty());
  ASSERT_TRUE(antecedentBitsEmptyA.empty());
  ASSERT_TRUE(antecedentQubitsEmptyB.empty());
  ASSERT_TRUE(antecedentBitsEmptyB.empty());
  ASSERT_TRUE(antecedentQubitsEmptyC.empty());
  ASSERT_TRUE(antecedentBitsEmptyC.empty());
}

TEST_F(RewriteCheckerTest, ImplyingBitC) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1});
  ut.propagateMeasurement(0, 0);
  ut.propagateGate(qc::H, {1}, {0});
  auto [antecedentQubits, antecedentBits] =
      RewriteChecker::getAntecedentsOfBit(ut, 0, false, {}, {1}, {}, {});
  auto [antecedentQubitsEmptyA, antecedentBitsEmptyA] =
      RewriteChecker::getAntecedentsOfBit(ut, 0, false, {1}, {}, {}, {});
  auto [antecedentQubitsEmptyB, antecedentBitsEmptyB] =
      RewriteChecker::getAntecedentsOfBit(ut, 0, true, {1}, {}, {}, {});
  auto [antecedentQubitsEmptyC, antecedentBitsEmptyC] =
      RewriteChecker::getAntecedentsOfBit(ut, 0, true, {}, {1}, {}, {});

  ASSERT_EQ(antecedentQubits.size(), 1);
  ASSERT_TRUE(antecedentBits.empty());
  ASSERT_EQ(*antecedentQubits.begin(), 1);
  ASSERT_TRUE(antecedentQubitsEmptyA.empty());
  ASSERT_TRUE(antecedentBitsEmptyA.empty());
  ASSERT_TRUE(antecedentQubitsEmptyB.empty());
  ASSERT_TRUE(antecedentBitsEmptyB.empty());
  ASSERT_TRUE(antecedentQubitsEmptyC.empty());
  ASSERT_TRUE(antecedentBitsEmptyC.empty());
}

TEST_F(RewriteCheckerTest, ImplyingQubitD) {
  ut.propagateGate(qc::H, {0});
  ut.propagateMeasurement(0, 0);
  ut.propagateGate(qc::X, {0});
  ut.propagateGate(qc::H, {1}, {0});
  auto [antecedentQubits, antecedentBits] =
      RewriteChecker::getAntecedentsOfQubit(ut, 1, true, {}, {}, {0}, {});
  auto [antecedentQubitsEmptyA, antecedentBitsEmptyA] =
      RewriteChecker::getAntecedentsOfQubit(ut, 1, false, {}, {}, {0}, {});
  auto [antecedentQubitsEmptyB, antecedentBitsEmptyB] =
      RewriteChecker::getAntecedentsOfQubit(ut, 1, false, {}, {}, {}, {0});
  auto [antecedentQubitsEmptyC, antecedentBitsEmptyC] =
      RewriteChecker::getAntecedentsOfQubit(ut, 1, true, {}, {}, {}, {0});

  ASSERT_TRUE(antecedentQubits.empty());
  ASSERT_EQ(antecedentBits.size(), 1);
  ASSERT_EQ(*antecedentBits.begin(), 0);
  ASSERT_TRUE(antecedentQubitsEmptyA.empty());
  ASSERT_TRUE(antecedentBitsEmptyA.empty());
  ASSERT_TRUE(antecedentQubitsEmptyB.empty());
  ASSERT_TRUE(antecedentBitsEmptyB.empty());
  ASSERT_TRUE(antecedentQubitsEmptyC.empty());
  ASSERT_TRUE(antecedentBitsEmptyC.empty());
}

TEST_F(RewriteCheckerTest, ImplyingBitD) {
  ut.propagateGate(qc::H, {0});
  ut.propagateMeasurement(0, 0);
  ut.propagateGate(qc::X, {0});
  ut.propagateGate(qc::H, {1}, {0});
  auto [antecedentQubits, antecedentBits] =
      RewriteChecker::getAntecedentsOfBit(ut, 0, true, {1}, {}, {}, {});
  auto [antecedentQubitsEmptyA, antecedentBitsEmptyA] =
      RewriteChecker::getAntecedentsOfBit(ut, 0, false, {1}, {}, {}, {});
  auto [antecedentQubitsEmptyB, antecedentBitsEmptyB] =
      RewriteChecker::getAntecedentsOfBit(ut, 0, true, {}, {1}, {}, {});
  auto [antecedentQubitsEmptyC, antecedentBitsEmptyC] =
      RewriteChecker::getAntecedentsOfBit(ut, 0, false, {}, {1}, {}, {});

  ASSERT_EQ(antecedentQubits.size(), 1);
  ASSERT_TRUE(antecedentBits.empty());
  ASSERT_EQ(*antecedentQubits.begin(), 1);
  ASSERT_TRUE(antecedentQubitsEmptyA.empty());
  ASSERT_TRUE(antecedentBitsEmptyA.empty());
  ASSERT_TRUE(antecedentQubitsEmptyB.empty());
  ASSERT_TRUE(antecedentBitsEmptyB.empty());
  ASSERT_TRUE(antecedentQubitsEmptyC.empty());
  ASSERT_TRUE(antecedentBitsEmptyC.empty());
}

TEST_F(RewriteCheckerTest, oneSetNonZeroOneQubit) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1});

  ASSERT_FALSE(RewriteChecker::isOnlyOneSetNotZero(ut, {0}, {{0}, {1}}));
  ASSERT_TRUE(RewriteChecker::isOnlyOneSetNotZero(ut, {1}, {{0}, {1}}));
}

TEST_F(RewriteCheckerTest, oneSetNonZeroTwoQubitFalse) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::H, {1});

  ASSERT_FALSE(
      RewriteChecker::isOnlyOneSetNotZero(ut, {0, 1}, {{0, 1, 2}, {3}}));
}

TEST_F(RewriteCheckerTest, oneSetNonZeroTwoQubitTrue) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});
  ut.propagateMeasurement(0, 0);

  ASSERT_TRUE(
      RewriteChecker::isOnlyOneSetNotZero(ut, {0, 1}, {{0, 1, 2}, {3}}));
}

TEST_F(RewriteCheckerTest, oneSetNonZeroMultipleQubitTrue) {
  ut.propagateQubitAlloc();
  ut.propagateQubitAlloc();
  ut.propagateGate(qc::X, {0});
  ut.propagateGate(qc::H, {1});
  ut.propagateGate(qc::H, {2}, {0});
  ut.propagateGate(qc::H, {3});
  ut.propagateMeasurement(3, 0);

  ASSERT_TRUE(RewriteChecker::isOnlyOneSetNotZero(
      ut, {0, 1, 2, 3},
      {{0}, {1, 2, 3, 4, 5, 6, 7}, {8, 9, 10, 11, 12, 13, 14, 15}}));
}

TEST_F(RewriteCheckerTest, oneSetNonZeroMultipleQubitFalse) {
  ut.propagateQubitAlloc();
  ut.propagateQubitAlloc();
  ut.propagateGate(qc::X, {0});
  ut.propagateGate(qc::H, {1});
  ut.propagateGate(qc::H, {2}, {0});
  ut.propagateGate(qc::X, {3});

  ASSERT_FALSE(RewriteChecker::isOnlyOneSetNotZero(
      ut, {0, 1, 2, 3},
      {{0}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, {15}}));
}

TEST_F(RewriteCheckerTest, findNonSatisfiableCombinationsA) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1});
  ut.propagateGate(qc::X, {1}, {0});

  ASSERT_FALSE(RewriteChecker::areThereSatisfiableCombinations(ut, {0, 1}));
  ASSERT_TRUE(RewriteChecker::areThereSatisfiableCombinations(ut, {0}, {1}));
}

TEST_F(RewriteCheckerTest, findNonSatisfiableCombinationsB) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});

  ASSERT_TRUE(RewriteChecker::areThereSatisfiableCombinations(ut, {0, 1}));
  ASSERT_TRUE(RewriteChecker::areThereSatisfiableCombinations(ut, {}, {0, 1}));
  ASSERT_FALSE(RewriteChecker::areThereSatisfiableCombinations(ut, {0}, {1}));
}

TEST_F(RewriteCheckerTest, findNonSatisfiableCombinationsC) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});
  ut.propagateMeasurement(0, 0);
  ut.propagateMeasurement(1, 1);
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});

  ASSERT_TRUE(
      RewriteChecker::areThereSatisfiableCombinations(ut, {0}, {1}, {0, 1}));
  ASSERT_TRUE(RewriteChecker::areThereSatisfiableCombinations(ut, {}, {0, 1},
                                                              {}, {0, 1}));
  ASSERT_FALSE(
      RewriteChecker::areThereSatisfiableCombinations(ut, {0}, {1}, {0}, {1}));
  ASSERT_FALSE(RewriteChecker::areThereSatisfiableCombinations(ut, {0}, {1}, {},
                                                               {0, 1}));
}

class RewriteCheckerSuperfluousTest : public testing::Test {
protected:
  UnionTable ut;

  RewriteCheckerSuperfluousTest() : ut(UnionTable(8, 4)) {}

  void SetUp() override {
    ut.propagateQubitAlloc();
    ut.propagateQubitAlloc();
    ut.propagateQubitAlloc();
    ut.propagateQubitAlloc();
    ut.propagateQubitAlloc();
    ut.propagateGate(qc::H, {0});
    ut.propagateMeasurement(0, 0);
    ut.propagateGate(qc::H, {0});
    ut.propagateMeasurement(0, 1);
    ut.propagateGate(qc::H, {0});

    ut.propagateGate(qc::H, {1});
    ut.propagateGate(qc::H, {1});
    ut.propagateMeasurement(1, 2); // bit 2 = false

    ut.propagateGate(qc::H, {2});

    ut.propagateGate(qc::H, {3});
    ut.propagateGate(qc::Z, {3});
    ut.propagateGate(qc::H, {3});
    ut.propagateMeasurement(3, 3); // bit 3 = true

    ut.propagateGate(qc::H, {4});
  }

  void TearDown() override {}
};

TEST_F(RewriteCheckerSuperfluousTest, oneSuperfluousEach) {
  auto [superfluousQubits, superfluousBits] =
      RewriteChecker::getSuperfluousControls(ut, {0}, {3, 4}, {1, 2}, {0, 3},
                                             {1, 2});
  ASSERT_EQ(superfluousQubits.size(), 2);
  ASSERT_EQ(superfluousBits.size(), 2);
  ASSERT_TRUE(superfluousQubits.contains(1));
  ASSERT_TRUE(superfluousQubits.contains(3));
  ASSERT_TRUE(superfluousBits.contains(2));
  ASSERT_TRUE(superfluousBits.contains(3));
}

TEST_F(RewriteCheckerSuperfluousTest, targetSuperfluousToNegativeQuantumCtrl) {
  auto [superfluousQubits, _] = RewriteChecker::getSuperfluousControls(
      ut, {0}, {1, 3, 4}, {2}, {0, 3}, {1, 2});
  ASSERT_TRUE(superfluousQubits.contains(0));
}

TEST_F(RewriteCheckerSuperfluousTest,
       targetSuperfluousToPositiveNegQuantumCtrl) {
  auto [superfluousQubits, _] = RewriteChecker::getSuperfluousControls(
      ut, {0}, {4}, {1, 2, 3}, {0, 3}, {1, 2});
  ASSERT_TRUE(superfluousQubits.contains(0));
}

TEST_F(RewriteCheckerSuperfluousTest,
       targetSuperfluousToNegativeClassicalCtrl) {
  auto [superfluousQubits, _] = RewriteChecker::getSuperfluousControls(
      ut, {0}, {3, 4}, {1, 2}, {0, 2, 3}, {1});
  ASSERT_TRUE(superfluousQubits.contains(0));
}

TEST_F(RewriteCheckerSuperfluousTest,
       targetSuperfluousToPositiveNegClassicalCtrl) {
  auto [superfluousQubits, _] = RewriteChecker::getSuperfluousControls(
      ut, {0}, {3, 4}, {1, 2}, {0}, {1, 2, 3});
  ASSERT_TRUE(superfluousQubits.contains(0));
}