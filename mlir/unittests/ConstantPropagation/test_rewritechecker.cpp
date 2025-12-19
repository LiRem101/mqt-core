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

#include <gtest/gtest.h>
#include <mlir/Dialect/MQTOpt/Transforms/Passes.h>

using namespace mqt::ir::opt::qcp;

class RewriteCheckerTest : public ::testing::Test {
protected:
  UnionTable ut;
  RewriteChecker rc = RewriteChecker();

  RewriteCheckerTest() : ut(UnionTable(8, 4)) {}

  void SetUp() override {
    ut.propagateQubitAlloc();
    ut.propagateQubitAlloc();
  }

  void TearDown() override {}
};

// ##################################################
// # Helper functions
// ##################################################

// ##################################################
// # Basic tests
// ##################################################

TEST_F(RewriteCheckerTest, FindEquivalentBit) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});
  ut.propagateMeasurement(1, 0);
  std::optional<std::pair<unsigned int, bool>> result =
      rc.getEquivalentBit(ut, 1);
  ASSERT_TRUE(result.has_value());
  std::pair<unsigned int, bool> resValue = result.value();
  ASSERT_TRUE(resValue.first == 0);
  ASSERT_TRUE(resValue.second);
}

TEST_F(RewriteCheckerTest, FindEquivalentReversedBit) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});
  ut.propagateGate(qc::X, {0});
  ut.propagateMeasurement(0, 0);
  std::optional<std::pair<unsigned int, bool>> result =
      rc.getEquivalentBit(ut, 1);
  ASSERT_TRUE(result.has_value());
  std::pair<unsigned int, bool> resValue = result.value();
  ASSERT_TRUE(resValue.first == 0);
  ASSERT_FALSE(resValue.second);
}

TEST_F(RewriteCheckerTest, FindNoEquivalentBit) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});
  ut.propagateGate(qc::H, {0});
  ut.propagateMeasurement(1, 0);
  std::optional<std::pair<unsigned int, bool>> result =
      rc.getEquivalentBit(ut, 0);
  ASSERT_FALSE(result.has_value());
}

TEST_F(RewriteCheckerTest, ZeroIsAlwaysAntecedent) {
  ut.propagateMeasurement(0, 0);
  ut.propagateGate(qc::H, {1});
  std::pair<std::set<unsigned int>, std::set<unsigned int>> result =
      rc.getAntecedentsOfQubit(ut, 1, false, {0}, {0}, {0}, {0});
  ASSERT_EQ(result.first.size(), 1);
  ASSERT_EQ(result.second.size(), 1);
  ASSERT_EQ(*result.first.begin(), 0);
  ASSERT_EQ(*result.second.begin(), 0);
}

TEST_F(RewriteCheckerTest, ImplyingQubitA) {
  ut.propagateGate(qc::H, {0});
  ut.propagateMeasurement(0, 0);
  ut.propagateGate(qc::H, {1}, {0});
  auto [antecedentQubits, antecedentBits] =
      rc.getAntecedentsOfQubit(ut, 1, true, {}, {0}, {}, {0});
  auto [antecedentQubitsEmptyA, antecedentBitsEmptyA] =
      rc.getAntecedentsOfQubit(ut, 1, false, {0}, {}, {0}, {});
  auto [antecedentQubitsEmptyB, antecedentBitsEmptyB] =
      rc.getAntecedentsOfQubit(ut, 1, false, {}, {0}, {}, {0});
  auto [antecedentQubitsEmptyC, antecedentBitsEmptyC] =
      rc.getAntecedentsOfQubit(ut, 1, true, {0}, {}, {0}, {});

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
      rc.getAntecedentsOfBit(ut, 0, false, {1}, {}, {}, {});
  auto [antecedentQubitsEmptyA, antecedentBitsEmptyA] =
      rc.getAntecedentsOfBit(ut, 0, true, {1}, {}, {}, {});
  auto [antecedentQubitsEmptyB, antecedentBitsEmptyB] =
      rc.getAntecedentsOfBit(ut, 0, false, {}, {1}, {}, {});
  auto [antecedentQubitsEmptyC, antecedentBitsEmptyC] =
      rc.getAntecedentsOfBit(ut, 0, true, {}, {1}, {}, {});

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
  ut.propagateGate(qc::X, {1}, {}, {0});
  ut.propagateGate(qc::H, {1}, {0});
  auto [antecedentQubits, antecedentBits] =
      rc.getAntecedentsOfQubit(ut, 0, false, {}, {}, {0}, {});
  auto [antecedentQubitsEmptyA, antecedentBitsEmptyA] =
      rc.getAntecedentsOfQubit(ut, 0, true, {}, {}, {0}, {});
  auto [antecedentQubitsEmptyB, antecedentBitsEmptyB] =
      rc.getAntecedentsOfQubit(ut, 0, false, {}, {}, {}, {0});
  auto [antecedentQubitsEmptyC, antecedentBitsEmptyC] =
      rc.getAntecedentsOfQubit(ut, 0, true, {}, {}, {}, {0});

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
  auto [antecedentQubits, antecedentBits] =
      rc.getAntecedentsOfBit(ut, 0, true, {}, {0}, {}, {});
  auto [antecedentQubitsEmptyA, antecedentBitsEmptyA] =
      rc.getAntecedentsOfBit(ut, 0, false, {0}, {}, {}, {});
  auto [antecedentQubitsEmptyB, antecedentBitsEmptyB] =
      rc.getAntecedentsOfBit(ut, 0, false, {}, {0}, {}, {});
  auto [antecedentQubitsEmptyC, antecedentBitsEmptyC] =
      rc.getAntecedentsOfBit(ut, 0, true, {0}, {}, {}, {});

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
      rc.getAntecedentsOfQubit(ut, 1, false, {}, {}, {}, {0});
  auto [antecedentQubitsEmptyA, antecedentBitsEmptyA] =
      rc.getAntecedentsOfQubit(ut, 1, false, {}, {}, {0}, {});
  auto [antecedentQubitsEmptyB, antecedentBitsEmptyB] =
      rc.getAntecedentsOfQubit(ut, 1, false, {}, {}, {0}, {});
  auto [antecedentQubitsEmptyC, antecedentBitsEmptyC] =
      rc.getAntecedentsOfQubit(ut, 1, true, {}, {}, {}, {0});

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
      rc.getAntecedentsOfBit(ut, 0, false, {}, {1}, {}, {});
  auto [antecedentQubitsEmptyA, antecedentBitsEmptyA] =
      rc.getAntecedentsOfBit(ut, 0, false, {1}, {}, {}, {});
  auto [antecedentQubitsEmptyB, antecedentBitsEmptyB] =
      rc.getAntecedentsOfBit(ut, 0, true, {1}, {}, {}, {});
  auto [antecedentQubitsEmptyC, antecedentBitsEmptyC] =
      rc.getAntecedentsOfBit(ut, 0, true, {}, {1}, {}, {});

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
      rc.getAntecedentsOfQubit(ut, 1, true, {}, {}, {0}, {});
  auto [antecedentQubitsEmptyA, antecedentBitsEmptyA] =
      rc.getAntecedentsOfQubit(ut, 1, false, {}, {}, {0}, {});
  auto [antecedentQubitsEmptyB, antecedentBitsEmptyB] =
      rc.getAntecedentsOfQubit(ut, 1, false, {}, {}, {}, {0});
  auto [antecedentQubitsEmptyC, antecedentBitsEmptyC] =
      rc.getAntecedentsOfQubit(ut, 1, true, {}, {}, {}, {0});

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

TEST_F(RewriteCheckerTest, ImplyingBitD) {
  ut.propagateGate(qc::H, {0});
  ut.propagateMeasurement(0, 0);
  ut.propagateGate(qc::X, {0});
  ut.propagateGate(qc::H, {1}, {0});
  auto [antecedentQubits, antecedentBits] =
      rc.getAntecedentsOfBit(ut, 0, true, {1}, {}, {}, {});
  auto [antecedentQubitsEmptyA, antecedentBitsEmptyA] =
      rc.getAntecedentsOfBit(ut, 0, false, {1}, {}, {}, {});
  auto [antecedentQubitsEmptyB, antecedentBitsEmptyB] =
      rc.getAntecedentsOfBit(ut, 0, true, {}, {1}, {}, {});
  auto [antecedentQubitsEmptyC, antecedentBitsEmptyC] =
      rc.getAntecedentsOfBit(ut, 0, false, {}, {1}, {}, {});

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

  ASSERT_FALSE(rc.isOnlyOneSetNotZero(ut, {0}, {{0}, {1}}));
  ASSERT_TRUE(rc.isOnlyOneSetNotZero(ut, {0}, {{0}, {1}}));
}

TEST_F(RewriteCheckerTest, oneSetNonZeroTwoQubitFalse) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::H, {1});

  ASSERT_FALSE(rc.isOnlyOneSetNotZero(ut, {0, 1}, {{0, 1, 2}, {3}}));
}

TEST_F(RewriteCheckerTest, oneSetNonZeroTwoQubitTrue) {
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1}, {0});
  ut.propagateMeasurement(0, 0);

  ASSERT_TRUE(rc.isOnlyOneSetNotZero(ut, {0, 1}, {{0, 1, 2}, {3}}));
}

TEST_F(RewriteCheckerTest, findNonSatisfiableCombinations) {
  ut.propagateQubitAlloc();
  ut.propagateGate(qc::H, {0});
  ut.propagateGate(qc::X, {1});
  ut.propagateGate(qc::X, {1}, {0});

  std::pair<std::set<unsigned int>, std::set<unsigned int>> superfluousControls(
      {2}, {0, 1});
  ASSERT_TRUE(superfluousControls.first.contains(2));
}

class RewriteCheckerSuperfluousTest : public ::testing::Test {
protected:
  UnionTable ut;
  RewriteChecker rc = RewriteChecker();

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
    ut.propagateGate(qc::X, {2}, {1});
    ut.propagateGate(qc::H, {1});
    ut.propagateMeasurement(1, 2); // bit 2 = false

    ut.propagateGate(qc::H, {3});
    ut.propagateGate(qc::X, {4}, {3});
    ut.propagateGate(qc::Z, {3});
    ut.propagateGate(qc::H, {3});
    ut.propagateMeasurement(3, 3); // bit 3 = true
  }

  void TearDown() override {}
};

TEST_F(RewriteCheckerSuperfluousTest, oneSuperfluousEach) {
  std::pair<std::set<unsigned int>, std::set<unsigned int>>
      superfluousControls =
          rc.getSuperfluousControls(ut, {0}, {3, 4}, {1, 2}, {0, 3}, {1, 2});
  ASSERT_EQ(superfluousControls.first.size(), 2);
  ASSERT_EQ(superfluousControls.second.size(), 2);
  ASSERT_TRUE(superfluousControls.first.contains(1));
  ASSERT_TRUE(superfluousControls.first.contains(3));
  ASSERT_TRUE(superfluousControls.second.contains(2));
  ASSERT_TRUE(superfluousControls.second.contains(3));
}

TEST_F(RewriteCheckerSuperfluousTest, targetSuperfluousToNegativeQuantumCtrl) {
  std::pair<std::set<unsigned int>, std::set<unsigned int>>
      superfluousControls =
          rc.getSuperfluousControls(ut, {0}, {1, 3, 4}, {2}, {0, 3}, {1, 2});
  ASSERT_TRUE(superfluousControls.first.contains(0));
}

TEST_F(RewriteCheckerSuperfluousTest,
       targetSuperfluousToPositiveNegQuantumCtrl) {
  std::pair<std::set<unsigned int>, std::set<unsigned int>>
      superfluousControls =
          rc.getSuperfluousControls(ut, {0}, {4}, {1, 2, 3}, {0, 3}, {1, 2});
  ASSERT_TRUE(superfluousControls.first.contains(0));
}

TEST_F(RewriteCheckerSuperfluousTest,
       targetSuperfluousToNegativeClassicalCtrl) {
  std::pair<std::set<unsigned int>, std::set<unsigned int>>
      superfluousControls =
          rc.getSuperfluousControls(ut, {0}, {3, 4}, {1, 2}, {0, 2, 3}, {1});
  ASSERT_TRUE(superfluousControls.first.contains(0));
}

TEST_F(RewriteCheckerSuperfluousTest,
       targetSuperfluousToPositiveNegClassicalCtrl) {
  std::pair<std::set<unsigned int>, std::set<unsigned int>>
      superfluousControls =
          rc.getSuperfluousControls(ut, {0}, {3, 4}, {1, 2}, {0}, {1, 2, 3});
  ASSERT_TRUE(superfluousControls.first.contains(0));
}