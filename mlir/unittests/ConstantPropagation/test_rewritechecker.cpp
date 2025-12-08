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

namespace {

class RewriteCheckerTest : public ::testing::Test {
protected:
  UnionTable* unionTable = nullptr;
  RewriteChecker* checker = nullptr;

  void SetUp() override {
    // unionTable = new UnionTable(1, 1);
    // checker = new RewriteChecker(*unionTable);
  }

  void TearDown() override {
    // delete checker;
    // delete unionTable;
  }
};

// ##################################################
// # Helper functions
// ##################################################

// ##################################################
// # Basic tests
// ##################################################

TEST_F(RewriteCheckerTest, FirstTest) { ASSERT_TRUE(false); }

} // namespace
