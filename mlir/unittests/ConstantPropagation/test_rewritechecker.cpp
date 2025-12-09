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
  UnionTable unionTable;
  RewriteChecker checker;

  RewriteCheckerTest()
      : unionTable(UnionTable()), checker(RewriteChecker(unionTable)) {}

  void SetUp() override {}

  void TearDown() override {}
};

// ##################################################
// # Helper functions
// ##################################################

// ##################################################
// # Basic tests
// ##################################################

TEST_F(RewriteCheckerTest, FirstTest) { ASSERT_TRUE(false); }
