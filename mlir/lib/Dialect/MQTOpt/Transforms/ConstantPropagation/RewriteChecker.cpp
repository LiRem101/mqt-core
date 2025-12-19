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

#include <optional>
#include <stdexcept>

namespace mqt::ir::opt::qcp {
RewriteChecker::RewriteChecker() {}

RewriteChecker::~RewriteChecker() = default;

std::pair<std::set<unsigned int>, std::set<unsigned int>>
RewriteChecker::getSuperfluousControls(UnionTable unionTable,
                                       std::vector<unsigned int> qubitTargets,
                                       std::vector<unsigned int> qubitPosCtrls,
                                       std::vector<unsigned int> qubitNegCtrls,
                                       std::vector<unsigned int> bitPosCtrls,
                                       std::vector<unsigned int> bitNegCtrls) {
  throw std::logic_error("Not implemented");
}

std::optional<std::pair<unsigned int, bool>>
RewriteChecker::getEquivalentBit(UnionTable unionTable, unsigned int q) {
  unsigned int possibleBits = unionTable.getNumberOfBits();
  for (unsigned int i = 0; i < possibleBits; ++i) {
    std::optional<bool> result = unionTable.getIsBitEquivalentToQubit(i, q);
    if (result.has_value()) {
      return std::optional<std::pair<unsigned int, bool>>(
          std::make_pair(i, result.value()));
    }
  }
  return std::optional<std::pair<unsigned int, bool>>();
}

std::pair<std::set<unsigned int>, std::set<unsigned int>>
RewriteChecker::getAntecedentsOfQubit(UnionTable unionTable, unsigned int q,
                                      bool negative,
                                      std::set<unsigned int> qubitsPositive,
                                      std::set<unsigned int> qubitsNegative,
                                      std::set<unsigned int> bitsPositive,
                                      std::set<unsigned int> bitsNegative) {
  throw std::logic_error("Not implemented");
}

std::pair<std::set<unsigned int>, std::set<unsigned int>>
RewriteChecker::getAntecedentsOfBit(UnionTable unionTable, unsigned int b,
                                    bool negative,
                                    std::set<unsigned int> qubitsPositive,
                                    std::set<unsigned int> qubitsNegative,
                                    std::set<unsigned int> bitsPositive,
                                    std::set<unsigned int> bitsNegative) {
  throw std::logic_error("Not implemented");
}

bool RewriteChecker::isOnlyOneSetNotZero(
    UnionTable unionTable, std::vector<unsigned int> qubits,
    std::set<std::set<unsigned int>> values) {
  throw std::logic_error("Not implemented");
}
} // namespace mqt::ir::opt::qcp