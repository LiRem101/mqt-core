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

RewriteChecker::RewriteChecker(UnionTable ut) : unionTable(ut) {}

RewriteChecker::~RewriteChecker() = default;

std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
RewriteChecker::getSuperfluousControls(std::vector<unsigned int> qubitTargets,
                                       std::vector<unsigned int> qubitPosCtrls,
                                       std::vector<unsigned int> qubitNegCtrls,
                                       std::vector<unsigned int> bitPosCtrls,
                                       std::vector<unsigned int> bitNegCtrls) {
  throw std::logic_error("Not implemented");
}

std::optional<unsigned int> RewriteChecker::getEquivalentBit(unsigned int q) {
  throw std::logic_error("Not implemented");
}

std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
RewriteChecker::getImplingQubit(unsigned int q,
                                std::vector<unsigned int> qubits,
                                std::vector<unsigned int> bits) {
  throw std::logic_error("Not implemented");
}

std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
RewriteChecker::getImplingBit(unsigned int b, std::vector<unsigned int> qubits,
                              std::vector<unsigned int> bits) {
  throw std::logic_error("Not implemented");
}

bool RewriteChecker::isOnlyOneSetNotZero(
    std::vector<unsigned int> qubits,
    std::vector<std::vector<unsigned int>> values) {
  throw std::logic_error("Not implemented");
}
