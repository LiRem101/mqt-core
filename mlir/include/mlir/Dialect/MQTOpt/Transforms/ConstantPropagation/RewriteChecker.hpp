/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_CORE_REWRITECHECKER_H
#define MQT_CORE_REWRITECHECKER_H

#include "UnionTable.hpp"

#include <optional>

class RewriteChecker {
public:
  explicit RewriteChecker(UnionTable ut);

  ~RewriteChecker();

  std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
  getSuperfluousControls(std::vector<unsigned int> qubitTargets,
                         std::vector<unsigned int> qubitPosCtrls,
                         std::vector<unsigned int> qubitNegCtrls,
                         std::vector<unsigned int> bitPosCtrls,
                         std::vector<unsigned int> bitNegCtrls);

  std::optional<unsigned int> getEquivalentBit(unsigned int q);

  std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
  getImplingQubit(unsigned int q, std::vector<unsigned int> qubits,
                  std::vector<unsigned int> bits);

  std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
  getImplingBit(unsigned int b, std::vector<unsigned int> qubits,
                std::vector<unsigned int> bits);

  bool isOnlyOneSetNotZero(std::vector<unsigned int> qubits,
                           std::vector<std::vector<unsigned int>> values);

private:
  UnionTable unionTable;
};

#endif // MQT_CORE_REWRITECHECKER_H
