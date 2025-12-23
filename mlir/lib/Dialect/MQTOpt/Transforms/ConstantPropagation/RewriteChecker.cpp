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
  std::set<unsigned int> superfluousQubits = {};
  std::set<unsigned int> superfluousBits = {};
  for (unsigned int posCtrlQubit : qubitPosCtrls) {
    if (unionTable.isQubitAlwaysOne(posCtrlQubit)) {
      superfluousQubits.insert(posCtrlQubit);
    } else if (unionTable.isQubitAlwaysZero(posCtrlQubit)) {
      superfluousQubits.clear();
      std::for_each(qubitTargets.cbegin(), qubitTargets.cend(),
                    [&](unsigned int i) { superfluousQubits.insert(i); });
      return {superfluousQubits, {}};
    }
  }
  for (unsigned int negCtrlQubit : qubitNegCtrls) {
    if (unionTable.isQubitAlwaysZero(negCtrlQubit)) {
      superfluousQubits.insert(negCtrlQubit);
    } else if (unionTable.isQubitAlwaysOne(negCtrlQubit)) {
      superfluousQubits.clear();
      std::for_each(qubitTargets.cbegin(), qubitTargets.cend(),
                    [&](unsigned int i) { superfluousQubits.insert(i); });
      return {superfluousQubits, {}};
    }
  }
  for (unsigned int posCtrlBit : bitPosCtrls) {
    if (unionTable.isBitAlwaysOne(posCtrlBit)) {
      superfluousBits.insert(posCtrlBit);
    } else if (unionTable.isBitAlwaysZero(posCtrlBit)) {
      superfluousQubits.clear();
      std::for_each(qubitTargets.cbegin(), qubitTargets.cend(),
                    [&](unsigned int i) { superfluousQubits.insert(i); });
      return {superfluousQubits, {}};
    }
  }
  for (unsigned int negCtrlBit : bitNegCtrls) {
    if (unionTable.isBitAlwaysZero(negCtrlBit)) {
      superfluousBits.insert(negCtrlBit);
    } else if (unionTable.isBitAlwaysOne(negCtrlBit)) {
      superfluousQubits.clear();
      std::for_each(qubitTargets.cbegin(), qubitTargets.cend(),
                    [&](unsigned int i) { superfluousQubits.insert(i); });
      return {superfluousQubits, {}};
    }
  }
  return {superfluousQubits, superfluousBits};
}
bool RewriteChecker::areThereSatisfiableCombinations(
    UnionTable unionTable, std::vector<unsigned int> qubitPosCtrls,
    std::vector<unsigned int> qubitNegCtrls,
    std::vector<unsigned int> bitPosCtrls,
    std::vector<unsigned int> bitNegCtrls) {
  std::vector<unsigned int> qubits = qubitNegCtrls;
  unsigned int value = 0;
  for (unsigned int i = qubitNegCtrls.size();
       i < qubitNegCtrls.size() + qubitPosCtrls.size(); ++i) {
    value += static_cast<unsigned int>(pow(2, i) + 0.1);
    qubits.push_back(qubitPosCtrls.at(i - qubitNegCtrls.size()));
  }
  std::vector<unsigned int> bits = {};
  std::vector<bool> bitValues = {};
  for (unsigned int posBit : bitPosCtrls) {
    bits.push_back(posBit);
    bitValues.push_back(true);
  }
  for (unsigned int negBit : bitNegCtrls) {
    bits.push_back(negBit);
    bitValues.push_back(false);
  }
  return !unionTable.hasAlwaysZeroAmplitude(qubits, value, bits, bitValues);
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
  std::pair<std::set<unsigned int>, std::set<unsigned int>> result;
  // A value is antecedent of qubit, if the combination antecedent = 1 and qubit
  // = 0 does not exist
  unsigned int qubitValueToCheck = negative ? 1 : 0;
  for (unsigned int posQubit : qubitsPositive) {
    bool hasZeroAmplitude =
        unionTable.hasAlwaysZeroAmplitude({q, posQubit}, qubitValueToCheck + 2);
    if (hasZeroAmplitude) {
      result.first.insert(posQubit);
    }
  }
  for (unsigned int negQubit : qubitsNegative) {
    bool hasZeroAmplitude =
        unionTable.hasAlwaysZeroAmplitude({q, negQubit}, qubitValueToCheck);
    if (hasZeroAmplitude) {
      result.first.insert(negQubit);
    }
  }
  for (unsigned int posBit : bitsPositive) {
    bool hasZeroAmplitude = unionTable.hasAlwaysZeroAmplitude(
        {q}, qubitValueToCheck, {posBit}, {true});
    if (hasZeroAmplitude) {
      result.second.insert(posBit);
    }
  }
  for (unsigned int negBit : bitsNegative) {
    bool hasZeroAmplitude = unionTable.hasAlwaysZeroAmplitude(
        {q}, qubitValueToCheck, {negBit}, {false});
    if (hasZeroAmplitude) {
      result.second.insert(negBit);
    }
  }
  return result;
}

std::pair<std::set<unsigned int>, std::set<unsigned int>>
RewriteChecker::getAntecedentsOfBit(UnionTable unionTable, unsigned int b,
                                    bool negative,
                                    std::set<unsigned int> qubitsPositive,
                                    std::set<unsigned int> qubitsNegative,
                                    std::set<unsigned int> bitsPositive,
                                    std::set<unsigned int> bitsNegative) {
  std::pair<std::set<unsigned int>, std::set<unsigned int>> result;
  // A value is antecedent of bit, if the combination antecedent = 1 and bit = 0
  // does not exist
  for (unsigned int posQubit : qubitsPositive) {
    bool hasZeroAmplitude =
        unionTable.hasAlwaysZeroAmplitude({posQubit}, 1, {b}, {negative});
    if (hasZeroAmplitude) {
      result.first.insert(posQubit);
    }
  }
  for (unsigned int negQubit : qubitsNegative) {
    bool hasZeroAmplitude =
        unionTable.hasAlwaysZeroAmplitude({negQubit}, 0, {b}, {negative});
    if (hasZeroAmplitude) {
      result.first.insert(negQubit);
    }
  }
  for (unsigned int posBit : bitsPositive) {
    bool hasZeroAmplitude =
        unionTable.hasAlwaysZeroAmplitude({}, 0, {posBit, b}, {true, negative});
    if (hasZeroAmplitude) {
      result.second.insert(posBit);
    }
  }
  for (unsigned int negBit : bitsNegative) {
    bool hasZeroAmplitude = unionTable.hasAlwaysZeroAmplitude(
        {}, 0, {negBit, b}, {false, negative});
    if (hasZeroAmplitude) {
      result.second.insert(negBit);
    }
  }
  return result;
}

bool RewriteChecker::isOnlyOneSetNotZero(
    UnionTable unionTable, std::vector<unsigned int> qubits,
    std::set<std::vector<unsigned int>> values) {
  return unionTable.isOnlyOneSetNotZero(qubits, values);
}
} // namespace mqt::ir::opt::qcp