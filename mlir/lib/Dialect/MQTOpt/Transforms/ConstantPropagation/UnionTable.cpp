/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/Transforms/ConstantPropagation/UnionTable.hpp"

namespace mqt::ir::opt::qcp {
UnionTable::UnionTable(unsigned int maxNonzeroAmplitudes,
                       unsigned int maxNumberOfBitValues)
    : maxNonzeroAmplitudes(maxNonzeroAmplitudes),
      maxNumberOfBitValues(maxNumberOfBitValues),
      mappingGlobalToLocalQubitIndices({}), mappingGlobalToLocalBitIndices({}),
      hRegOfQubits({}), hRegOfBits({}), indizesInSameState({}) {}

UnionTable::~UnionTable() = default;

std::string UnionTable::toString() const {
  std::string result;
  for (const std::pair<std::vector<unsigned int>, std::vector<unsigned int>>&
           qubitAndBitIndices : indizesInSameState) {
    std::vector<unsigned int> qubitIndices = qubitAndBitIndices.first;
    std::vector<unsigned int> bitIndices = qubitAndBitIndices.second;

    result += "Qubits: ";
    for (int i = static_cast<int>(qubitIndices.size()) - 1; i >= 0; i--) {
      result += std::to_string(qubitIndices.at(i));
    }
    if (!bitIndices.empty()) {
      result += ", Bits: ";
    }
    for (int i = static_cast<int>(bitIndices.size()) - 1; i >= 0; i--) {
      result += std::to_string(bitIndices.at(i));
    }

    result += ", HybridStates: {";
    bool first = true;
    for (HybridStateOrTop const& hs : *(hRegOfQubits.at(qubitIndices.at(0)))) {
      if (!first) {
        result += ", ";
      }
      first = false;
      result += hs.toString();
    }
    result += "}\n";
  }
  return result;
}

void UnionTable::unify(std::vector<unsigned int> qubits,
                       std::vector<unsigned int> bits) {
  throw std::logic_error("Not implemented");
}

bool UnionTable::allTop() { throw std::logic_error("Not implemented"); }

void UnionTable::propagateGate(
    qc::OpType gate, std::vector<unsigned int> targets,
    std::vector<unsigned int> posCtrlsQuantum,
    std::vector<unsigned int> negCtrlsQuantum,
    const std::vector<unsigned int>& posCtrlsClassical,
    const std::vector<unsigned int>& negCtrlsClassical,
    std::vector<double> params) {
  throw std::logic_error("Not implemented");
}

void UnionTable::propagateMeasurement(unsigned int quantumTarget,
                                      unsigned int classicalTarget) {
  throw std::logic_error("Not implemented");
}

void UnionTable::propagateReset(unsigned int target) {
  throw std::logic_error("Not implemented");
}

unsigned int UnionTable::propagateQubitAlloc() {
  unsigned int qubitIndex = mappingGlobalToLocalQubitIndices.size();
  mappingGlobalToLocalQubitIndices.push_back(0);
  HybridState hs =
      HybridState(1, maxNonzeroAmplitudes, maxNumberOfBitValues, {}, 1.0);
  std::vector<HybridStateOrTop> setForNewQubit = {
      HybridStateOrTop(std::make_shared<HybridState>(hs))};
  hRegOfQubits.push_back(
      std::make_shared<std::vector<HybridStateOrTop>>(setForNewQubit));
  indizesInSameState.insert({{qubitIndex}, {}});
  return qubitIndex;
}

void UnionTable::propagateQubitDealloc(unsigned int target) {
  throw std::logic_error("Not implemented");
}

unsigned int UnionTable::propagateBitDef(bool value) {
  throw std::logic_error("Not implemented");
}

bool UnionTable::isQubitAlwaysOne(size_t q) {
  throw std::logic_error("Not implemented");
}

bool UnionTable::isQubitAlwaysZero(size_t q) {
  throw std::logic_error("Not implemented");
}

bool UnionTable::isBitAlwaysOne(size_t q) {
  throw std::logic_error("Not implemented");
}

bool UnionTable::isBitAlwaysZero(size_t q) {
  throw std::logic_error("Not implemented");
}

bool UnionTable::hasAlwaysZeroAmplitude(std::vector<unsigned int> qubits,
                                        unsigned int value) {
  throw std::logic_error("Not implemented");
}
} // namespace mqt::ir::opt::qcp