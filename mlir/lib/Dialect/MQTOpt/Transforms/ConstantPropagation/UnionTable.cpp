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
  for (const std::pair<std::set<unsigned int>, std::set<unsigned int>>&
           qubitAndBitIndices : indizesInSameState) {
    std::set<unsigned int> qubitIndices = qubitAndBitIndices.first;
    std::set<unsigned int> bitIndices = qubitAndBitIndices.second;

    result += "Qubits: ";
    for (auto qit = qubitIndices.rbegin(); qit != qubitIndices.rend(); ++qit) {
      result += std::to_string(*qit);
    }
    if (!bitIndices.empty()) {
      result += ", Bits: ";
    }
    for (auto bit = bitIndices.rbegin(); bit != bitIndices.rend(); ++bit) {
      result += std::to_string(*bit);
    }

    result += ", HybridStates: {";
    bool first = true;
    for (HybridStateOrTop const& hs :
         *(hRegOfQubits.at(*qubitIndices.begin()))) {
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
  std::vector<unsigned int> qubits;
  qubits.insert(qubits.end(), targets.begin(), targets.end());
  qubits.insert(qubits.end(), posCtrlsQuantum.begin(), posCtrlsQuantum.end());
  qubits.insert(qubits.end(), negCtrlsQuantum.begin(), negCtrlsQuantum.end());
  std::vector<unsigned int> bits;
  bits.insert(bits.end(), posCtrlsClassical.begin(), posCtrlsClassical.end());
  bits.insert(bits.end(), negCtrlsClassical.begin(), negCtrlsClassical.end());

  std::set<std::pair<std::set<unsigned int>, std::set<unsigned int>>>
      involvedStates;
  for (std::pair<std::set<unsigned int>, std::set<unsigned int>> const&
           setsInSameState : indizesInSameState) {
    std::set<unsigned int> qubitIndizes = setsInSameState.first;
    std::set<unsigned int> bitIndizes = setsInSameState.second;
    if (std::ranges::find_first_of(qubitIndizes, qubits) !=
            qubitIndizes.end() ||
        std::ranges::find_first_of(bitIndizes, bits) != bitIndizes.end()) {
      involvedStates.insert(setsInSameState);
    }
  }
  if (involvedStates.size() > 1) {
    std::pair<std::set<unsigned int>, std::set<unsigned int>> currentStates =
        *involvedStates.begin();
    auto nextStateIt = ++involvedStates.begin();
    while (nextStateIt != involvedStates.end()) {
      currentStates = unifyHybridStates(currentStates, *nextStateIt);
      nextStateIt++;
    }
  }

  std::vector<unsigned int> targetsLocal = {};
  std::vector<unsigned int> posCtrlsQuantumLocal = {};
  std::vector<unsigned int> negCtrlsQuantumLocal = {};
  std::vector<unsigned int> posCtrlsClassicalLocal = {};
  std::vector<unsigned int> negCtrlsClassicalLocal = {};

  for (unsigned int const target : targets) {
    targetsLocal.push_back(mappingGlobalToLocalQubitIndices.at(target));
  }
  for (unsigned int const posCtrl : posCtrlsQuantum) {
    posCtrlsQuantumLocal.push_back(
        mappingGlobalToLocalQubitIndices.at(posCtrl));
  }
  for (unsigned int const negCtrl : negCtrlsQuantum) {
    negCtrlsQuantumLocal.push_back(
        mappingGlobalToLocalQubitIndices.at(negCtrl));
  }
  for (unsigned int const posCtrl : posCtrlsClassical) {
    posCtrlsClassicalLocal.push_back(
        mappingGlobalToLocalBitIndices.at(posCtrl));
  }
  for (unsigned int const negCtrl : negCtrlsClassical) {
    negCtrlsClassicalLocal.push_back(
        mappingGlobalToLocalBitIndices.at(negCtrl));
  }

  for (unsigned int i = 0; i < hRegOfQubits.at(targets.at(0))->size(); i++) {
    const HybridStateOrTop hs = hRegOfQubits.at(targets.at(0))->at(i);
    if (hs.isHybridState()) {
      try {
        hs.getHybridState()->propagateGate(
            gate, targetsLocal, posCtrlsQuantumLocal, negCtrlsQuantumLocal,
            posCtrlsClassicalLocal, negCtrlsClassicalLocal, params);
      } catch (std::domain_error const&) {
        hs.getHybridState().reset();
        hRegOfQubits.at(targets.at(0))->at(i).getHybridState().reset();
        hRegOfQubits.at(targets.at(0))->at(i) = HybridStateOrTop(T);
      }
    }
  }
}

void UnionTable::propagateMeasurement(unsigned int quantumTarget,
                                      unsigned int classicalTarget) {

  std::set<std::pair<std::set<unsigned int>, std::set<unsigned int>>>
      involvedStates;
  bool bitExists = false;
  for (std::pair<std::set<unsigned int>, std::set<unsigned int>> const&
           setsInSameState : indizesInSameState) {
    std::set<unsigned int> qubitIndizes = setsInSameState.first;
    std::set<unsigned int> bitIndizes = setsInSameState.second;
    if (qubitIndizes.contains(quantumTarget) ||
        bitIndizes.contains(classicalTarget)) {
      involvedStates.insert(setsInSameState);
      bitExists |= bitIndizes.contains(classicalTarget);
    }
  }
  std::pair<std::set<unsigned int>, std::set<unsigned int>>
      toChangedHybridStates = *involvedStates.begin();
  if (involvedStates.size() > 1) {
    toChangedHybridStates =
        unifyHybridStates(*involvedStates.begin(), *++involvedStates.begin());
  }
  if (!bitExists && classicalTarget != hRegOfBits.size()) {
    throw std::invalid_argument("New classical bit index needs to be equal to "
                                "number of existing bits.");
  }
  std::vector<HybridStateOrTop> involvedHybridStates =
      *hRegOfQubits.at(quantumTarget);
  unsigned int localBitIndex = 0;
  if (!bitExists) {
    bool top = false;
    for (HybridStateOrTop hs : involvedHybridStates) {
      if (hs.isTop()) {
        top = true;
      } else {
        try {
          localBitIndex = hs.getHybridState()->addClassicalBit();
        } catch (std::domain_error&) {
          top = true;
        }
      }
    }
    mappingGlobalToLocalBitIndices.push_back(localBitIndex);
    if (top) {
      for (HybridStateOrTop hs : involvedHybridStates) {
        if (hs.isHybridState()) {
          hs.getHybridState().reset();
        }
      }
      involvedHybridStates.clear();
      involvedHybridStates.push_back({HybridStateOrTop(T)});
    }

    // TODO: Propagate measurement into the states in involvedHybridStates

    for (unsigned int qubitIndizes : toChangedHybridStates.first) {
      hRegOfQubits.at(qubitIndizes) =
          std::make_shared<std::vector<HybridStateOrTop>>(involvedHybridStates);
    }
    for (unsigned int bitIndizes : toChangedHybridStates.second) {
      hRegOfBits.at(bitIndizes) =
          std::make_shared<std::vector<HybridStateOrTop>>(involvedHybridStates);
    }
  }
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
  unsigned int bitIndex = mappingGlobalToLocalBitIndices.size();
  mappingGlobalToLocalBitIndices.push_back(0);
  HybridState hs =
      HybridState(0, maxNonzeroAmplitudes, maxNumberOfBitValues, {value}, 1.0);
  std::vector<HybridStateOrTop> setForNewBit = {
      HybridStateOrTop(std::make_shared<HybridState>(hs))};
  hRegOfBits.push_back(
      std::make_shared<std::vector<HybridStateOrTop>>(setForNewBit));
  indizesInSameState.insert({{}, {bitIndex}});
  return bitIndex;
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