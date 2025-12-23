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
std::set<std::pair<std::set<unsigned int>, std::set<unsigned int>>>
UnionTable::getInvolvedStates(std::set<unsigned int> qubits,
                              std::set<unsigned int> bits) {
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
  return involvedStates;
}

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
        result += " ";
      }
      first = false;
      result += hs.toString();
    }
    result += "}\n";
  }
  return result;
}

unsigned int UnionTable::getNumberOfBits() {
  return mappingGlobalToLocalBitIndices.size();
}

void UnionTable::propagateGate(
    qc::OpType gate, std::vector<unsigned int> targets,
    std::vector<unsigned int> posCtrlsQuantum,
    std::vector<unsigned int> negCtrlsQuantum,
    const std::vector<unsigned int>& posCtrlsClassical,
    const std::vector<unsigned int>& negCtrlsClassical,
    std::vector<double> params) {
  std::set<unsigned int> qubits;
  qubits.insert(targets.begin(), targets.end());
  qubits.insert(posCtrlsQuantum.begin(), posCtrlsQuantum.end());
  qubits.insert(negCtrlsQuantum.begin(), negCtrlsQuantum.end());
  std::set<unsigned int> bits;
  bits.insert(posCtrlsClassical.begin(), posCtrlsClassical.end());
  bits.insert(negCtrlsClassical.begin(), negCtrlsClassical.end());

  std::set<std::pair<std::set<unsigned int>, std::set<unsigned int>>>
      involvedStates = getInvolvedStates(qubits, bits);
  if (involvedStates.size() > 1) {
    std::pair<std::set<unsigned int>, std::set<unsigned int>> currentStates =
        *involvedStates.begin();
    auto nextStateIt = ++involvedStates.begin();
    while (nextStateIt != involvedStates.end()) {
      currentStates = unifyHybridStates(currentStates, *nextStateIt);
      ++nextStateIt;
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
      involvedStates = getInvolvedStates({quantumTarget}, {classicalTarget});
  bool bitExists = false;
  for (const auto& [_, bitsInState] : involvedStates) {
    bitExists |= bitsInState.contains(classicalTarget);
  }

  std::pair<std::set<unsigned int>, std::set<unsigned int>>
      involvedStateIndizes = *involvedStates.begin();
  indizesInSameState.erase(*involvedStates.begin());
  if (involvedStates.size() > 1) {
    indizesInSameState.erase(*++involvedStates.begin());
    involvedStateIndizes =
        unifyHybridStates(*involvedStates.begin(), *++involvedStates.begin());
    indizesInSameState.insert(involvedStateIndizes);
  }
  if (!bitExists && classicalTarget != hRegOfBits.size()) {
    throw std::invalid_argument("New classical bit index needs to be equal to "
                                "number of existing bits.");
  }
  std::vector<HybridStateOrTop> involvedHybridStates =
      *hRegOfQubits.at(quantumTarget);
  bool top = false;
  if (!bitExists) {
    involvedStateIndizes.second.insert(mappingGlobalToLocalBitIndices.size());
    indizesInSameState.insert(involvedStateIndizes);
    unsigned int localBitIndex = 0;
    for (HybridStateOrTop const& hs : involvedHybridStates) {
      if (hs.isTop()) {
        top = true;
      } else {
        try {
          localBitIndex = hs.getHybridState()->addClassicalBit();
        } catch (std::domain_error const&) {
          top = true;
        }
      }
    }
    mappingGlobalToLocalBitIndices.push_back(localBitIndex);
    if (top) {
      for (HybridStateOrTop const& hs : involvedHybridStates) {
        if (hs.isHybridState()) {
          hs.getHybridState().reset();
        }
      }
      involvedHybridStates.clear();
      involvedHybridStates.push_back({HybridStateOrTop(T)});
    }
    hRegOfBits.push_back(
        std::make_shared<std::vector<HybridStateOrTop>>(involvedHybridStates));
  }

  // Propagate measurement into the states in involvedHybridStates
  std::vector<HybridStateOrTop> newHybridStatesOrTops;
  for (HybridStateOrTop const& hs :
       *hRegOfQubits.at(*involvedStateIndizes.first.begin())) {
    if (!top && hs.isHybridState()) {
      try {
        std::vector<HybridState> newHybridState =
            hs.getHybridState()->propagateMeasurement(
                mappingGlobalToLocalQubitIndices.at(quantumTarget),
                mappingGlobalToLocalBitIndices.at(classicalTarget));
        for (HybridState const& newState : newHybridState) {
          newHybridStatesOrTops.push_back(
              HybridStateOrTop(std::make_shared<HybridState>(newState)));
        }
      } catch (std::domain_error const&) {
        top = true;
      }
    } else {
      top = true;
    }
  }
  if (top) {
    for (HybridStateOrTop const& hs : newHybridStatesOrTops) {
      if (hs.isHybridState()) {
        hs.getHybridState().reset();
      }
    }
    newHybridStatesOrTops.clear();
    newHybridStatesOrTops.push_back({HybridStateOrTop(T)});
  }

  for (unsigned int const qubitIndizes : involvedStateIndizes.first) {
    hRegOfQubits.at(qubitIndizes) =
        std::make_shared<std::vector<HybridStateOrTop>>(newHybridStatesOrTops);
  }
  for (unsigned int const bitIndizes : involvedStateIndizes.second) {
    hRegOfBits.at(bitIndizes) =
        std::make_shared<std::vector<HybridStateOrTop>>(newHybridStatesOrTops);
  }
}

void UnionTable::propagateReset(unsigned int target) {
  std::pair<std::set<unsigned int>, std::set<unsigned int>> involvedIndizes =
      *getInvolvedStates({target}, {}).begin();

  bool top = false;

  // Propagate reset into the states in involvedHybridStates
  std::vector<HybridStateOrTop> newHybridStatesOrTops;
  for (HybridStateOrTop const& hs : *hRegOfQubits.at(target)) {
    if (!top && hs.isHybridState()) {
      try {
        std::vector<HybridState> newHybridState =
            hs.getHybridState()->propagateReset(target);
        for (HybridState const& newState : newHybridState) {
          newHybridStatesOrTops.push_back(
              HybridStateOrTop(std::make_shared<HybridState>(newState)));
        }
      } catch (std::domain_error const&) {
        top = true;
      }
    } else {
      top = true;
    }
  }
  if (top) {
    for (HybridStateOrTop const& hs : newHybridStatesOrTops) {
      if (hs.isHybridState()) {
        hs.getHybridState().reset();
      }
    }
    newHybridStatesOrTops.clear();
    newHybridStatesOrTops.push_back({HybridStateOrTop(T)});
  }

  for (unsigned int const qubitIndizes : involvedIndizes.first) {
    hRegOfQubits.at(qubitIndizes) =
        std::make_shared<std::vector<HybridStateOrTop>>(newHybridStatesOrTops);
  }
  for (unsigned int const bitIndizes : involvedIndizes.second) {
    hRegOfBits.at(bitIndizes) =
        std::make_shared<std::vector<HybridStateOrTop>>(newHybridStatesOrTops);
  }
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
  for (HybridStateOrTop hs : *hRegOfQubits.at(q)) {
    if (!hs.isQubitAlwaysOne(mappingGlobalToLocalQubitIndices.at(q))) {
      return false;
    }
  }
  return true;
}

bool UnionTable::isQubitAlwaysZero(size_t q) {
  for (HybridStateOrTop hs : *hRegOfQubits.at(q)) {
    if (!hs.isQubitAlwaysZero(mappingGlobalToLocalQubitIndices.at(q))) {
      return false;
    }
  }
  return true;
}

bool UnionTable::isBitAlwaysOne(size_t q) {
  for (HybridStateOrTop hs : *hRegOfBits.at(q)) {
    if (!hs.isBitAlwaysOne(mappingGlobalToLocalBitIndices.at(q))) {
      return false;
    }
  }
  return true;
}

bool UnionTable::isBitAlwaysZero(size_t q) {
  for (HybridStateOrTop hs : *hRegOfBits.at(q)) {
    if (!hs.isBitAlwaysZero(mappingGlobalToLocalBitIndices.at(q))) {
      return false;
    }
  }
  return true;
}

bool UnionTable::allTop() {
  for (const auto& [qubitIndizes, bitIndizes] : indizesInSameState) {
    std::vector<HybridStateOrTop> states;
    if (!qubitIndizes.empty()) {
      states = *hRegOfQubits.at(*qubitIndizes.begin());
    } else {
      states = *hRegOfBits.at(*bitIndizes.begin());
    }
    for (HybridStateOrTop hs : states) {
      if (hs.isHybridState()) {
        return false;
      }
    }
  }
  return true;
}

bool UnionTable::hasAlwaysZeroAmplitude(std::vector<unsigned int> qubits,
                                        unsigned int value,
                                        std::vector<unsigned int> bits,
                                        std::vector<bool> bitValues) {
  for (const auto& [qubitIndizes, bitIndizes] : indizesInSameState) {
    std::vector<unsigned int> qubitsInThisState = {};
    std::vector<unsigned int> bitsInThisState = {};
    unsigned int localValueForQubitsInThisState = 0;
    std::vector<bool> valuesForBitsInThisState = {};
    unsigned int includedQubitIndex;
    unsigned int includedBitIndex;
    // Retrieve local qubit values
    for (unsigned int i = 0; i < qubits.size(); ++i) {
      if (qubitIndizes.contains(qubits.at(i))) {
        unsigned int localQubitIndex =
            mappingGlobalToLocalQubitIndices.at(qubits.at(i));
        qubitsInThisState.push_back(localQubitIndex);
        includedQubitIndex = qubits.at(i);
        unsigned int mask = static_cast<unsigned int>(pow(2, i) + 0.1);
        if ((value & mask) == mask) {
          localValueForQubitsInThisState +=
              static_cast<unsigned int>(pow(2, localQubitIndex) + 0.1);
        }
      }
    }
    // Retrieve global bit values
    for (unsigned int i = 0; i < bits.size(); ++i) {
      if (bitIndizes.contains(bits.at(i))) {
        unsigned int localBitIndex =
            mappingGlobalToLocalBitIndices.at(bits.at(i));
        bitsInThisState.push_back(localBitIndex);
        includedBitIndex = bits.at(i);
        valuesForBitsInThisState.push_back(bitValues.at(includedBitIndex));
      }
    }
    // Call hasAlwaysNonZeroAmplitude with the local values on the respective
    // states
    if (!qubitsInThisState.empty() || !bitsInThisState.empty()) {
      std::vector<HybridStateOrTop> relevantStates;
      if (!qubitsInThisState.empty()) {
        relevantStates = *hRegOfQubits.at(includedQubitIndex);
      } else {
        relevantStates = *hRegOfBits.at(includedBitIndex);
      }
      bool noNonzeroAmplitude = true;
      for (HybridStateOrTop hs : relevantStates) {
        noNonzeroAmplitude &= hs.hasAlwaysZeroAmplitude(
            qubitsInThisState, localValueForQubitsInThisState, bitsInThisState,
            valuesForBitsInThisState);
      }
      if (noNonzeroAmplitude) {
        return true;
      }
    }
  }
  return false;
}

std::optional<bool> UnionTable::getIsBitEquivalentToQubit(unsigned int bit,
                                                          unsigned int qubit) {
  // Check if qubit and bit are in the same state
  bool areTargetsInSameState = false;
  for (const auto& [qubits, bits] : indizesInSameState) {
    areTargetsInSameState |= qubits.contains(qubit) && bits.contains(bit);
  }

  // If the targets are not in same state, both need to be always zero or both
  // always one.
  if (!areTargetsInSameState) {
    bool qubitAlwaysZero = isQubitAlwaysZero(qubit);
    bool qubitAlwaysOne = isQubitAlwaysOne(qubit);
    if ((qubitAlwaysZero && isBitAlwaysZero(bit)) ||
        (qubitAlwaysOne && isBitAlwaysOne(bit))) {
      return true;
    }
    if ((qubitAlwaysZero && isBitAlwaysOne(bit)) ||
        (qubitAlwaysOne && isBitAlwaysZero(bit))) {
      return false;
    }
    return std::optional<bool>();
  }

  // If targets are in the same state
  bool targetsEquivalent = true;
  bool targetsNegEquivalent = true;
  for (HybridStateOrTop hs : *hRegOfQubits.at(qubit)) {
    if (hs.isTop()) {
      return std::optional<bool>();
    }
    std::optional<bool> res = hs.getHybridState()->getIsBitEquivalentToQubit(
        mappingGlobalToLocalBitIndices.at(bit),
        mappingGlobalToLocalQubitIndices.at(qubit));
    if (!res.has_value()) {
      return std::optional<bool>();
    }
    targetsEquivalent &= res.value();
    targetsNegEquivalent &= !res.value();
  }
  if (targetsEquivalent) {
    return true;
  }
  if (targetsNegEquivalent) {
    return false;
  }
  return std::optional<bool>();
}
} // namespace mqt::ir::opt::qcp