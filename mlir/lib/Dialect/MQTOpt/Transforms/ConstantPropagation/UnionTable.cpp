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

#include <algorithm>
#include <ranges>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mqt::ir::opt::qcp {
std::set<std::pair<std::set<unsigned int>, std::set<unsigned int>>>
UnionTable::getInvolvedStates(const std::set<unsigned int>& qubits,
                              const std::set<unsigned int>& bits) const {
  std::set<std::pair<std::set<unsigned int>, std::set<unsigned int>>>
      involvedStates;
  for (std::pair<std::set<unsigned int>, std::set<unsigned int>> const&
           setsInSameState : indicesInSameState) {
    std::set<unsigned int> const qubitIndices = setsInSameState.first;
    if (std::set<unsigned int> const bitIndices = setsInSameState.second;
        std::ranges::find_first_of(qubitIndices, qubits) !=
            qubitIndices.end() ||
        std::ranges::find_first_of(bitIndices, bits) != bitIndices.end()) {
      involvedStates.insert(setsInSameState);
    }
  }
  return involvedStates;
}

bool UnionTable::checkIfOnlyOneSetIsNotZero(
    std::vector<unsigned int> qubits,
    const std::set<std::vector<unsigned int>>& values,
    const std::set<std::pair<std::set<unsigned int>, std::set<unsigned int>>>&
        involvedIndices) {
  std::set<unsigned int> const qubitIndices = involvedIndices.begin()->first;
  // Get index of the given qubits in their vector to match with values
  std::vector<unsigned int> qubitIndicesInGivenVector;
  for (unsigned int q : qubits) {
    auto it = std::ranges::find(qubitIndices, q);
    if (it != qubitIndices.end()) {
      const std::size_t index = std::distance(qubitIndices.begin(), it);
      qubitIndicesInGivenVector.push_back(index);
    }
  }
  // Check if the hybrid states contain only one set
  for (HybridStateOrTop hs : *hRegOfQubits.at(*qubitIndices.begin())) {
    if (hs.isTop()) {
      return false;
    }
    std::set<std::vector<unsigned int>> valuesThatAreNonZero = {};
    std::vector<unsigned int> nonzeroValuesOfCurrentValues = {};
    for (std::vector currentValues : values) {
      for (unsigned int currentValue : currentValues) {
        // map current values to the current qubitIndices
        unsigned int localValue = 0;
        std::vector<unsigned int> localQubitVector = {};
        for (unsigned int i = 0; i < qubitIndicesInGivenVector.size(); ++i) {
          auto it = qubitIndices.begin();
          std::advance(it, qubitIndicesInGivenVector.at(i));
          localQubitVector.push_back(mappingGlobalToLocalQubitIndices.at(*it));

          const unsigned int mask =
              static_cast<unsigned int>(pow(2, *it) + 0.1);
          if ((mask & currentValue) != 0) {
            localValue += static_cast<unsigned int>(
                pow(2, mappingGlobalToLocalQubitIndices.at(*it)) + 0.1);
          }
        }
        if (!hs.hasAlwaysZeroAmplitude(localQubitVector, localValue)) {
          nonzeroValuesOfCurrentValues.push_back(currentValue);
        }
      }
      if (!nonzeroValuesOfCurrentValues.empty()) {
        valuesThatAreNonZero.insert(nonzeroValuesOfCurrentValues);
        nonzeroValuesOfCurrentValues.clear();
      }
    }
    if (valuesThatAreNonZero.size() > 1) {
      if (involvedIndices.size() == 1) {
        return false;
      }
      // call the other qubits on the hybridState with the leftover sets
      const auto it = std::next(involvedIndices.begin());
      if (std::set remainingIndices(it, involvedIndices.end());
          !checkIfOnlyOneSetIsNotZero(qubits, valuesThatAreNonZero,
                                      remainingIndices)) {
        return false;
      }
    }
  }
  return true;
}

std::pair<std::set<unsigned int>, std::set<unsigned int>>
UnionTable::unifyHybridStates(
    std::pair<std::set<unsigned int>, std::set<unsigned int>> involvedStates1,
    std::pair<std::set<unsigned int>, std::set<unsigned int>> involvedStates2) {
  std::vector<HybridStateOrTop> const firstStates =
      !involvedStates1.first.empty()
          ? *hRegOfQubits.at(*involvedStates1.first.begin())
          : *hRegOfBits.at(*involvedStates1.second.begin());
  std::vector<HybridStateOrTop> const secondStates =
      !involvedStates2.first.empty()
          ? *hRegOfQubits.at(*involvedStates2.first.begin())
          : *hRegOfBits.at(*involvedStates2.second.begin());

  // Adapt global to local mapping and create new pairs for indices in same
  // state
  std::pair<std::set<unsigned int>, std::set<unsigned int>> newStatesInSameSet =
      {{}, {}};
  std::vector<unsigned int> qubitIndicesOfSecondState = {};
  unsigned int currentIndex = 0;
  auto it1 = involvedStates1.first.begin();
  auto it2 = involvedStates2.first.begin();
  while (it1 != involvedStates1.first.end() ||
         it2 != involvedStates2.first.end()) {
    if (it2 == involvedStates2.first.end() ||
        (it1 != involvedStates1.first.end() && *it1 < *it2)) {
      mappingGlobalToLocalQubitIndices.at(*it1) = currentIndex;
      newStatesInSameSet.first.insert(*it1);
      ++it1;
    } else {
      qubitIndicesOfSecondState.push_back(currentIndex);
      mappingGlobalToLocalQubitIndices.at(*it2) = currentIndex;
      newStatesInSameSet.first.insert(*it2);
      ++it2;
    }
    currentIndex++;
  }
  std::vector<unsigned int> bitIndicesOfSecondState = {};
  currentIndex = 0;
  it1 = involvedStates1.second.begin();
  it2 = involvedStates2.second.begin();
  while (it1 != involvedStates1.second.end() ||
         it2 != involvedStates2.second.end()) {
    if (it2 == involvedStates2.second.end() ||
        (it1 != involvedStates1.second.end() && *it1 < *it2)) {
      mappingGlobalToLocalBitIndices.at(*it1) = currentIndex;
      newStatesInSameSet.second.insert(*it1);
      ++it1;
    } else {
      bitIndicesOfSecondState.push_back(currentIndex);
      mappingGlobalToLocalBitIndices.at(*it2) = currentIndex;
      newStatesInSameSet.second.insert(*it2);
      ++it2;
    }
    currentIndex++;
  }

  indicesInSameState.erase(involvedStates1);
  indicesInSameState.erase(involvedStates2);
  indicesInSameState.insert(newStatesInSameSet);

  // Create new State set
  std::vector<HybridStateOrTop> newHybridStates;
  bool encounteredTop = false;
  for (HybridStateOrTop const& hs1 : firstStates) {
    for (HybridStateOrTop const& hs2 : secondStates) {
      if (hs1.isTop() || hs2.isTop()) {
        encounteredTop = true;
        break;
      }
      if (!encounteredTop) {
        try {
          HybridState newHybridState = hs1.getHybridState()->unify(
              *hs2.getHybridState(), qubitIndicesOfSecondState,
              bitIndicesOfSecondState);
          newHybridStates.push_back({HybridStateOrTop(
              std::make_shared<HybridState>(newHybridState))});
        } catch (std::domain_error const&) {
          encounteredTop = true;
        }
      }
      if (hs2.isHybridState()) {
        hs2.getHybridState().reset();
      }
    }
    if (hs1.isHybridState()) {
      hs1.getHybridState().reset();
    }
  }
  if (encounteredTop) {
    for (HybridStateOrTop hs : newHybridStates) {
      hs.getHybridState().reset();
    }
    newHybridStates.clear();
    newHybridStates.push_back({HybridStateOrTop(T)});
  }

  // Update pointers in hRegs
  for (unsigned int qubitIndices : newStatesInSameSet.first) {
    hRegOfQubits.at(qubitIndices) =
        std::make_shared<std::vector<HybridStateOrTop>>(newHybridStates);
  }
  for (unsigned int bitIndices : newStatesInSameSet.second) {
    hRegOfBits.at(bitIndices) =
        std::make_shared<std::vector<HybridStateOrTop>>(newHybridStates);
  }

  return newStatesInSameSet;
}

void UnionTable::applySwapGate(const unsigned int target1,
                               const unsigned int target2) {
  bool changed = false;
  std::pair<std::set<unsigned int>, std::set<unsigned int>> newStates1;
  std::pair<std::set<unsigned int>, std::set<unsigned int>> newStates2;
  std::set<std::pair<std::set<unsigned int>, std::set<unsigned int>>> toErase =
      {};
  for (auto it = indicesInSameState.begin(); it != indicesInSameState.end();
       ++it) {
    if (it->first.contains(target1) && !it->first.contains(target2)) {
      changed = true;
      for (auto qubits : it->first) {
        if (qubits == target1) {
          newStates1.first.insert(target2);
        } else {
          newStates1.first.insert(qubits);
        }
      }
      toErase.insert(*it);
    } else if (!it->first.contains(target1) && it->first.contains(target2)) {
      changed = true;
      for (auto qubits : it->first) {
        if (qubits == target2) {
          newStates2.first.insert(target1);
        } else {
          newStates2.first.insert(qubits);
        }
      }
      toErase.insert(*it);
    }
  }
  if (changed) {
    indicesInSameState.insert(newStates1);
    indicesInSameState.insert(newStates2);
    for (const auto& erasing : toErase) {
      indicesInSameState.erase(erasing);
    }
  }

  const unsigned int localTarget2 =
      mappingGlobalToLocalQubitIndices.at(target2);
  const std::shared_ptr<std::vector<HybridStateOrTop>> ptrOfTarget2 =
      hRegOfQubits.at(target2);

  mappingGlobalToLocalQubitIndices.at(target2) =
      mappingGlobalToLocalQubitIndices.at(target1);
  mappingGlobalToLocalQubitIndices.at(target1) = localTarget2;
  hRegOfQubits.at(target2) = hRegOfQubits.at(target1);
  hRegOfQubits.at(target1) = ptrOfTarget2;
}

UnionTable::UnionTable(const unsigned int maxNonzeroAmplitudes,
                       const unsigned int maxNumberOfBitValues)
    : maxNonzeroAmplitudes(maxNonzeroAmplitudes),
      maxNumberOfBitValues(maxNumberOfBitValues),
      mappingGlobalToLocalQubitIndices({}), mappingGlobalToLocalBitIndices({}),
      hRegOfQubits({}), hRegOfBits({}), indicesInSameState({}) {}

UnionTable::~UnionTable() = default;

std::string UnionTable::toString() const {
  std::string result;
  for (const auto& [qubitIndices, bitIndices] : indicesInSameState) {
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
    for (HybridStateOrTop const& hs : *hRegOfQubits.at(*qubitIndices.begin())) {
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

unsigned int UnionTable::getNumberOfBits() const {
  return mappingGlobalToLocalBitIndices.size();
}

void UnionTable::propagateGate(
    qc::OpType gate, std::vector<unsigned int> targets,
    std::vector<unsigned int> posCtrlsQuantum,
    std::vector<unsigned int> negCtrlsQuantum,
    const std::vector<unsigned int>& posCtrlsClassical,
    const std::vector<unsigned int>& negCtrlsClassical,
    std::vector<double> params) {
  if (gate == qc::Barrier) {
    return;
  }
  if (gate == qc::SWAP && posCtrlsQuantum.empty() && negCtrlsQuantum.empty() &&
      posCtrlsClassical.empty() && negCtrlsClassical.empty()) {
    applySwapGate(targets.at(0), targets.at(1));
    return;
  }

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
    if (const HybridStateOrTop hs = hRegOfQubits.at(targets.at(0))->at(i);
        hs.isHybridState()) {
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

void UnionTable::propagateMeasurement(
    unsigned int quantumTarget, unsigned int classicalTarget,
    const std::vector<unsigned int>& posCtrlsClassical,
    const std::vector<unsigned int>& negCtrlsClassical) {

  std::set involvedBits = {classicalTarget};
  involvedBits.insert(posCtrlsClassical.begin(), posCtrlsClassical.end());
  involvedBits.insert(negCtrlsClassical.begin(), negCtrlsClassical.end());
  std::set<std::pair<std::set<unsigned int>, std::set<unsigned int>>>
      involvedStates = getInvolvedStates({quantumTarget}, involvedBits);
  bool bitExists = false;
  for (const auto& bitsInState : involvedStates | std::views::values) {
    bitExists |= bitsInState.contains(classicalTarget);
  }

  std::pair<std::set<unsigned int>, std::set<unsigned int>>
      involvedStateIndices = *involvedStates.begin();
  if (involvedStates.size() > 1) {
    std::pair<std::set<unsigned int>, std::set<unsigned int>> currentStates =
        *involvedStates.begin();
    auto nextStateIt = ++involvedStates.begin();
    while (nextStateIt != involvedStates.end()) {
      currentStates = unifyHybridStates(currentStates, *nextStateIt);
      ++nextStateIt;
    }
    involvedStateIndices = currentStates;
  }
  if (!bitExists && classicalTarget != hRegOfBits.size()) {
    throw std::invalid_argument("New classical bit index needs to be equal to "
                                "number of existing bits.");
  }
  std::vector<HybridStateOrTop> involvedHybridStates =
      *hRegOfQubits.at(quantumTarget);
  bool top = false;
  if (!bitExists) {
    indicesInSameState.erase(involvedStateIndices);
    involvedStateIndices.second.insert(mappingGlobalToLocalBitIndices.size());
    indicesInSameState.insert(involvedStateIndices);
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

  // Get local vector for ctrl bits
  std::vector<unsigned int> localPosCtrl = {};
  for (const unsigned int posCtrlBit : posCtrlsClassical) {
    localPosCtrl.push_back(mappingGlobalToLocalBitIndices.at(posCtrlBit));
  }
  std::vector<unsigned int> localNegCtrl = {};
  for (const unsigned int posNegBit : negCtrlsClassical) {
    localNegCtrl.push_back(mappingGlobalToLocalBitIndices.at(posNegBit));
  }

  // Propagate measurement into the states in involvedHybridStates
  std::vector<HybridStateOrTop> newHybridStatesOrTops;
  for (HybridStateOrTop const& hs :
       *hRegOfQubits.at(*involvedStateIndices.first.begin())) {
    if (!top && hs.isHybridState()) {
      try {
        std::vector<HybridState> newHybridState =
            hs.getHybridState()->propagateMeasurement(
                mappingGlobalToLocalQubitIndices.at(quantumTarget),
                mappingGlobalToLocalBitIndices.at(classicalTarget),
                localPosCtrl, localNegCtrl);
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

  for (unsigned int const qubitIndices : involvedStateIndices.first) {
    hRegOfQubits.at(qubitIndices) =
        std::make_shared<std::vector<HybridStateOrTop>>(newHybridStatesOrTops);
  }
  for (unsigned int const bitIndices : involvedStateIndices.second) {
    hRegOfBits.at(bitIndices) =
        std::make_shared<std::vector<HybridStateOrTop>>(newHybridStatesOrTops);
  }
}

void UnionTable::propagateReset(
    unsigned int target, const std::vector<unsigned int>& posCtrlsClassical,
    const std::vector<unsigned int>& negCtrlsClassical) {
  std::set<unsigned int> involvedBits = {};
  involvedBits.insert(posCtrlsClassical.begin(), posCtrlsClassical.end());
  involvedBits.insert(negCtrlsClassical.begin(), negCtrlsClassical.end());
  std::set<std::pair<std::set<unsigned int>, std::set<unsigned int>>>
      involvedStates = getInvolvedStates({target}, involvedBits);

  std::pair<std::set<unsigned int>, std::set<unsigned int>>
      involvedStateIndices = *involvedStates.begin();
  if (involvedStates.size() > 1) {
    std::pair<std::set<unsigned int>, std::set<unsigned int>> currentStates =
        *involvedStates.begin();
    auto nextStateIt = ++involvedStates.begin();
    while (nextStateIt != involvedStates.end()) {
      currentStates = unifyHybridStates(currentStates, *nextStateIt);
      ++nextStateIt;
    }
    involvedStateIndices = currentStates;
  }

  auto [involvedQubitIndices, involvedBitIndices] = involvedStateIndices;

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

  for (unsigned int const qubitIndices : involvedQubitIndices) {
    hRegOfQubits.at(qubitIndices) =
        std::make_shared<std::vector<HybridStateOrTop>>(newHybridStatesOrTops);
  }
  for (unsigned int const bitIndices : involvedBitIndices) {
    hRegOfBits.at(bitIndices) =
        std::make_shared<std::vector<HybridStateOrTop>>(newHybridStatesOrTops);
  }
}

unsigned int UnionTable::propagateQubitAlloc() {
  unsigned int qubitIndex = mappingGlobalToLocalQubitIndices.size();
  mappingGlobalToLocalQubitIndices.push_back(0);
  auto hs = HybridState(1, maxNonzeroAmplitudes, maxNumberOfBitValues, {}, 1.0);
  std::vector setForNewQubit = {
      HybridStateOrTop(std::make_shared<HybridState>(hs))};
  hRegOfQubits.push_back(
      std::make_shared<std::vector<HybridStateOrTop>>(setForNewQubit));
  indicesInSameState.insert({{qubitIndex}, {}});
  return qubitIndex;
}

unsigned int UnionTable::propagateBitDef(bool value) {
  unsigned int bitIndex = mappingGlobalToLocalBitIndices.size();
  mappingGlobalToLocalBitIndices.push_back(0);
  auto hs =
      HybridState(0, maxNonzeroAmplitudes, maxNumberOfBitValues, {value}, 1.0);
  std::vector setForNewBit = {
      HybridStateOrTop(std::make_shared<HybridState>(hs))};
  hRegOfBits.push_back(
      std::make_shared<std::vector<HybridStateOrTop>>(setForNewBit));
  indicesInSameState.insert({{}, {bitIndex}});
  return bitIndex;
}

bool UnionTable::isQubitAlwaysOne(const size_t q) const {
  for (HybridStateOrTop hs : *hRegOfQubits.at(q)) {
    if (!hs.isQubitAlwaysOne(mappingGlobalToLocalQubitIndices.at(q))) {
      return false;
    }
  }
  return true;
}

bool UnionTable::isQubitAlwaysZero(const size_t q) const {
  for (HybridStateOrTop hs : *hRegOfQubits.at(q)) {
    if (!hs.isQubitAlwaysZero(mappingGlobalToLocalQubitIndices.at(q))) {
      return false;
    }
  }
  return true;
}

bool UnionTable::isBitAlwaysOne(const size_t q) const {
  for (HybridStateOrTop hs : *hRegOfBits.at(q)) {
    if (!hs.isBitAlwaysOne(mappingGlobalToLocalBitIndices.at(q))) {
      return false;
    }
  }
  return true;
}

bool UnionTable::isBitAlwaysZero(const size_t q) const {
  for (HybridStateOrTop hs : *hRegOfBits.at(q)) {
    if (!hs.isBitAlwaysZero(mappingGlobalToLocalBitIndices.at(q))) {
      return false;
    }
  }
  return true;
}

bool UnionTable::allTop() {
  for (const auto& [qubitIndices, bitIndices] : indicesInSameState) {
    std::vector<HybridStateOrTop> states;
    if (!qubitIndices.empty()) {
      states = *hRegOfQubits.at(*qubitIndices.begin());
    } else {
      states = *hRegOfBits.at(*bitIndices.begin());
    }
    for (HybridStateOrTop hs : states) {
      if (hs.isHybridState()) {
        return false;
      }
    }
  }
  return true;
}

bool UnionTable::hasAlwaysZeroAmplitude(const std::vector<unsigned int>& qubits,
                                        const unsigned int value,
                                        const std::vector<unsigned int>& bits,
                                        std::vector<bool> bitValues) {
  for (const auto& [qubitIndices, bitIndices] : indicesInSameState) {
    std::vector<unsigned int> qubitsInThisState = {};
    std::vector<unsigned int> bitsInThisState = {};
    unsigned int localValueForQubitsInThisState = 0;
    std::vector<bool> valuesForBitsInThisState = {};
    unsigned int includedQubitIndex = 0;
    unsigned int includedBitIndex = 0;
    // Retrieve local qubit values
    int localPowerIndex = 0;
    for (unsigned int i = 0; i < qubits.size(); ++i) {
      if (qubitIndices.contains(qubits.at(i))) {
        unsigned int localQubitIndex =
            mappingGlobalToLocalQubitIndices.at(qubits.at(i));
        qubitsInThisState.push_back(localQubitIndex);
        includedQubitIndex = qubits.at(i);
        if (const unsigned int mask =
                static_cast<unsigned int>(pow(2, i) + 0.1);
            (value & mask) == mask) {
          localValueForQubitsInThisState +=
              static_cast<unsigned int>(pow(2, localPowerIndex) + 0.1);
        }
        ++localPowerIndex;
      }
    }
    // Retrieve global bit values
    for (unsigned int i = 0; i < bits.size(); ++i) {
      if (bitIndices.contains(bits.at(i))) {
        unsigned int localBitIndex =
            mappingGlobalToLocalBitIndices.at(bits.at(i));
        bitsInThisState.push_back(localBitIndex);
        includedBitIndex = bits.at(i);
        valuesForBitsInThisState.push_back(bitValues.at(i));
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
      bool zeroAmplitude = true;
      for (HybridStateOrTop hs : relevantStates) {
        zeroAmplitude &= hs.hasAlwaysZeroAmplitude(
            qubitsInThisState, localValueForQubitsInThisState, bitsInThisState,
            valuesForBitsInThisState);
      }
      if (zeroAmplitude) {
        return true;
      }
    }
  }
  return false;
}

std::optional<bool>
UnionTable::getIsBitEquivalentToQubit(const unsigned int bit,
                                      const unsigned int qubit) {
  // Check if qubit and bit are in the same state
  bool areTargetsInSameState = false;
  for (const auto& [qubits, bits] : indicesInSameState) {
    areTargetsInSameState |= qubits.contains(qubit) && bits.contains(bit);
  }

  // If the targets are not in same state, both need to be always zero or both
  // always one.
  if (!areTargetsInSameState) {
    const bool qubitAlwaysZero = isQubitAlwaysZero(qubit);
    const bool qubitAlwaysOne = isQubitAlwaysOne(qubit);
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

bool UnionTable::isOnlyOneSetNotZero(
    std::vector<unsigned int> qubits,
    const std::set<std::vector<unsigned int>>& values) {
  const std::set qubitsAsSet(qubits.begin(), qubits.end());
  const std::set<std::pair<std::set<unsigned int>, std::set<unsigned int>>>
      involvedIndices = getInvolvedStates(qubitsAsSet, {});

  return checkIfOnlyOneSetIsNotZero(qubits, values, involvedIndices);
}
} // namespace mqt::ir::opt::qcp