/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_CORE_HYBRIDSTATE
#define MQT_CORE_HYBRIDSTATE

#include "mlir/Dialect/MQTOpt/Transforms/ConstantPropagation/HybridState.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <format>
#include <llvm/ADT/STLExtras.h>
#include <memory>
#include <utility>
#include <variant>

namespace mqt::ir::opt::qcp {
HybridState::HybridState(std::size_t nQubits, std::size_t maxNonzeroAmplitudes,
                         unsigned int maxNumberOfBitValues,
                         std::vector<bool> bitValues, double probability)
    : qState(std::make_shared<QubitState>(nQubits, maxNonzeroAmplitudes)),
      probability(probability), bitValues(std::move(bitValues)),
      maxNumberOfBitValues(maxNumberOfBitValues) {}

HybridState::~HybridState() {
  if (qState.isQubitState()) {
    auto qS = qState.getQubitState();
    qS.reset();
  }
};

void HybridState::print(std::ostream& os) const { os << this->toString(); }

std::string HybridState::toString() const {
  std::string str = "{" + this->qState.toString() + "}: ";
  for (int i = static_cast<int>(bitValues.size()) - 1; i >= 0; i--) {
    str += bitValues.at(i) ? "1" : "0";
  }
  if (size(bitValues) > 0) {
    str += ", ";
  }
  str += "p = " + std::format("{:.2f}", this->probability) + ";";
  return str;
}

void HybridState::propagateGate(
    qc::OpType gate, std::vector<unsigned int> targets,
    std::vector<unsigned int> posCtrlsQuantum,
    std::vector<unsigned int> negCtrlsQuantum,
    const std::vector<unsigned int>& posCtrlsClassical,
    const std::vector<unsigned int>& negCtrlsClassical,
    std::vector<double> params) {
  if (qState.isTop()) {
    return;
  }

  for (const unsigned int posCtrl : posCtrlsClassical) {
    if (!bitValues.at(posCtrl)) {
      return;
    }
  }
  for (const unsigned int negCtrl : negCtrlsClassical) {
    if (bitValues.at(negCtrl)) {
      return;
    }
  }

  auto qS = qState.getQubitState();
  try {
    qS->propagateGate(gate, std::move(targets), std::move(posCtrlsQuantum),
                      std::move(negCtrlsQuantum), std::move(params));
  } catch (std::domain_error const&) {
    qState.getQubitState().reset();
    qState = QubitStateOrTop(T);
  }
}
unsigned int HybridState::addClassicalBit(bool value) {
  if (bitValues.size() >= maxNumberOfBitValues) {
    throw std::domain_error("Number of bits would exceed number of allowed "
                            "bits. HybridState needs to be treated as TOP.");
  }
  bitValues.push_back(value);
  return bitValues.size() - 1;
}

std::vector<HybridState>
HybridState::propagateMeasurement(unsigned int quantumTarget,
                                  unsigned int classicalTarget) {
  if (qState.isTop()) {
    throw std::domain_error("Measured QuantumState is TOP. HybridState needs "
                            "to be treated as TOP.");
  }

  if (classicalTarget >= bitValues.size()) {
    throw std::invalid_argument(
        "Bit to save measurement result in does not exist.");
  }

  std::map<unsigned int, std::pair<double, std::shared_ptr<QubitState>>> const
      measurementResults = qState.getQubitState()->measureQubit(quantumTarget);

  std::vector<HybridState> results;
  for (const auto& [resultBit, value] : measurementResults) {
    const double newProbability = value.first * probability;
    auto newQS = QubitStateOrTop(value.second);
    auto newHybrid = HybridState();

    auto newBitValues = std::vector<bool>(bitValues.size());
    for (unsigned int i = 0; i < size(bitValues); i++) {
      if (i == classicalTarget) {
        newBitValues[i] = resultBit == 1;
      } else {
        newBitValues[i] = bitValues.at(i);
      }
    }
    newHybrid.probability = newProbability;
    newHybrid.qState = newQS;
    newHybrid.bitValues = newBitValues;
    newHybrid.maxNumberOfBitValues = maxNumberOfBitValues;

    results.push_back(newHybrid);
  }

  return results;
}

std::vector<HybridState> HybridState::propagateReset(unsigned int target) {
  if (qState.isTop()) {
    throw std::domain_error("Measured QuantumState is TOP. HybridState needs "
                            "to be treated as TOP.");
  }

  std::set<std::pair<double, std::shared_ptr<QubitState>>> const resetResults =
      qState.getQubitState()->resetQubit(target);

  std::vector<HybridState> results;
  for (const auto& [prob, value] : resetResults) {
    const double newProbability = prob * probability;
    auto newQS = QubitStateOrTop(value);
    auto newHybrid = HybridState();

    auto newBitValues = std::vector<bool>(bitValues.size());
    for (unsigned int i = 0; i < size(bitValues); i++) {
      newBitValues[i] = bitValues.at(i);
    }
    newHybrid.probability = newProbability;
    newHybrid.qState = newQS;
    newHybrid.bitValues = newBitValues;
    newHybrid.maxNumberOfBitValues = maxNumberOfBitValues;

    results.push_back(newHybrid);
  }

  return results;
}

HybridState HybridState::unify(HybridState that,
                               std::vector<unsigned int> qubitsOccupiedByThat,
                               std::vector<unsigned int> bitsOccupiedByThat) {
  if (this->bitValues.size() + that.bitValues.size() > maxNumberOfBitValues) {
    throw std::domain_error("Too many bit values to track. HybridState needs "
                            "to be treated as TOP.");
  }

  QubitStateOrTop newQState;
  if (this->qState.isTop() || that.qState.isTop()) {
    newQState = QubitStateOrTop(T);
  } else {
    try {
      QubitState const unifiedQS = this->qState.getQubitState()->unify(
          *(that.qState.getQubitState().get()),
          std::move(qubitsOccupiedByThat));
      newQState = QubitStateOrTop(std::make_shared<QubitState>(unifiedQS));
    } catch (std::domain_error const&) {
      newQState = QubitStateOrTop(T);
    }
  }

  std::vector<bool> newBitValues;
  unsigned int const numberOfNewbits = bitValues.size() + that.bitValues.size();
  unsigned int thisCounter = 0;
  unsigned int thatCounter = 0;
  for (unsigned int i = 0; i < numberOfNewbits; i++) {
    if (std::ranges::find(bitsOccupiedByThat, i) == bitsOccupiedByThat.end()) {
      newBitValues.push_back(bitValues[thisCounter]);
      thisCounter++;
    } else {
      newBitValues.push_back(that.bitValues[thatCounter]);
      thatCounter++;
    }
  }

  HybridState result = HybridState();
  result.bitValues = newBitValues;
  result.qState = newQState;
  result.probability = probability * that.probability;
  result.maxNumberOfBitValues = this->maxNumberOfBitValues;

  return result;
}

bool HybridState::operator==(const HybridState& that) const {
  if (this->probability != that.probability ||
      this->bitValues != that.bitValues) {
    return false;
  }

  return this->qState == that.qState;
}

bool HybridState::isQubitAlwaysOne(size_t q) const {
  return qState.isQubitAlwaysOne(q);
}

bool HybridState::isQubitAlwaysZero(size_t q) const {
  return qState.isQubitAlwaysZero(q);
}

bool HybridState::isBitAlwaysOne(size_t q) const { return bitValues.at(q); }

bool HybridState::isBitAlwaysZero(size_t q) const { return !bitValues.at(q); }

bool HybridState::hasAlwaysZeroAmplitude(
    std::vector<unsigned int> qubits, unsigned int value,
    std::vector<unsigned int> bits, std::vector<bool> bitValuesToCheck) const {
  bool amplitudesZero = false;
  if (!qubits.empty()) {
    amplitudesZero = qState.hasAlwaysZeroAmplitude(qubits, value);
  }
  bool bitValuesExist = true;
  for (unsigned int i = 0; i < bits.size(); i++) {
    bitValuesExist &= bitValuesToCheck.at(i) == this->bitValues.at(i);
  }
  return amplitudesZero || !bitValuesExist;
}

std::optional<bool>
HybridState::getIsBitEquivalentToQubit(const unsigned int bit,
                                       const unsigned int qubit) {
  if ((bitValues.at(bit) && qState.isQubitAlwaysZero(qubit)) ||
      (!bitValues.at(bit) && qState.isQubitAlwaysOne(qubit))) {
    return false;
  }
  if ((bitValues.at(bit) && qState.isQubitAlwaysOne(qubit)) ||
      (!bitValues.at(bit) && qState.isQubitAlwaysZero(qubit))) {
    return true;
  }
  return std::optional<bool>();
}

HybridStateOrTop::HybridStateOrTop() : variant(TOP::T) {}

HybridStateOrTop::HybridStateOrTop(TOP top) : variant(top) {}

HybridStateOrTop::HybridStateOrTop(std::shared_ptr<HybridState> hybridState)
    : variant(hybridState) {}

HybridStateOrTop::HybridStateOrTop(const HybridStateOrTop& hybridStateOrTop) =
    default;

HybridStateOrTop&
HybridStateOrTop::operator=(const HybridStateOrTop& hybridStateOrTop) = default;

HybridStateOrTop&
HybridStateOrTop::operator=(const std::shared_ptr<HybridState>& hybridState) {
  this->variant = hybridState;
  return *this;
}

HybridStateOrTop& HybridStateOrTop::operator=(const TOP& t) {
  this->variant = t;
  return *this;
}

bool HybridStateOrTop::operator==(const HybridStateOrTop& that) const {
  if (this->isTop() && that.isTop()) {
    return true;
  }
  if (this->isTop() || that.isTop()) {
    return false;
  }
  return *this->getHybridState() == *that.getHybridState();
}

bool HybridStateOrTop::operator!=(const HybridStateOrTop& that) const {
  return !(that == *this);
}

HybridStateOrTop::~HybridStateOrTop() = default;

[[nodiscard("HybridStateOrTop::isTop called but ignored")]] bool
HybridStateOrTop::isTop() const {
  return std::holds_alternative<TOP>(variant);
}

[[nodiscard("HybridStateOrTop::isHybridState called but ignored")]] bool
HybridStateOrTop::isHybridState() const {
  return std::holds_alternative<std::shared_ptr<HybridState>>(variant);
}

[[nodiscard("HybridStateOrTop::getHybridState called but ignored")]] std::
    shared_ptr<HybridState>
    HybridStateOrTop::getHybridState() const {
  return std::get<std::shared_ptr<HybridState>>(variant);
}

[[nodiscard("HybridStateOrTop::toString called but ignored")]] std::string
HybridStateOrTop::toString() const {
  if (isTop()) {
    return "TOP";
  }
  return getHybridState()->toString();
}

void HybridStateOrTop::print(std::ostream& os) const { os << this->toString(); }

bool HybridStateOrTop::isQubitAlwaysOne(size_t q) const {
  if (isTop()) {
    return false;
  }
  return getHybridState()->isQubitAlwaysOne(q);
}

bool HybridStateOrTop::isQubitAlwaysZero(size_t q) const {
  if (isTop()) {
    return false;
  }
  return getHybridState()->isQubitAlwaysZero(q);
}

bool HybridStateOrTop::isBitAlwaysOne(size_t q) const {
  if (isTop()) {
    return false;
  }
  return getHybridState()->isBitAlwaysOne(q);
}

bool HybridStateOrTop::isBitAlwaysZero(size_t q) const {
  if (isTop()) {
    return false;
  }
  return getHybridState()->isBitAlwaysZero(q);
}

bool HybridStateOrTop::hasAlwaysZeroAmplitude(
    std::vector<unsigned int> qubits, unsigned int value,
    std::vector<unsigned int> bits, std::vector<bool> bitValues) const {
  if (isTop()) {
    return false;
  }
  return getHybridState()->hasAlwaysZeroAmplitude(qubits, value, bits,
                                                  bitValues);
}

} // namespace mqt::ir::opt::qcp

#endif // MQT_CORE_HYBRIDSTATE