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
#include <variant>

namespace mqt::ir::opt::qcp {
HybridState::HybridState(std::size_t nQubits, std::size_t maxNonzeroAmplitudes,
                         unsigned int maxNumberOfBitValues,
                         std::vector<bool> bitValues, double probability)
    : qState(std::shared_ptr<QubitState>{
          new QubitState(nQubits, maxNonzeroAmplitudes)}),
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
  for (int i = bitValues.size() - 1; i >= 0; i--) {
    str += bitValues.at(i) ? "1" : "0";
  }
  if (size(bitValues) > 0) {
    str += ", ";
  }
  str += "p = " + std::format("{:.2f}", this->probability) + ";";
  return str;
}

void HybridState::propagateGate(qc::OpType gate,
                                std::vector<unsigned int> targets,
                                std::vector<unsigned int> posCtrlsQuantum,
                                std::vector<unsigned int> negCtrlsQuantum,
                                std::vector<unsigned int> posCtrlsClassical,
                                std::vector<unsigned int> negCtrlsClassical,
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
    qS->propagateGate(gate, targets, posCtrlsQuantum, negCtrlsQuantum, params);
  } catch (std::domain_error) {
    qState.getQubitState().reset();
    qState = QubitStateOrTop(T);
  }
}

std::vector<HybridState>
HybridState::propagateMeasurement(unsigned int quantumTarget,
                                  unsigned int classicalTarget) {
  if (qState)

    throw std::logic_error("Not implemented");
}

void HybridState::resetQubit(unsigned int target) {
  throw std::logic_error("Not implemented");
}

HybridState HybridState::unify(HybridState that,
                               std::vector<unsigned int> qubitsOccupiedByThat,
                               std::vector<unsigned int> bitsOccupiedByThat) {
  throw std::logic_error("Not implemented");
}

bool HybridState::operator==(const HybridState& that) const {
  if (this->probability != that.probability ||
      this->bitValues != that.bitValues)
    return false;

  return this->qState == that.qState;
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
  } else if (this->isTop() || that.isTop()) {
    return false;
  } else {
    return *this->getHybridState() == *that.getHybridState();
  }
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
  } else {
    return getHybridState()->toString();
  }
}

void HybridStateOrTop::print(std::ostream& os) const { os << this->toString(); }
} // namespace mqt::ir::opt::qcp

#endif // MQT_CORE_HYBRIDSTATE