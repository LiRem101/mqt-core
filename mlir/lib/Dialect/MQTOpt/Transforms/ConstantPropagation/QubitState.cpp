/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_CORE_QUBITSTATE
#define MQT_CORE_QUBITSTATE

#include "mlir/Dialect/MQTOpt/Transforms/ConstantPropagation/QubitState.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <map>
#include <memory>
#include <unordered_map>
#include <variant>

namespace mqt::ir::opt::qcp {
QubitState::QubitState(std::size_t nQubits) {
  this->nQubits = nQubits;
  this->map = std::unordered_map<unsigned int, std::complex<double>>();
  this->map.insert({0, std::complex<double>(1.0, 0.0)});
}

QubitState::~QubitState() = default;

[[nodiscard("QubitState::getSize called but ignored")]] std::size_t
QubitState::getSize() const {
  return this->map.size();
}

[[nodiscard("QubitState::getNQubits called but ignored")]] size_t
QubitState::getNQubits() const {
  return this->nQubits;
}

void QubitState::print(std::ostream& os) const { os << this->toString(); }

std::string QubitState::toString() const {
  std::string str;
  for (std::map ordered = std::map<unsigned int, std::complex<double>>(
           this->map.begin(), this->map.end());
       auto const& [key, val] : ordered) {
    std::string cn = std::to_string(val.real());
    if (val.imag() > 1e-4) {
      cn += "+ i" + std::to_string(val.imag());
    } else if (val.imag() < -1e-4) {
      cn += "- i" + std::to_string(-val.imag());
    }
    str += "|" + qubitStringToBinary(key) + "> -> " + cn + ", ";
  }

  return str;
}

QubitState QubitState::unify(const QubitState that) {
  throw std::logic_error("Not implemented");
}

void QubitState::propagateGate(std::string gate, unsigned int targets[],
                               unsigned int posCtrls[],
                               unsigned int negCtrls[]) {
  throw std::logic_error("Not implemented");
}

std::map<MeasurementResult, QubitState>
QubitState::measureQubit(unsigned int target) {
  throw std::logic_error("Not implemented");
}

void QubitState::removeQubit(unsigned int target) {
  throw std::logic_error("Not implemented");
}

bool QubitState::operator==(const QubitState& that) const {
  if (this->getSize() != that.getSize())
    return false;

  return std::ranges::all_of(
      this->map, [&](const std::pair<unsigned int, std::complex<double>>& p) {
        auto [key, val] = p;
        return (that.map.contains(key)) && (val == that.map.at(key));
      });
};

QubitStateOrTop::QubitStateOrTop() : variant(TOP::T) {}

QubitStateOrTop::QubitStateOrTop(TOP top) : variant(top) {}

QubitStateOrTop::QubitStateOrTop(std::shared_ptr<QubitState> qubitState)
    : variant(qubitState) {}

QubitStateOrTop::QubitStateOrTop(const QubitStateOrTop& qubitStateOrTop) =
    default;

QubitStateOrTop&
QubitStateOrTop::operator=(const QubitStateOrTop& qubitStateOrTop) = default;

QubitStateOrTop&
QubitStateOrTop::operator=(const std::shared_ptr<QubitState>& qubitState) {
  this->variant = qubitState;
  return *this;
}

QubitStateOrTop& QubitStateOrTop::operator=(const TOP& t) {
  this->variant = t;
  return *this;
}

bool QubitStateOrTop::operator==(const QubitStateOrTop& that) const {
  if (this->isTop() && that.isTop()) {
    return true;
  } else if (this->isTop() || that.isTop()) {
    return false;
  } else {
    return *this->getQubitState() == *that.getQubitState();
  }
}

bool QubitStateOrTop::operator!=(const QubitStateOrTop& that) const {
  return !(that == *this);
}

QubitStateOrTop::~QubitStateOrTop() = default;

[[nodiscard("QubitStateOrTop::isTop called but ignored")]] bool
QubitStateOrTop::isTop() const {
  return std::holds_alternative<TOP>(variant);
}

[[nodiscard("QubitStateOrTop::isQubitState called but ignored")]] bool
QubitStateOrTop::isQubitState() const {
  return std::holds_alternative<std::shared_ptr<QubitState>>(variant);
}

[[nodiscard("QubitStateOrTop::getQubitState called but ignored")]] std::
    shared_ptr<QubitState>
    QubitStateOrTop::getQubitState() const {
  return std::get<std::shared_ptr<QubitState>>(variant);
}

[[nodiscard("QubitStateOrTop::toString called but ignored")]] std::string
QubitStateOrTop::toString() const {
  if (isTop()) {
    return "TOP";
  } else {
    return getQubitState()->toString();
  }
}

void QubitStateOrTop::print(std::ostream& os) const { os << this->toString(); }
} // namespace mqt::ir::opt::qcp

#endif // MQT_CORE_QUBITSTATE