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

#include "mlir/Dialect/MQTOpt/Transforms/ConstantPropagation/GateToMap.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <format>
#include <map>
#include <memory>
#include <unordered_map>
#include <variant>

namespace mqt::ir::opt::qcp {
QubitState::QubitState(std::size_t nQubits, std::size_t maxNonzeroAmplitudes)
    : nQubits(nQubits), maxNonzeroAmplitudes(maxNonzeroAmplitudes) {
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
  bool first = true;
  for (std::map ordered = std::map<unsigned int, std::complex<double>>(
           this->map.begin(), this->map.end());
       auto const& [key, val] : ordered) {
    if (!first) {
      str += ", ";
    }
    first = false;
    std::string cn = std::format("{:.2f}", val.real());
    if (val.imag() > 1e-4) {
      cn += " + i" + std::format("{:.2f}", val.imag());
    } else if (val.imag() < -1e-4) {
      cn += " - i" + std::format("{:.2f}", -val.imag());
    }
    str += "|" + qubitStringToBinary(key) + "> -> " + cn;
  }

  return str;
}

QubitState QubitState::unify(QubitState that,
                             std::vector<unsigned int> qubitsOccupiedByThat) {

  // Check if future state would be too large
  if (map.size() * that.map.size() > maxNonzeroAmplitudes) {
    throw std::domain_error("Number of nonzero amplitudes too high. State "
                            "needs to be treated as TOP.");
  }

  std::unordered_map<unsigned int, std::complex<double>> newValues;
  for (const auto& [keyThis, valThis] : map) {
    for (const auto& [keyThat, valThat] : that.map) {

      unsigned int loopVarThis = nQubits - 1;
      unsigned int loopVarThat = that.nQubits - 1;
      unsigned int newKey = 0;

      for (int i = nQubits + that.nQubits - 1; i >= 0; i--) {
        bool inThat = std::ranges::find(qubitsOccupiedByThat, i) !=
                      qubitsOccupiedByThat.end();
        bool isOne = false;
        if (inThat) {
          isOne = (keyThat &
                   static_cast<unsigned int>(pow(2, loopVarThat) + 0.1)) != 0;
          loopVarThat--;
        } else {
          isOne = (keyThis &
                   static_cast<unsigned int>(pow(2, loopVarThis) + 0.1)) != 0;
          loopVarThis--;
        }
        if (isOne) {
          newKey += static_cast<unsigned int>(pow(2, i) + 0.1);
        }
      }

      newValues[newKey] = valThis * valThat;
    }
  }
  QubitState newState =
      QubitState(nQubits + that.nQubits, maxNonzeroAmplitudes);
  newState.map = newValues;

  return newState;
}

void QubitState::propagateGate(qc::OpType gate,
                               std::vector<unsigned int> targets,
                               std::vector<unsigned int> posCtrls,
                               std::vector<unsigned int> negCtrls,
                               std::vector<double> params) {
  auto gateMapping = getQubitMappingOfGates(gate, params);

  unsigned int positiveCtrlMask = 0;
  unsigned int negativeCtrlMask = 0;
  for (unsigned int const posCtrl : posCtrls) {
    positiveCtrlMask += static_cast<unsigned int>(pow(2, posCtrl) + 0.1);
  }
  for (unsigned int const negCtrl : negCtrls) {
    negativeCtrlMask += static_cast<unsigned int>(pow(2, negCtrl) + 0.1);
  }

  std::unordered_map<unsigned int, std::complex<double>> newValues;
  std::unordered_map<unsigned int, unsigned int> bitmaskForQubitTargets;
  if (targets.size() == 2) {
    bitmaskForQubitTargets.insert({3, pow(2, targets[1]) + pow(2, targets[0])});
    bitmaskForQubitTargets.insert({2, pow(2, targets[0])});
    bitmaskForQubitTargets.insert({1, pow(2, targets[1])});
  } else {
    bitmaskForQubitTargets.insert({1, pow(2, targets[0])});
  }

  if (targets.size() == 2) {
    newValues =
        getNewMappingForTwoQubitGate(gateMapping, bitmaskForQubitTargets,
                                     positiveCtrlMask, negativeCtrlMask);
  } else if (targets.size() == 1) {
    newValues =
        getNewMappingForSingleQubitGate(gateMapping, bitmaskForQubitTargets,
                                        positiveCtrlMask, negativeCtrlMask);
  }

  map.clear();
  for (const auto& [key, value] : newValues) {
    map.insert({key, value});
  }
  if (map.size() > maxNonzeroAmplitudes) {
    throw std::domain_error("Number of nonzero amplitudes too high. State "
                            "needs to be treated as TOP.");
  }
}

std::map<unsigned int, std::pair<double, std::shared_ptr<QubitState>>>
QubitState::measureQubit(unsigned int target) {
  unsigned int qubitMask = static_cast<unsigned int>(pow(2, target) + 0.1);

  double probabilityZero = 0.0;
  double probabilityOne = 0.0;
  std::unordered_map<unsigned int, std::complex<double>> newValuesZeroRes;
  std::unordered_map<unsigned int, std::complex<double>> newValuesOneRes;

  for (const auto& [key, value] : map) {
    if ((qubitMask & key) == 0) {
      probabilityZero += norm(value);
      newValuesZeroRes.insert({key, value});
    } else {
      probabilityOne += norm(value);
      newValuesOneRes.insert({key, value});
    }
  }

  if (std::abs(1.0 - probabilityZero - probabilityOne) > 1e-4) {
    throw std::domain_error(
        "Probabilities of 0 and 1 do not add up to one after measurement.");
  }

  QubitState stateZero = QubitState(nQubits, maxNonzeroAmplitudes);
  stateZero.map = newValuesZeroRes;
  stateZero.normalize();
  QubitState stateOne = QubitState(nQubits, maxNonzeroAmplitudes);
  stateOne.map = newValuesOneRes;
  stateOne.normalize();

  auto resPairZero =
      std::make_pair(probabilityZero, std::make_shared<QubitState>(stateZero));
  auto resPairOne =
      std::make_pair(probabilityOne, std::make_shared<QubitState>(stateOne));

  if (probabilityZero < 1e-4) {
    return {{1, resPairOne}};
  } else if (probabilityOne < 1e-4) {
    return {{0, resPairZero}};
  } else {
    return {{0, resPairZero}, {1, resPairOne}};
  }
}

void QubitState::removeQubit(unsigned int target) {
  throw std::logic_error("Not implemented");
}

void QubitState::normalize() {
  double denominator = 0.0;
  for (const auto& [key, value] : map) {
    denominator += norm(value);
  }
  for (const auto& [key, value] : map) {
    map[key] /= sqrt(denominator);
  }
}

bool QubitState::operator==(const QubitState& that) const {
  if (this->getSize() != that.getSize())
    return false;

  return std::ranges::all_of(
      this->map, [&](const std::pair<unsigned int, std::complex<double>>& p) {
        auto [key, val] = p;
        return (that.map.contains(key)) && (abs(val - that.map.at(key)) < 1e-4);
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
QubitStateOrTop::operator=(std::shared_ptr<QubitState> qubitState) {
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
    return this->getQubitState() == that.getQubitState();
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