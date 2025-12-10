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
QubitState::QubitState(std::size_t nQubits, std::size_t maxNonzeroAmplitudes) {
  this->nQubits = nQubits;
  this->maxNonzeroAmplitudes = maxNonzeroAmplitudes;
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
    std::string cn = std::format("{:.2f}", val.real());
    if (val.imag() > 1e-4) {
      cn += " + i" + std::format("{:.2f}", val.imag());
    } else if (val.imag() < -1e-4) {
      cn += " - i" + std::format("{:.2f}", -val.imag());
    }
    str += "|" + qubitStringToBinary(key) + "> -> " + cn + ", ";
  }

  return str;
}

QubitStateOrTop QubitState::unify(QubitState that) {
  throw std::logic_error("Not implemented");
}

QubitState QubitState::propagateGate(qc::OpType gate,
                                     std::vector<unsigned int> targets,
                                     std::vector<unsigned int> posCtrls,
                                     std::vector<unsigned int> negCtrls,
                                     std::vector<double> params) {
  auto gateMapping = getQubitMappingOfGates(gate, params);

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
    for (const auto& [key, value] : map) {
      unsigned int mapFrom;
      std::vector<unsigned int> keysForNewValue(4);

      if ((key & bitmaskForQubitTargets[3]) == bitmaskForQubitTargets[3]) {
        mapFrom = 3;
        keysForNewValue[3] = key;
        keysForNewValue[2] = key - bitmaskForQubitTargets[1];
        keysForNewValue[1] = key - bitmaskForQubitTargets[2];
        keysForNewValue[0] = key - bitmaskForQubitTargets[3];
      } else if ((key & bitmaskForQubitTargets[2]) ==
                 bitmaskForQubitTargets[2]) {
        mapFrom = 2;
        keysForNewValue[3] = key + bitmaskForQubitTargets[1];
        keysForNewValue[2] = key;
        keysForNewValue[1] = key ^ bitmaskForQubitTargets[3];
        keysForNewValue[0] = key - bitmaskForQubitTargets[2];
      } else if ((key & bitmaskForQubitTargets[1]) ==
                 bitmaskForQubitTargets[1]) {
        mapFrom = 1;
        keysForNewValue[3] = key + bitmaskForQubitTargets[2];
        keysForNewValue[2] = key ^ bitmaskForQubitTargets[3];
        keysForNewValue[1] = key;
        keysForNewValue[0] = key - bitmaskForQubitTargets[1];
      } else {
        mapFrom = 0;
        keysForNewValue[3] = key + bitmaskForQubitTargets[3];
        keysForNewValue[2] = key + bitmaskForQubitTargets[2];
        keysForNewValue[1] = key + bitmaskForQubitTargets[1];
        keysForNewValue[0] = key;
      }

      auto mapForThisQubit = gateMapping[mapFrom];
      for (int i = 0; i < 4; i++) {
        auto valueToI = mapForThisQubit[i];
        if (abs(valueToI) > 1e-4) {
          newValues[keysForNewValue[i]] += valueToI * value;
        }
      }
    }
  } else if (targets.size() == 1) {
    for (const auto& [key, value] : map) {
      unsigned int mapFrom;
      std::vector<unsigned int> keysForNewValue(2);

      if ((key & bitmaskForQubitTargets[1]) == bitmaskForQubitTargets[1]) {
        mapFrom = 1;
        keysForNewValue[1] = key;
        keysForNewValue[0] = key - bitmaskForQubitTargets[1];
      } else {
        mapFrom = 0;
        keysForNewValue[1] = key + bitmaskForQubitTargets[1];
        keysForNewValue[0] = key;
      }

      auto mapForThisQubit = gateMapping[mapFrom];
      for (int i = 0; i < 2; i++) {
        auto valueToI = mapForThisQubit[i];
        if (abs(valueToI) > 1e-4) {
          newValues[keysForNewValue[i]] += valueToI * value;
        }
      }
    }
  }

  // auto target = targets[0];
  // for (const auto& [key, value] : map) {
  //   unsigned int mapFrom;
  //   unsigned int zeroKey;
  //   unsigned int oneKey;
  //   unsigned int currentDigit = pow(2, target);
  //   if ((key & currentDigit) == 0) {
  //     mapFrom = 0;
  //     zeroKey = key;
  //     oneKey = key + currentDigit;
  //   } else {
  //     mapFrom = 1;
  //     zeroKey = key - currentDigit;
  //     oneKey = key;
  //   }
  //   auto mapForThisQubit = gateMapping[mapFrom];
  //   auto valueToZero = mapForThisQubit[0];
  //   auto valueToOne = mapForThisQubit[1];
  //   if (abs(valueToZero) > 1e-4) {
  //     newValues[zeroKey] += valueToZero * value;
  //   }
  //   if (abs(valueToOne) > 1e-4) {
  //     newValues[oneKey] += valueToOne * value;
  //   }
  // }
  map.clear();
  for (const auto& [key, value] : newValues) {
    map.insert({key, value});
  }

  return *this;
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