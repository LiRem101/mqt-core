/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "QubitState.cpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <llvm/ADT/STLExtras.h>
#include <memory>
#include <variant>

/**
 * @brief This class represents a hybrid state.
 *
 * This class holds a QubitState and zero to mmax additional bit values.
 * The class also hold a probability.
 */
class HybridState {
  QubitState qState;
  double probability;
  std::vector<bool> bitValues;

public:
  explicit HybridState(std::size_t nQubits, std::vector<bool> bitValues,
                       double probability)
      : qState(QubitState(nQubits)), probability(probability),
        bitValues(std::move(bitValues)) {}

  ~HybridState() = default;

  void print(std::ostream& os) const { os << this->toString(); }

  std::string toString() const {
    std::string str = "{" + this->qState.toString() + "}: ";
    for (auto bit : bitValues) {
      str += bit ? "1 " : "0 ";
    }
    str += "p = " + std::to_string(this->probability) + ";";
    return str;
  }

  /**
   * @brief This method applies a gate to the state.
   *
   * This method changes the hybrid state according to a gate.
   *
   * @param gate The name of the gate to be applied.
   * @param targets An array of the indices of the target qubits.
   * @param posCtrlsQuantum An array of the indices of the ctrl qubits.
   * @param negCtrlsQuantum An array of the indices of the negative ctrl qubits.
   * @param posCtrlsClassical An array of the indices of the ctrl bits.
   * @param negCtrlsClassical An array of the indices of the negative ctrl bits.
   */
  void propagateGate(std::string gate, unsigned int targets[],
                     unsigned int posCtrlsQuantum[],
                     unsigned int negCtrlsQuantum[],
                     unsigned int posCtrlsClassical[],
                     unsigned int negCtrlsClassical[]) {
    throw std::logic_error("Not implemented");
  }

  /**
   * @brief This method applies a measurement.
   *
   * This method applies a measurement, changing the qubits and the classical
   * bit corresponding to the measurement.
   *
   * @param quantumTarget The index of the qubit to be measured.
   * @param classicalTarget The index of the bit to save the measurement result
   * in.
   */
  void propagateMeasurement(unsigned int quantumTarget,
                            unsigned int classicalTarget) {
    throw std::logic_error("Not implemented");
  }

  /**
   * @brief This method propagates a qubit reset.
   *
   * This method propagates a qubit reset. This means that the qubit is put into
   * zero state. It is also put in its own QubitState again if it does not
   * correspond to already assigned bit values.
   *
   * @param target The index of the qubit to be reset.
   */
  void resetQubit(unsigned int target) {
    throw std::logic_error("Not implemented");
  }

  bool operator==(const HybridState& that) const {
    if (this->probability != that.probability ||
        this->bitValues != that.bitValues)
      return false;

    return this->qState == that.qState;
  }
};

class HybridStateOrTop {
  std::variant<TOP, std::shared_ptr<HybridState>> variant;

public:
  HybridStateOrTop() : variant(TOP::T) {}

  HybridStateOrTop(TOP top) : variant(top) {}

  HybridStateOrTop(std::shared_ptr<HybridState> hybridState)
      : variant(hybridState) {}

  HybridStateOrTop(const HybridStateOrTop& hybridStateOrTop) = default;

  HybridStateOrTop&
  operator=(const HybridStateOrTop& hybridStateOrTop) = default;

  HybridStateOrTop& operator=(const std::shared_ptr<HybridState>& hybridState) {
    this->variant = hybridState;
    return *this;
  }

  HybridStateOrTop& operator=(const TOP& t) {
    this->variant = t;
    return *this;
  }

  bool operator==(const HybridStateOrTop& that) const {
    if (this->isTop() && that.isTop()) {
      return true;
    } else if (this->isTop() || that.isTop()) {
      return false;
    } else {
      return *this->getHybridState() == *that.getHybridState();
    }
  }

  bool operator!=(const HybridStateOrTop& that) const {
    return !(that == *this);
  }

  ~HybridStateOrTop() = default;

  [[nodiscard("HybridStateOrTop::isTop called but ignored")]] bool
  isTop() const {
    return std::holds_alternative<TOP>(variant);
  }

  [[nodiscard("HybridStateOrTop::isHybridState called but ignored")]] bool
  isHybridState() const {
    return std::holds_alternative<std::shared_ptr<HybridState>>(variant);
  }

  [[nodiscard("HybridStateOrTop::getHybridState called but ignored")]] std::
      shared_ptr<HybridState>
      getHybridState() const {
    return std::get<std::shared_ptr<HybridState>>(variant);
  }

  [[nodiscard("HybridStateOrTop::toString called but ignored")]] std::string
  toString() const {
    if (isTop()) {
      return "TOP";
    } else {
      return getHybridState()->toString();
    }
  }

  void print(std::ostream& os) const { os << this->toString(); }
};
