/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include <algorithm>
#include <complex>
#include <cstddef>
#include <map>
#include <memory>
#include <unordered_map>
#include <variant>

struct MeasurementResult {
  bool measurementResult;
  double probability;
};

/**
 * @brief This class represents a qubit state.
 *
 * This class holds n qubits in different basis states with their corresponding
 * complex amplitude.
 */
class QubitState {
public:
  explicit QubitState(std::size_t nQubits) {
    this->nQubits = nQubits;
    this->map = std::unordered_map<unsigned int, std::complex<double>>();
    this->map.insert({0, std::complex<double>(1.0, 0.0)});
  }

  ~QubitState() = default;

  [[nodiscard("QubitsState::getSize called but ignored")]] std::size_t
  getSize() const {
    return this->map.size();
  }

  [[nodiscard("QubitsState::getNQubits called but ignored")]] size_t
  getNQubits() const {
    return this->nQubits;
  }

  void print(std::ostream& os) const { os << this->toString(); }

  std::string toString() const {
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

  /**
   * @brief This method unifies two QubitStates.
   *
   * This method unifies the current QubitState with the given one and returns
   * a new QubitsState.
   *
   * @param that The QubitState to unify this with.
   * @return A new unified QubitState.
   */
  QubitState unify(const QubitState that) {
    throw std::logic_error("Not implemented");
  }

  /**
   * @brief This method applies a gate to the qubits.
   *
   * This method changes the amplitudes of a QubitsState according to the
   * applied gate.
   *
   * @param gate The name of the gate to be applied.
   * @param targets An array of the indices of the target qubits.
   * @param posCtrls An array of the indices of the ctrl qubits.
   * @param negCtrls An array of the indices of the negative ctrl qubits.
   */
  void propagateGate(std::string gate, unsigned int targets[],
                     unsigned int posCtrls[], unsigned int negCtrls[]) {
    throw std::logic_error("Not implemented");
  }

  /**
   * @brief This method applies a measurement to the qubits.
   *
   * This method applies a measurement to the qubits. It returns the QubitState
   * in case the measurement was 0 and in case it was 1, alongside the
   * respective probabilities.
   *
   * @param target The index of the qubit to be measured.
   * @return A map of two MeasurementResults pointing to the QubitStates after
   * measurement.
   */
  std::map<MeasurementResult, QubitState> measureQubit(unsigned int target) {
    throw std::logic_error("Not implemented");
  }

  /**
   * @brief This method removes a qubit from the state.
   *
   * This method removes a qubit from the state. It normalizes the sum of the
   * squared amplitudes of the other qubits back to one.
   *
   * @param target The index of the qubit to be removed.
   */
  void removeQubit(unsigned int target) {
    throw std::logic_error("Not implemented");
  }

  bool operator==(const QubitState& that) const {
    if (this->getSize() != that.getSize())
      return false;

    return std::ranges::all_of(
        this->map, [&](const std::pair<unsigned int, std::complex<double>>& p) {
          auto [key, val] = p;
          return (that.map.contains(key)) && (val == that.map.at(key));
        });
  }

private:
  std::size_t nQubits;
  std::unordered_map<unsigned int, std::complex<double>> map;

  std::string qubitStringToBinary(unsigned int q) const {
    std::string str;
    for (int i = nQubits - 1; i >= 0; i--) {
      unsigned int currentDigit = 2 ^ i;
      if (q >> currentDigit) {
        str += "1";
        q -= currentDigit;
      } else {
        str += "0";
      }
    }
    return str;
  }
};

enum TOP { T };

class QubitStateOrTop {
private:
  std::variant<TOP, std::shared_ptr<QubitState>> variant;

public:
  QubitStateOrTop() : variant(TOP::T) {}

  QubitStateOrTop(TOP top) : variant(top) {}

  QubitStateOrTop(std::shared_ptr<QubitState> qubitState)
      : variant(qubitState) {}

  QubitStateOrTop(const QubitStateOrTop& qubitStateOrTop) = default;

  QubitStateOrTop& operator=(const QubitStateOrTop& qubitStateOrTop) = default;

  QubitStateOrTop& operator=(const std::shared_ptr<QubitState>& qubitState) {
    this->variant = qubitState;
    return *this;
  }

  QubitStateOrTop& operator=(const TOP& t) {
    this->variant = t;
    return *this;
  }

  bool operator==(const QubitStateOrTop& that) const {
    if (this->isTop() && that.isTop()) {
      return true;
    } else if (this->isTop() || that.isTop()) {
      return false;
    } else {
      return *this->getQubitState() == *that.getQubitState();
    }
  }

  bool operator!=(const QubitStateOrTop& that) const {
    return !(that == *this);
  }

  ~QubitStateOrTop() = default;

  [[nodiscard("QubitsStateOrTop::isTop called but ignored")]] bool
  isTop() const {
    return std::holds_alternative<TOP>(variant);
  }

  [[nodiscard("QubitsStateOrTop::isQubitState called but ignored")]] bool
  isQubitState() const {
    return std::holds_alternative<std::shared_ptr<QubitState>>(variant);
  }

  [[nodiscard("QubitsStateOrTop::getQubitState called but ignored")]] std::
      shared_ptr<QubitState>
      getQubitState() const {
    return std::get<std::shared_ptr<QubitState>>(variant);
  }

  [[nodiscard("QubitsStateOrTop::to_string called but ignored")]] std::string
  toString() const {
    if (isTop()) {
      return "TOP";
    } else {
      return getQubitState()->to_string();
    }
  }

  void print(std::ostream& os) const { os << this->toString(); }
};