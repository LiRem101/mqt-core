/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_CORE_QUBITSTATE_H
#define MQT_CORE_QUBITSTATE_H

#include <algorithm>
#include <complex>
#include <cstddef>
#include <ir/operations/OpType.hpp>
#include <map>
#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>

namespace mqt::ir::opt::qcp {
struct MeasurementResult {
  bool measurementResult;
  double probability;
};

class QubitStateOrTop;

/**
 * @brief This class represents a qubit state.
 *
 * This class holds n qubits in different basis states with their corresponding
 * complex amplitude.
 */
class QubitState {
  std::size_t nQubits;
  std::size_t maxNonzeroAmplitudes;
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

public:
  explicit QubitState(std::size_t nQubits, std::size_t maxNonzeroAmplitudes);

  ~QubitState();

  [[nodiscard("QubitState::getSize called but ignored")]] std::size_t
  getSize() const;

  [[nodiscard("QubitState::getNQubits called but ignored")]] size_t
  getNQubits() const;

  void print(std::ostream& os) const;

  std::string toString() const;

  /**
   * @brief This method unifies two QubitStates.
   *
   * This method unifies the current QubitState with the given one and returns
   * a new QubitsState, if the new state has no more than maxNonzeroAmplitude
   * nonzero amplitudes.
   *
   * @param that The QubitState to unify this with.
   * @return A new unified QubitState or TOP.
   */
  QubitStateOrTop unify(QubitState that);

  /**
   * @brief This method applies a gate to the qubits.
   *
   * This method changes the amplitudes of a QubitsState according to the
   * applied gate. Returns the current qubitState if it has no more than
   * maxNonZeroAmplitude nonzero amplitudes. Otherwise, it returns top.
   *
   * @param gate The gate to be applied.
   * @param targets A vector of the indices of the target qubits.
   * @param posCtrls A vector of the indices of the ctrl qubits.
   * @param negCtrls A vector of the indices of the negative ctrl qubits.
   * @return A new unified QubitState or TOP.
   */
  QubitStateOrTop propagateGate(qc::OpType gate,
                                std::vector<unsigned int> targets,
                                std::vector<unsigned int> posCtrls = {},
                                std::vector<unsigned int> negCtrls = {});

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
  std::map<MeasurementResult, QubitState> measureQubit(unsigned int target);

  /**
   * @brief This method removes a qubit from the state.
   *
   * This method removes a qubit from the state. It normalizes the sum of the
   * squared amplitudes of the other qubits back to one.
   *
   * @param target The index of the qubit to be removed.
   */
  void removeQubit(unsigned int target);

  bool operator==(const QubitState& that) const;
};

enum TOP { T };

class QubitStateOrTop {
  std::variant<TOP, std::shared_ptr<QubitState>> variant;

public:
  QubitStateOrTop();

  QubitStateOrTop(TOP top);

  QubitStateOrTop(std::shared_ptr<QubitState> qubitState);

  QubitStateOrTop(const QubitStateOrTop& qubitStateOrTop);

  QubitStateOrTop& operator=(const QubitStateOrTop& qubitStateOrTop);

  QubitStateOrTop& operator=(const std::shared_ptr<QubitState>& qubitState);

  QubitStateOrTop& operator=(const TOP& t);

  bool operator==(const QubitStateOrTop& that) const;

  bool operator!=(const QubitStateOrTop& that) const;

  ~QubitStateOrTop();

  [[nodiscard("QubitStateOrTop::isTop called but ignored")]] bool isTop() const;

  [[nodiscard("QubitStateOrTop::isQubitState called but ignored")]] bool
  isQubitState() const;

  [[nodiscard("QubitStateOrTop::getQubitState called but ignored")]] std::
      shared_ptr<QubitState>
      getQubitState() const;

  [[nodiscard("QubitStateOrTop::toString called but ignored")]] std::string
  toString() const;

  void print(std::ostream& os) const;
};
} // namespace mqt::ir::opt::qcp

#endif // MQT_CORE_QUBITSTATE_H