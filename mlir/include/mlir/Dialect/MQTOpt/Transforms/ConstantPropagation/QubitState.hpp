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

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <ir/operations/OpType.hpp>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace mqt::ir::opt::qcp {

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
    for (int i = static_cast<int>(nQubits) - 1; i >= 0; i--) {
      if (const auto currentDigit = static_cast<unsigned int>(pow(2, i));
          q & currentDigit) {
        str += "1";
        q -= currentDigit;
      } else {
        str += "0";
      }
    }
    return str;
  }

  /**
   * @brief This method receives a two qubit gate mapping and a bitmask for
   * target, pos ctrls and neg ctrl qubits.
   *
   * This method receives a two qubit gate mapping and a bitmask for target, pos
   * ctrls and neg ctrl qubits. The gate is applied to the valid qubit states.
   * It returns the map which would be the qubit state after gate application.
   *
   * @param gateMapping The mapping representing the gate
   * @param bitmaskForQubitTargets The bitmask of the qubit targets. I.e. 011,
   * if zeroth and first qubit are targets.
   * @param bitmaskForPosCtrls The bitmask of the positively controlling qubits.
   * @param bitmaskForNegCtrls The bitmask of the negatively controlling qubits.
   * @return The qubit state after the gate has been applied.
   */
  std::unordered_map<unsigned int, std::complex<double>>
  getNewMappingForTwoQubitGate(
      std::unordered_map<unsigned int,
                         std::unordered_map<unsigned int, std::complex<double>>>
          gateMapping,
      std::unordered_map<unsigned int, unsigned int> bitmaskForQubitTargets,
      const unsigned int bitmaskForPosCtrls,
      const unsigned int bitmaskForNegCtrls) {
    std::unordered_map<unsigned int, std::complex<double>> newValues;

    for (const auto& [key, value] : map) {
      if ((bitmaskForPosCtrls & key) != bitmaskForPosCtrls ||
          (bitmaskForNegCtrls & key) != 0) {
        newValues[key] += value;
        continue;
      }

      unsigned int mapFrom = 0;
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
        keysForNewValue[3] = key + bitmaskForQubitTargets[3];
        keysForNewValue[2] = key + bitmaskForQubitTargets[2];
        keysForNewValue[1] = key + bitmaskForQubitTargets[1];
        keysForNewValue[0] = key;
      }

      auto mapForThisQubit = gateMapping[mapFrom];
      for (int i = 0; i < 4; i++) {
        if (auto valueToI = mapForThisQubit[i]; abs(valueToI) > 1e-4) {
          newValues[keysForNewValue[i]] += valueToI * value;
        }
      }
    }

    return newValues;
  }

  /**
   * @brief This method receives a single qubit gate mapping and a bitmask for
   * target, pos ctrls and neg ctrl qubits.
   *
   * This method receives a single qubit gate mapping and a bitmask for target,
   * pos ctrls and neg ctrl qubits. The gate is applied to the valid qubit
   * states. It returns the map which would be the qubit state after gate
   * application.
   *
   * @param gateMapping The mapping representing the gate
   * @param bitmaskForQubitTargets The bitmask of the qubit targets. I.e. 011,
   * if zeroth and first qubit are targets.
   * @param bitmaskForPosCtrls The bitmask of the positively controlling qubits.
   * @param bitmaskForNegCtrls The bitmask of the negatively controlling qubits.
   * @return The qubit state after the gate has been applied.
   */
  std::unordered_map<unsigned int, std::complex<double>>
  getNewMappingForSingleQubitGate(
      std::unordered_map<unsigned int,
                         std::unordered_map<unsigned int, std::complex<double>>>
          gateMapping,
      std::unordered_map<unsigned int, unsigned int> bitmaskForQubitTargets,
      const unsigned int bitmaskForPosCtrls,
      const unsigned int bitmaskForNegCtrls) {
    std::unordered_map<unsigned int, std::complex<double>> newValues;

    for (const auto& [key, value] : map) {
      if ((bitmaskForPosCtrls & key) != bitmaskForPosCtrls ||
          (bitmaskForNegCtrls & key) != 0) {
        newValues[key] += value;
        continue;
      }

      unsigned int mapFrom = 0;
      std::vector<unsigned int> keysForNewValue(2);

      if ((key & bitmaskForQubitTargets[1]) == bitmaskForQubitTargets[1]) {
        mapFrom = 1;
        keysForNewValue[1] = key;
        keysForNewValue[0] = key - bitmaskForQubitTargets[1];
      } else {
        keysForNewValue[1] = key + bitmaskForQubitTargets[1];
        keysForNewValue[0] = key;
      }

      auto mapForThisQubit = gateMapping[mapFrom];
      for (int i = 0; i < 2; i++) {
        if (auto valueToI = mapForThisQubit[i]; abs(valueToI) > 1e-4) {
          newValues[keysForNewValue[i]] += valueToI * value;
        }
      }
    }

    return newValues;
  }

public:
  QubitState(std::size_t nQubits, std::size_t maxNonzeroAmplitudes);

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
   * a new QubitState, if the new state has no more than maxNonzeroAmplitude
   * nonzero amplitudes. Otherwise, throws a domain_error.
   *
   * @param that The QubitState to unify this with.
   * @param qubitsOccupiedByThat Qubit positions that-QubitState will
   * provide.
   * @throw std::domain_error If the number of nonzero amplitudes would exceed
   * maxNonzeroAmplitudes of this.
   */
  QubitState unify(const QubitState& that,
                   std::vector<unsigned int> qubitsOccupiedByThat);

  /**
   * @brief This method applies a gate to the qubits.
   *
   * This method changes the amplitudes of a QubitsState according to the
   * applied gate. Returns the current qubitState if it has no more than
   * maxNonZeroAmplitude nonzero amplitudes. Otherwise, throws a domain_error.
   *
   * @param gate The gate to be applied.
   * @param targets A vector of the indices of the target qubits.
   * @param posCtrls A vector of the indices of the ctrl qubits.
   * @param negCtrls A vector of the indices of the negative ctrl qubits.
   * @param params The parameter applied to the gate.
   * @throw std::domain_error If the number of nonzero amplitudes would exceed
   * maxNonzeroAmplitudes.
   */
  void propagateGate(qc::OpType gate, const std::vector<unsigned int>& targets,
                     const std::vector<unsigned int>& posCtrls = {},
                     const std::vector<unsigned int>& negCtrls = {},
                     const std::vector<double>& params = {});

  /**
   * @brief This method applies a measurement to the qubits.
   *
   * This method applies a measurement to the qubits. It returns the QubitState
   * in case the measurement was 0 and in case it was 1, alongside the
   * respective probabilities.
   *
   * @param target The index of the qubit to be measured.
   * @return A map of the measurement result (zero and/or one) pointing to the
   * probability for the result and the QubitStates after measurement.
   */
  std::map<unsigned int, std::pair<double, std::shared_ptr<QubitState>>>
  measureQubit(unsigned int target);

  /**
   * @brief This method resets a qubit.
   *
   * This method resets a qubit. This is done by assuming that a measurement is
   * applied and measurements in the one-state are set to the zero state. In
   * order to not get a mixed state, the method returns one to two QubitStates,
   * one in case the measurement was 0 and one in case it was 1, alongside the
   * respective probabilities. In both cases, the target qubit will be zero (as
   * a reset is performed).
   *
   * @param target The index of the qubit to be measured.
   * @return A set of one to two QubitStates and their corresponding
   * probabilities.
   */
  std::set<std::pair<double, std::shared_ptr<QubitState>>>
  resetQubit(unsigned int target);

  /**
   * @brief This method normalizes the amplitudes of a state.
   */
  void normalize();

  bool operator==(const QubitState& that) const;

  [[nodiscard("QubitState::isQubitAlwaysOne called but ignored")]] bool
  isQubitAlwaysOne(size_t q) const;

  [[nodiscard("QubitState::isQubitAlwaysZero called but ignored")]] bool
  isQubitAlwaysZero(size_t q) const;

  /**
   * @brief Returns whether the given qubits have for value always a zero
   * amplitude.
   *
   * This method receives a number of qubit indices and checks whether they have
   * for a given value always a zero amplitude. If the qubit values are top, it
   * is not guaranteed that the amplitude is always zero and false is returned.
   *
   * @param qubits The qubits which are being checked.
   * @param value The value for which is tested whether there is a nonzero
   * amplitude.
   * @returns True if the amplitude is always zero, false otherwise.
   */
  [[nodiscard("QubitState::hasAlwaysZeroAmplitude called but ignored")]] bool
  hasAlwaysZeroAmplitude(const std::vector<unsigned int>& qubits,
                         unsigned int value) const;
};

enum TOP : std::uint8_t { T };

class QubitStateOrTop {
  std::variant<TOP, std::shared_ptr<QubitState>> variant;

public:
  QubitStateOrTop();

  explicit QubitStateOrTop(TOP top);

  explicit QubitStateOrTop(std::shared_ptr<QubitState> qubitState);

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

  [[nodiscard("QubitStateOrTop::isQubitAlwaysOne called but ignored")]] bool
  isQubitAlwaysOne(size_t q) const;

  [[nodiscard("QubitStateOrTop::isQubitAlwaysZero called but ignored")]] bool
  isQubitAlwaysZero(size_t q) const;

  /**
   * @brief Returns whether the given qubits have for value always a zero
   * amplitude.
   *
   * This method receives a number of qubit indices and checks whether they have
   * for a given value always a zero amplitude. If the qubit values are top, it
   * is not guaranteed that the amplitude is always zero and false is returned.
   *
   * @param qubits The qubits which are being checked.
   * @param value The value for which is tested whether there is a nonzero
   * amplitude.
   * @returns True if the amplitude is always zero, false otherwise.
   */
  [[nodiscard(
      "QubitStateOrTop::hasAlwaysZeroAmplitude called but ignored")]] bool
  hasAlwaysZeroAmplitude(const std::vector<unsigned int>& qubits,
                         unsigned int value) const;
};
} // namespace mqt::ir::opt::qcp

#endif // MQT_CORE_QUBITSTATE_H