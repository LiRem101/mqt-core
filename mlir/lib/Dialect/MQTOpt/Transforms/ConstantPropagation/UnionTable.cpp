/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "HybridState.cpp"

#include <cstddef>
/**
 * @brief This class holds the hybrid states of multiple qubits and bits.
 *
 * This class holds the hybrid states of multiple qubits and bits. It can unify
 * the states. One can propagate gates, measurements and resets through the
 * states.
 */
class UnionTable {
public:
  UnionTable(size_t nQubits, size_t nBits) {
    this->nQubits = nQubits;
    this->nBits = nBits;
    this->hReg = new HybridStateOrTop[nQubits];
    for (size_t i = 0; i < nQubits; i++) {
      this->hReg[i] = std::make_shared<HybridState>(1, std::vector<bool>(), 1);
    }
  }

  ~UnionTable() { delete[] this->hReg; }

  /**
   * @brief This method unifies hybrid states.
   *
   * This method unifies the hybrid states that consit of the given bits and
   * qubits.
   *
   * @param qubits The qubits that should be unified.
   * @param bits The bits that should be unified.
   */
  void unify(std::vector<unsigned int> qubits, std::vector<unsigned int> bits) {
    throw std::logic_error("Not implemented");
  }

  bool allTop() {
    for (size_t i = 0; i < nQubits; i++) {
      if (!hReg[i].isTop()) {
        return false;
      }
    }
    return true;
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
  void propagateReset(unsigned int target) {
    throw std::logic_error("Not implemented");
  }

  bool isQubitAlwaysOne(size_t q) { throw std::logic_error("Not implemented"); }

  bool isQubitAlwaysZero(size_t q) {
    throw std::logic_error("Not implemented");
  }

  bool isBitAlwaysOne(size_t q) { throw std::logic_error("Not implemented"); }

  bool isBitAlwaysZero(size_t q) { throw std::logic_error("Not implemented"); }

  /**
   * @brief Returns whether the given qubits have for value values a nonzero
   * amplitude.
   *
   * This method receives a number of qubit indices and checks whether they have
   * for a given value a nonzero amplitude.
   *
   * @param qubits The qubits which are being checked.
   * @param value The value for which is tested whether there is a nonzero
   * amplitude.
   * @returns True if the amplitude is nonzero, false otherwise.
   */
  bool hasNonzeroAmplitude(std::vector<unsigned int> qubits,
                           unsigned int value) {
    throw std::logic_error("Not implemented");
  }

private:
  std::size_t nQubits;
  std::size_t nBits;
  HybridStateOrTop* hReg;
};