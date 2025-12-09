/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_CORE_UNIONTABLE_H
#define MQT_CORE_UNIONTABLE_H

#include "ir/operations/OpType.hpp"

#include <cstddef>
#include <vector>

namespace mqt::ir::opt::qcp {
class HybridStateOrTop;
/**
 * @brief This class holds the hybrid states of multiple qubits and bits.
 *
 * This class holds the hybrid states of multiple qubits and bits. It can unify
 * the states. One can propagate gates, measurements and resets through the
 * states.
 */
class UnionTable {
  std::size_t nQubits;
  std::size_t nBits;
  HybridStateOrTop* hReg;

public:
  UnionTable(size_t nQubits, size_t nBits);

  ~UnionTable();

  /**
   * @brief This method unifies hybrid states.
   *
   * This method unifies the hybrid states that consit of the given bits and
   * qubits.
   *
   * @param qubits The qubits that should be unified.
   * @param bits The bits that should be unified.
   */
  void unify(std::vector<unsigned int> qubits, std::vector<unsigned int> bits);

  bool allTop();

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
  void propagateGate(qc::OpType gate, std::vector<unsigned int> targets,
                     std::vector<unsigned int> posCtrls,
                     std::vector<unsigned int> negCtrls,
                     std::vector<double> params = {});

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
                            unsigned int classicalTarget);

  /**
   * @brief This method propagates a qubit reset.
   *
   * This method propagates a qubit reset. This means that the qubit is put into
   * zero state. It is also put in its own QubitState again if it does not
   * correspond to already assigned bit values.
   *
   * @param target The index of the qubit to be reset.
   */
  void propagateReset(unsigned int target);

  /**
   * @brief This method propagates a qubit alloc and returns the qubit's index.
   *
   * This method propagates a qubit alloc. This means that the qubit is added to
   * the UnionTable in zero state. The methos returns the index with which the
   * qubit can be identified in the UnionTable.
   *
   * @return The index with which the qubit can be identified in the UnionTable.
   */
  unsigned int propagateQubitAlloc();

  /**
   * @brief This method propagates a qubit dealloc.
   *
   * This method propagates a qubit dealloc. This means that the qubit removed
   * from the UnionTable.
   *
   * @param target The index of the qubit to be removed.
   */
  void propagateQubitDealloc(unsigned int target);

  bool isQubitAlwaysOne(size_t q);

  bool isQubitAlwaysZero(size_t q);

  bool isBitAlwaysOne(size_t q);

  bool isBitAlwaysZero(size_t q);

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
                           unsigned int value);
};
} // namespace mqt::ir::opt::qcp
#endif // MQT_CORE_UNIONTABLE_H
