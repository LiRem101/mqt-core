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

#include "HybridState.hpp"
#include "ir/operations/OpType.hpp"

#include <cstddef>
#include <set>
#include <vector>

namespace mqt::ir::opt::qcp {
/**
 * @brief This class holds the hybrid states of multiple qubits and bits.
 *
 * This class holds the hybrid states of multiple qubits and bits. It can unify
 * the states. One can propagate gates, measurements and resets through the
 * states.
 */
class UnionTable {
  unsigned int maxNonzeroAmplitudes;
  unsigned int maxNumberOfBitValues;
  // At position i the local mapping k (in the respective hybrid state) of qubit
  // i is given
  std::vector<unsigned int> mappingGlobalToLocalQubitIndices;
  // At position i the local mapping k (in the respective hybrid state) of bit i
  // is given
  std::vector<unsigned int> mappingGlobalToLocalBitIndices;
  // The entries i, l and k point to the set of hybrid states corresponding to
  // qubits i, l and k. Two entries point to the same set if the qubits are in
  // one set. They point to the same sets as hRegOfBits.
  std::vector<std::shared_ptr<std::set<HybridStateOrTop>>> hRegOfQubits;
  // The entries i, l and k point to the set of hybrid states corresponding to
  // bits i, l and k. Two entries point to the same set if the bits are in one
  // set. They point to the same sets as hRegOfQubits.
  std::vector<std::shared_ptr<std::set<HybridStateOrTop>>> hRegOfBits;
  // Pairs of vectors of qubit, bit indices which are in the same hybrid state.
  std::set<std::pair<std::vector<unsigned int>, std::vector<unsigned int>>>
      indizesInSameState;

public:
  UnionTable(unsigned int maxNonzeroAmplitudes,
             unsigned int maxNumberOfBitValues);

  ~UnionTable();

  std::string toString() const;

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
   * @param posCtrlsQuantum An array of the indices of the ctrl qubits.
   * @param negCtrlsQuantum An array of the indices of the negative ctrl qubits.
   * @param posCtrlsClassical An array of the indices of the ctrl bits.
   * @param negCtrlsClassical An array of the indices of the negative ctrl bits.
   * @param params The parameter applied to the gate.
   */
  void propagateGate(qc::OpType gate, std::vector<unsigned int> targets,
                     std::vector<unsigned int> posCtrlsQuantum = {},
                     std::vector<unsigned int> negCtrlsQuantum = {},
                     const std::vector<unsigned int>& posCtrlsClassical = {},
                     const std::vector<unsigned int>& negCtrlsClassical = {},
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

  /**
   * @brief This method propagates a bit definition and returns the bit's index.
   *
   * This method propagates a bit definition. This means that the bit is added
   * to the UnionTable with the given value. The methos returns the index with
   * which the bit can be identified in the UnionTable.
   *
   * @param value The value that the bit should get.
   * @return The index with which the bit can be identified in the UnionTable.
   */
  unsigned int propagateBitDef(bool value);

  bool isQubitAlwaysOne(size_t q);

  bool isQubitAlwaysZero(size_t q);

  bool isBitAlwaysOne(size_t q);

  bool isBitAlwaysZero(size_t q);

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
  bool hasAlwaysZeroAmplitude(std::vector<unsigned int> qubits,
                              unsigned int value);
};
} // namespace mqt::ir::opt::qcp
#endif // MQT_CORE_UNIONTABLE_H
