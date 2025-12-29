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
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
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
  // At position j the local mapping k (in the respective hybrid state) of qubit
  // j is given
  std::vector<unsigned int> mappingGlobalToLocalQubitIndices;
  // At position j the local mapping k (in the respective hybrid state) of bit j
  // is given
  std::vector<unsigned int> mappingGlobalToLocalBitIndices;
  // The entries i, l and k point to the set of hybrid states corresponding to
  // qubits i, l and k. Two entries point to the same set if the qubits are in
  // one set. They point to the same sets as hRegOfBits.
  std::vector<std::shared_ptr<std::vector<HybridStateOrTop>>> hRegOfQubits;
  // The entries i, l and k point to the set of hybrid states corresponding to
  // bits i, l and k. Two entries point to the same set if the bits are in one
  // set. They point to the same sets as hRegOfQubits.
  std::vector<std::shared_ptr<std::vector<HybridStateOrTop>>> hRegOfBits;
  // Pairs of vectors of qubit, bit indices which are in the same hybrid state.
  std::set<std::pair<std::set<unsigned int>, std::set<unsigned int>>>
      indicesInSameState;

  /**
   * @brief This method retrieves the states containing the given qubits and
   * bits.
   *
   * This method retrieves a set of pairs, with the pairs holding the qubit and
   * bit indices of all states that hold at least one of the given qubit and
   * bit indices.
   *
   * @param qubits The qubit indices.
   * @param bits The bit indices
   * @return A set of pairs with all involved qubit and bit indices.
   */
  [[nodiscard("UnionTable::getInvolvedStates called but ignored")]] std::set<
      std::pair<std::set<unsigned int>, std::set<unsigned int>>>
  getInvolvedStates(const std::set<unsigned int>& qubits,
                    const std::set<unsigned int>& bits) const;

  /**
   * @brief Returns true if only the values of at maximum one given set of the
   * qubits is nonzero.
   *
   * This method receives a number of qubit indices, vectors of possible
   * amplitudes of the qubits and a set of pairs of involved indices. It checks
   * whether more than one of the value sets corresponds to nonzero amplitudes.
   * Does a recursive call to map to all HybridStates in case qubits from
   * multiple states are involved.
   *
   * @param qubits The qubits which are being checked.
   * @param values The sets of values for which is tested whether there is a
   * nonzero amplitude.
   * @param involvedIndices The indices grouped by the qubits in one state.
   * @returns True if at maximum the amplitudes in one set are not equal to
   * zero.
   */
  bool checkIfOnlyOneSetIsNotZero(
      std::vector<unsigned int> qubits,
      const std::set<std::vector<unsigned int>>& values,
      const std::set<std::pair<std::set<unsigned int>, std::set<unsigned int>>>&
          involvedIndices);

  /**
   * @brief This method unifies two hybrid states.
   *
   * This method unifies the two hybrid states pointed to by involvedStates1
   * and involvedStates2. If any of the hybridStates are top, the result
   * will be top.
   *
   * @param involvedStates1 The first qubit and bit indices of the states to
   * be unified.
   * @param involvedStates2 The first qubit and bit indices of the states to
   * be unified.
   */
  std::pair<std::set<unsigned int>, std::set<unsigned int>> unifyHybridStates(
      std::pair<std::set<unsigned int>, std::set<unsigned int>> involvedStates1,
      std::pair<std::set<unsigned int>, std::set<unsigned int>>
          involvedStates2);

  /**
   * @brief This method swaps two qubits without fusing the hybrid states of the
   * two.
   *
   * @param target1 Swapped with target2.
   * @param target2 Swapped with target1.
   */
  void applySwapGate(unsigned int target1, unsigned int target2);

public:
  UnionTable(unsigned int maxNonzeroAmplitudes,
             unsigned int maxNumberOfBitValues);

  ~UnionTable();

  [[nodiscard("UnionTable::toString called but ignored")]] std::string
  toString() const;

  bool allTop();

  [[nodiscard("UnionTable::getNumberOfBits called but ignored")]] unsigned int
  getNumberOfBits() const;

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
   * @throws invalid_argument if classicalTarget is given, but is not found in
   * the existing bits and also not the next non-existing bit index.
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
   * the UnionTable in zero state. The method returns the index with which the
   * qubit can be identified in the UnionTable.
   *
   * @return The index with which the qubit can be identified in the UnionTable.
   */
  unsigned int propagateQubitAlloc();

  /**
   * @brief This method propagates a newly defined bit and returns the bit's
   * index.
   *
   * This method propagates a newly defined bit. This means that the bit is
   * added to the UnionTable with the given value. The method returns the index
   * with which the bit can be identified in the UnionTable.
   *
   * @param value The value that the bit should get.
   * @return The index with which the bit can be identified in the UnionTable.
   */
  unsigned int propagateBitDef(bool value);

  [[nodiscard("UnionTable::isQubitAlwaysOne called but ignored")]] bool
  isQubitAlwaysOne(size_t q) const;

  [[nodiscard("UnionTable::isQubitAlwaysZero called but ignored")]] bool
  isQubitAlwaysZero(size_t q) const;

  [[nodiscard("UnionTable::isBitAlwaysOne called but ignored")]] bool
  isBitAlwaysOne(size_t q) const;

  [[nodiscard("UnionTable::isBitAlwaysZero called but ignored")]] bool
  isBitAlwaysZero(size_t q) const;

  /**
   * @brief Checks if a given combination of bit-qubit values has a nonzero
   * probability
   *
   * This method receives a number of qubit  and bit indices and checks whether
   * they have for a given value always a zero amplitude. If the qubit or bit
   * values are top, it is not guaranteed that the amplitude is always zero and
   * false is returned.
   *
   * @param qubits The qubits which are being checked.
   * @param value The value for which is tested whether there is a nonzero
   * amplitude.
   * @param bits The bit indices to check.
   * @param bitValues The values of the bits to check. The value at k is the
   * value of bit index at k.
   * @returns True if the amplitude is always zero, false otherwise.
   */
  bool hasAlwaysZeroAmplitude(const std::vector<unsigned int>& qubits,
                              unsigned int value,
                              const std::vector<unsigned int>& bits = {},
                              std::vector<bool> bitValues = {});

  /**
   * @brief Returns whether the given qubit and the given bit always have the
   * same value or always a different value.
   *
   * @param bit Index of bit.
   * @param qubit Index of qubit.
   * @returns Non-empty optional if the bit and qubit have always the same or
   * different values. Optional contains true if they have the same value, false
   * if they have always different values.
   */
  std::optional<bool> getIsBitEquivalentToQubit(unsigned int bit,
                                                unsigned int qubit);

  /**
   * @brief Returns true if only the values of one given set of teh qubits is
   * nonzero.
   *
   * This method receives a number of qubit indices and vectors of possible
   * amplitudes of the qubits. It checks whether more than one of the value sets
   * corresponds to nonzero amplitudes.
   *
   * @param qubits The qubits which are being checked.
   * @param values The sets of values for which is tested whether there is a
   * nonzero amplitude.
   * @returns True if at maximum the amplitudes in one set are not equal to
   * zero.
   */
  bool isOnlyOneSetNotZero(std::vector<unsigned int> qubits,
                           const std::set<std::vector<unsigned int>>& values);
};
} // namespace mqt::ir::opt::qcp
#endif // MQT_CORE_UNIONTABLE_H
