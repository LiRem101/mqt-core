/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_CORE_HYBRIDSTATE_H
#define MQT_CORE_HYBRIDSTATE_H

#include "QubitState.hpp"
#include "ir/operations/OpType.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <variant>
#include <vector>

namespace mqt::ir::opt::qcp {
/**
 * @brief This class represents a hybrid state.
 *
 * This class holds a QubitState and zero to mmax additional bit values.
 * The class also hold a probability.
 */
class HybridState {
  QubitStateOrTop qState;
  double probability;
  std::vector<bool> bitValues;
  unsigned int maxNumberOfBitValues;

  HybridState() : probability(0.0), maxNumberOfBitValues(0) {}

public:
  explicit HybridState(std::size_t nQubits, std::size_t maxNonzeroAmplitudes,
                       unsigned int maxNumberOfBitValues,
                       std::vector<bool> bitValues = {},
                       double probability = 1.0);

  ~HybridState();

  void print(std::ostream& os) const;

  [[nodiscard("HybridState::toString called but ignored")]]
  std::string toString() const;

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
   * @param params The parameter applied to the gate.
   * @throw std::domain_error If the number of nonzero amplitudes would exceed
   * maxNonzeroAmplitudes.
   */
  void propagateGate(qc::OpType gate, const std::vector<unsigned int>& targets,
                     const std::vector<unsigned int>& posCtrlsQuantum = {},
                     const std::vector<unsigned int>& negCtrlsQuantum = {},
                     const std::vector<unsigned int>& posCtrlsClassical = {},
                     const std::vector<unsigned int>& negCtrlsClassical = {},
                     const std::vector<double>& params = {});

  /**
   * @brief This method adds a classical bit to the hybrid state and returns the
   * bit's index.
   *
   * @param value The value the new bit gets.
   * @throws domain_error If the number of bits would exceed the allowed number
   * of bits.
   * @return The index of the created bit.
   */
  unsigned int addClassicalBit(bool value = false);

  /**
   * @brief This method applies a measurement.
   *
   * This method applies a measurement, changing the qubits and the classical
   * bit corresponding to the measurement.
   *
   * @param quantumTarget The index of the qubit to be measured.
   * @param classicalTarget The index of the bit to save the measurement result
   * in. Has to be a valid classical bit in the hybrid state.
   * @param posCtrlsClassical An array of the indices of the ctrl bits.
   * @param negCtrlsClassical An array of the indices of the negative ctrl bits.
   * @throws domain_error If the quantum state of the hybrid state is TOP.
   * @throw invalid_argument If the bit is not a valid bit of the hybrid state.
   * @return One or two classical states corresponding to the measurement
   * outcomes.
   */
  std::vector<HybridState>
  propagateMeasurement(unsigned int quantumTarget, unsigned int classicalTarget,
                       const std::vector<unsigned int>& posCtrlsClassical = {},
                       const std::vector<unsigned int>& negCtrlsClassical = {});

  /**
   * @brief This method applies a reset.
   *
   * This method applies a reset, changing the qubits and creates one or two new
   * states. The procedure is done as if the qubit was measured, put to zero if
   * the measurement was one, and the result discarded.
   *
   * @param target The index of the qubit to be measured.
   * @param posCtrlsClassical An array of the indices of the ctrl bits.
   * @param negCtrlsClassical An array of the indices of the negative ctrl bits.
   * @throws domain_error If the quantum state of the hybrid state is TOP.
   * @return One or two classical states corresponding to the measurement
   * outcomes during the reset, but with the qubit always in the zero state.
   */
  std::vector<HybridState>
  propagateReset(unsigned int target,
                 const std::vector<unsigned int>& posCtrlsClassical = {},
                 const std::vector<unsigned int>& negCtrlsClassical = {});

  /**
   * @brief This method unifies two HybridStates.
   *
   * This method unifies the current HybridState with the given one and returns
   * a new HybridState, if the new state has no more than maxNumberOfBitValues.
   * Otherwise, throws a domain_error.
   *
   * @param that The HybridState to unify this with.
   * @param qubitsOccupiedByThat Qubit positions that the QubitState of the
   * other HybridState will provide.
   * @param bitsOccupiedByThat Bit positions that the other HybridState will
   * provide.
   * @throw std::domain_error If the number of bits would exceed
   * maxNumberOfBitValues of this.
   */
  HybridState unify(HybridState that,
                    std::vector<unsigned int> qubitsOccupiedByThat,
                    std::vector<unsigned int> bitsOccupiedByThat);

  bool operator==(const HybridState& that) const;

  [[nodiscard("HybridState::isQubitAlwaysOne called but ignored")]] bool
  isQubitAlwaysOne(size_t q) const;

  [[nodiscard("HybridState::isQubitAlwaysZero called but ignored")]] bool
  isQubitAlwaysZero(size_t q) const;

  [[nodiscard("HybridState::isBitAlwaysOne called but ignored")]] bool
  isBitAlwaysOne(size_t q) const;

  [[nodiscard("HybridState::isBitAlwaysZero called but ignored")]] bool
  isBitAlwaysZero(size_t q) const;

  /**
   * @brief Checks if a given combination of bit-qubit values has a nonzero
   * probability
   *
   * This method receives a number of qubit and bit indices and checks whether
   * they have for a given value always a zero amplitude. If the qubit or bit
   * values are top, it is not guaranteed that the amplitude is always zero and
   * false is returned.
   *
   * @param qubits The qubits which are being checked.
   * @param value The value for which is tested whether there is a nonzero
   * amplitude.
   * @param bits The bit indices to check.
   * @param bitValuesToCheck The values of the bits to check. The value at k is
   * the value of bit index at k.
   * @returns True if the amplitude is always zero, false otherwise.
   */
  [[nodiscard("HybridState::hasAlwaysZeroAmplitude called but ignored")]] bool
  hasAlwaysZeroAmplitude(const std::vector<unsigned int>& qubits,
                         unsigned int value,
                         const std::vector<unsigned int>& bits = {},
                         std::vector<bool> bitValuesToCheck = {}) const;

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
};

class HybridStateOrTop {
  std::variant<TOP, std::shared_ptr<HybridState>> variant;

public:
  HybridStateOrTop();

  explicit HybridStateOrTop(TOP top);

  explicit HybridStateOrTop(std::shared_ptr<HybridState> hybridState);

  HybridStateOrTop(const HybridStateOrTop& hybridStateOrTop);

  HybridStateOrTop& operator=(const HybridStateOrTop& hybridStateOrTop);

  HybridStateOrTop& operator=(const std::shared_ptr<HybridState>& hybridState);

  HybridStateOrTop& operator=(const TOP& t);

  bool operator==(const HybridStateOrTop& that) const;

  bool operator!=(const HybridStateOrTop& that) const;

  ~HybridStateOrTop();

  [[nodiscard("HybridStateOrTop::isTop called but ignored")]] bool
  isTop() const;

  [[nodiscard("HybridStateOrTop::isHybridState called but ignored")]] bool
  isHybridState() const;

  [[nodiscard("HybridStateOrTop::getHybridState called but ignored")]] std::
      shared_ptr<HybridState>
      getHybridState() const;

  [[nodiscard("HybridStateOrTop::toString called but ignored")]] std::string
  toString() const;

  void print(std::ostream& os) const;

  [[nodiscard("HybridStateOrTop::isQubitAlwaysOne called but ignored")]] bool
  isQubitAlwaysOne(size_t q) const;

  [[nodiscard("HybridStateOrTop::isQubitAlwaysZero called but ignored")]] bool
  isQubitAlwaysZero(size_t q) const;

  [[nodiscard("HybridStateOrTop::isBitAlwaysOne called but ignored")]] bool
  isBitAlwaysOne(size_t q) const;

  [[nodiscard("HybridStateOrTop::isBitAlwaysZero called but ignored")]] bool
  isBitAlwaysZero(size_t q) const;

  /**
   * @brief Checks if a given combination of bit-qubit values has a nonzero
   * probability
   *
   * This method receives a number of qubit and bit indices and checks whether
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
  [[nodiscard(
      "HybridStateOrTop::hasAlwaysZeroAmplitude called but ignored")]] bool
  hasAlwaysZeroAmplitude(const std::vector<unsigned int>& qubits,
                         unsigned int value,
                         const std::vector<unsigned int>& bits = {},
                         const std::vector<bool>& bitValues = {}) const;
};
} // namespace mqt::ir::opt::qcp

#endif // MQT_CORE_HYBRIDSTATE_H