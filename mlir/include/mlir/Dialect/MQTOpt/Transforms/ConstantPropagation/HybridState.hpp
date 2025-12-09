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

#include <llvm/ADT/STLExtras.h>
#include <memory>
#include <variant>

namespace mqt::ir::opt::qcp {
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
                       double probability);

  ~HybridState();

  void print(std::ostream& os) const;

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
   */
  void propagateGate(std::string gate, unsigned int targets[],
                     unsigned int posCtrlsQuantum[],
                     unsigned int negCtrlsQuantum[],
                     unsigned int posCtrlsClassical[],
                     unsigned int negCtrlsClassical[]);

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
  void resetQubit(unsigned int target);

  bool operator==(const HybridState& that) const;
};

class HybridStateOrTop {
  std::variant<TOP, std::shared_ptr<HybridState>> variant;

public:
  HybridStateOrTop();

  HybridStateOrTop(TOP top);

  HybridStateOrTop(std::shared_ptr<HybridState> hybridState);

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
};
} // namespace mqt::ir::opt::qcp

#endif // MQT_CORE_HYBRIDSTATE_H