/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_CORE_REWRITECHECKER_H
#define MQT_CORE_REWRITECHECKER_H

#include "UnionTable.hpp"

#include <optional>

namespace mqt::ir::opt::qcp {
/**
 * @brief This class holds a UnionTable and does checks on the rewrite
 * properties.
 */
class RewriteChecker {
  UnionTable unionTable;

public:
  explicit RewriteChecker(UnionTable ut);

  ~RewriteChecker();

  /**
   * @brief This method checks which qubits and bits are superfluous given a
   * controlled gate.
   *
   * This method checks which qubits and bits are superfluous given a
   * controlled gate. If the gate can never be executed, the target qubits are
   * superfluous. Apart from that, all posCtrl (negCtrl) qubits/bits that are
   * always positive (negative) are superfluous.
   *
   * @param qubitTargets The indices of the target qubits.
   * @param qubitPosCtrls The indices of the positively controlling qubits.
   * @param qubitNegCtrls The indices of the negatively controlling qubits.
   * @param bitPosCtrls The indices of the positively controlling bits.
   * @param bitNegCtrls The indices of the negatively controlling bits.
   * @returns A pair of superfluous qubits and superfluous bits.
   */
  std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
  getSuperfluousControls(std::vector<unsigned int> qubitTargets,
                         std::vector<unsigned int> qubitPosCtrls,
                         std::vector<unsigned int> qubitNegCtrls,
                         std::vector<unsigned int> bitPosCtrls,
                         std::vector<unsigned int> bitNegCtrls);

  /**
   * @brief Returns an equivalent bit if that exists.
   *
   * This method checks whether there is bit equivalent to a given qubit (i.e.
   * always has the same values as the qubit). If that is the case, the bit is
   * returned. If multiple bits are equivalent to the qubit, one of them is
   * returned randomly.
   *
   * @param q The index of the qubit for which an equivalent bit is searched
   * for.
   * @returns The index of an equivalent bit, if there is one.
   */
  std::optional<unsigned int> getEquivalentBit(unsigned int q);

  /**
   * @brief Returns the qubits and bits that imply the given qubit.
   *
   * This method checks whether in the given vectors are qubits or bit that
   * imply (are antecedents of) the given qubit. I.e. all qubits and bits a are
   * returned for which holds: a -> q.
   *
   * @param q The qubit for which is checked whether it is implied.
   * @param qubits The qubits for which are checked if they imply q.
   * @param bits The bits for which are checked if they imply q.
   * @returns A pair of 1. qubits and 2. bits that are antecedents of q.
   */
  std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
  getAntecedentsOfQubit(unsigned int q, std::vector<unsigned int> qubits,
                        std::vector<unsigned int> bits);

  /**
   * @brief Returns the qubits and bits that imply the given bit.
   *
   * This method checks whether in the given vectors are qubits or bit that
   * imply (are antecedents of) the given bit. I.e. all qubits and bits a are
   * returned for which holds: a -> b.
   *
   * @param b The qubit for which is checked whether it is implied.
   * @param qubits The qubits for which are checked if they imply b.
   * @param bits The bits for which are checked if they imply b.
   * @returns A pair of 1. qubits and 2. bits that are antecedents of b.
   */
  std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
  getAntecedentsOfBit(unsigned int b, std::vector<unsigned int> qubits,
                      std::vector<unsigned int> bits);

  /**
   * @brief Returns true if only the values of one given set of teh qubits is
   * nonzero.
   *
   * This method receives a number of qubit indices and vectors of possible
   * amplitudes of the qubits. It checks whether more than one of the value sets
   * corrsponds to nonzero amplitudes.
   *
   * @param qubits The qubits which are being checked.
   * @param values The sets of values for which is tested whether there is a
   * nonzero amplitude.
   * @returns True if at maximum the amplitudes in one set are not equal to
   * zero.
   */
  bool isOnlyOneSetNotZero(std::vector<unsigned int> qubits,
                           std::vector<std::vector<unsigned int>> values);
};
} // namespace mqt::ir::opt::qcp
#endif // MQT_CORE_REWRITECHECKER_H
