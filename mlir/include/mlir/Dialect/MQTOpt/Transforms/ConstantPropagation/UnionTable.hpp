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

#include <cstddef>
#include <string>
#include <vector>
class HybridStateOrTop;
/**
 * @brief This class holds the hybrid states of multiple qubits and bits.
 *
 * This class holds the hybrid states of multiple qubits and bits. It can unify
 * the states. One can propagate gates, measurements and resets through the
 * states.
 */
class UnionTable {
public:
  UnionTable(size_t nQubits, size_t nBits);

  ~UnionTable();

  ;
  void unify(std::vector<unsigned int> qubits, std::vector<unsigned int> bits);

  bool allTop();

  void propagateGate(std::string gate, unsigned int targets[],
                     unsigned int posCtrls[], unsigned int negCtrls[]);

  void propagateMeasurement(unsigned int quantumTarget,
                            unsigned int classicalTarget);

  void propagateReset(unsigned int target);

  bool isQubitAlwaysOne(size_t q);

  bool isQubitAlwaysZero(size_t q);

  bool isBitAlwaysOne(size_t q);

  bool isBitAlwaysZero(size_t q);

  bool hasNonzeroAmplitude(std::vector<unsigned int> qubits,
                           unsigned int value);

private:
  std::size_t nQubits;
  std::size_t nBits;
  HybridStateOrTop* hReg;
};

#endif // MQT_CORE_UNIONTABLE_H
