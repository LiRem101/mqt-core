/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/Transforms/ConstantPropagation/UnionTable.hpp"

namespace mqt::ir::opt::qcp {
UnionTable::UnionTable() {
  this->nQubits = 0;
  this->nBits = 0;
  this->hReg = {};
}

UnionTable::~UnionTable() {}

void UnionTable::unify(std::vector<unsigned int> qubits,
                       std::vector<unsigned int> bits) {
  throw std::logic_error("Not implemented");
}

bool UnionTable::allTop() {
  for (size_t i = 0; i < nQubits; i++) {
    if (!hReg[i].isTop()) {
      return false;
    }
  }
  return true;
}

void UnionTable::propagateGate(qc::OpType gate,
                               std::vector<unsigned int> targets,
                               std::vector<unsigned int> posCtrls,
                               std::vector<unsigned int> negCtrls,
                               std::vector<double> params) {
  throw std::logic_error("Not implemented");
}

void UnionTable::propagateMeasurement(unsigned int quantumTarget,
                                      unsigned int classicalTarget) {
  throw std::logic_error("Not implemented");
}

void UnionTable::propagateReset(unsigned int target) {
  throw std::logic_error("Not implemented");
}

unsigned int UnionTable::propagateQubitAlloc() {
  throw std::logic_error("Not implemented");
}

void UnionTable::propagateQubitDealloc(unsigned int target) {
  throw std::logic_error("Not implemented");
}

bool UnionTable::isQubitAlwaysOne(size_t q) {
  throw std::logic_error("Not implemented");
}

bool UnionTable::isQubitAlwaysZero(size_t q) {
  throw std::logic_error("Not implemented");
}

bool UnionTable::isBitAlwaysOne(size_t q) {
  throw std::logic_error("Not implemented");
}

bool UnionTable::isBitAlwaysZero(size_t q) {
  throw std::logic_error("Not implemented");
}

bool UnionTable::hasNonzeroAmplitude(std::vector<unsigned int> qubits,
                                     unsigned int value) {
  throw std::logic_error("Not implemented");
}
} // namespace mqt::ir::opt::qcp