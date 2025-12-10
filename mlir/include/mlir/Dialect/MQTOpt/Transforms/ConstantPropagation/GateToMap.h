//
// Created by lian on 2025-12-09.
//

#ifndef MQT_CORE_GATETOMAP_H
#define MQT_CORE_GATETOMAP_H
#include "ir/operations/OpType.hpp"

#include <complex>
#include <unordered_map>

namespace mqt::ir::opt::qcp {
inline std::unordered_map<
    unsigned int, std::unordered_map<unsigned int, std::complex<double>>>
getQubitMappingOfGates(qc::OpType gate) {
  switch (gate) {
  case (qc::I):
    return {{0, {{0, std::complex<double>(1, 0)}}},
            {1, {{1, std::complex<double>(1, 0)}}}};
  case (qc::H):
    return {{0,
             {{0, std::complex<double>(1 / sqrt(2), 0)},
              {1, std::complex<double>(1 / sqrt(2), 0)}}},
            {1,
             {{0, std::complex<double>(1 / sqrt(2), 0)},
              {1, std::complex<double>(-1 / sqrt(2), 0)}}}};
  case (qc::X):
    return {{0, {{1, std::complex<double>(1, 0)}}},
            {1, {{0, std::complex<double>(1, 0)}}}};
  case (qc::Y):
    return {{0, {{1, std::complex<double>(0, -1)}}},
            {1, {{0, std::complex<double>(0, 1)}}}};
  case (qc::Z):
    return {{0, {{0, std::complex<double>(1, 0)}}},
            {1, {{1, std::complex<double>(-1, 0)}}}};
  case (qc::S):
    return {{0, {{0, std::complex<double>(1, 0)}}},
            {1, {{1, std::complex<double>(0, 1)}}}};
  case (qc::Sdg):
    return {{0, {{0, std::complex<double>(1, 0)}}},
            {1, {{1, std::complex<double>(0, -1)}}}};
  case (qc::T):
    return {{0, {{0, std::complex<double>(1, 0)}}},
            {1, {{1, std::complex<double>(1 / sqrt(2), 1 / sqrt(2))}}}};
  case (qc::Tdg):
    return {{0, {{0, std::complex<double>(1, 0)}}},
            {1, {{1, std::complex<double>(1 / sqrt(2), -1 / sqrt(2))}}}};
  case (qc::V):
    return {{0,
             {{0, std::complex<double>(1 / sqrt(2), 0)},
              {1, std::complex<double>(0, -1 / sqrt(2))}}},
            {1,
             {{0, std::complex<double>(0, -1 / sqrt(2))},
              {1, std::complex<double>(-1 / sqrt(2), 0)}}}};
  case (qc::Vdg):
    return {{0,
             {{0, std::complex<double>(1 / sqrt(2), 0)},
              {1, std::complex<double>(0, 1 / sqrt(2))}}},
            {1,
             {{0, std::complex<double>(0, -1 / sqrt(2))},
              {1, std::complex<double>(-1 / sqrt(2), 0)}}}};
  case (qc::SX):
    return {{0,
             {{0, std::complex<double>(1 / 2, 1 / 2)},
              {1, std::complex<double>(1 / 2, -1 / 2)}}},
            {1,
             {{0, std::complex<double>(1 / 2, -1 / 2)},
              {1, std::complex<double>(1 / 2, 1 / 2)}}}};
  case (qc::SXdg):
    return {{0,
             {{0, std::complex<double>(1 / 2, -1 / 2)},
              {1, std::complex<double>(1 / 2, 1 / 2)}}},
            {1,
             {{0, std::complex<double>(1 / 2, 1 / 2)},
              {1, std::complex<double>(1 / 2, -1 / 2)}}}};
  default:
    throw std::runtime_error(
        "Unsupported gate in mqt::ir::opt::qcp::getQubitMappingOfGates");
  }
}
} // namespace mqt::ir::opt::qcp

#endif // MQT_CORE_GATETOMAP_H
