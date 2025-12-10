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
  case (qc::H):
    return {{0,
             {{0, std::complex<double>(0.70, 0)},
              {1, std::complex<double>(0.70, 0)}}},
            {1,
             {{0, std::complex<double>(0.70, 0)},
              {1, std::complex<double>(-0.70, 0)}}}};
  default:
    throw std::runtime_error(
        "Unsupported gate in mqt::ir::opt::qcp::getQubitMappingOfGates");
  }
}
} // namespace mqt::ir::opt::qcp

#endif // MQT_CORE_GATETOMAP_H
