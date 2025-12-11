//
// Created by lian on 2025-12-09.
//

#ifndef MQT_CORE_GATETOMAP_H
#define MQT_CORE_GATETOMAP_H
#include "ir/operations/OpType.hpp"

#include <complex>
#include <unordered_map>
#include <vector>

namespace mqt::ir::opt::qcp {
inline std::unordered_map<
    unsigned int, std::unordered_map<unsigned int, std::complex<double>>>
getQubitMappingOfGates(qc::OpType gate, std::vector<double> params) {
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
    return {{0, {{1, std::complex<double>(0, 1)}}},
            {1, {{0, std::complex<double>(0, -1)}}}};
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
  case (qc::U):
    return {
        {0,
         {{0, std::complex<double>(cos(params[0] / 2), 0)},
          {1, exp(std::complex<double>(0, params[1])) * sin(params[0] / 2)}}},
        {1,
         {{0, -exp(std::complex<double>(0, params[2])) * sin(params[0] / 2)},
          {1, exp(std::complex<double>(0, params[1] + params[2])) *
                  cos(params[0] / 2)}}}};
  case (qc::U2):
    return {{0,
             {{0, std::complex<double>(1 / sqrt(2), 0)},
              {1, exp(std::complex<double>(0, params[0]))}}},
            {1,
             {{0, -exp(std::complex<double>(0, params[1]))},
              {1, exp(std::complex<double>(0, params[0] + params[1]))}}}};
  case (qc::P):
    return {{0, {{0, std::complex<double>(1, 0)}}},
            {1, {{1, exp(std::complex<double>(0, params[0]))}}}};
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
  case (qc::RX):
    return {{0,
             {{0, std::complex<double>(cos(params[0] / 2), 0)},
              {1, std::complex<double>(0, -sin(params[0] / 2))}}},
            {1,
             {{0, std::complex<double>(0, -sin(params[0] / 2))},
              {1, std::complex<double>(cos(params[0] / 2), 0)}}}};
  case (qc::RY):
    return {{0,
             {{0, std::complex<double>(cos(params[0] / 2), 0)},
              {1, std::complex<double>(sin(params[0] / 2), 0)}}},
            {1,
             {{0, std::complex<double>(-sin(params[0] / 2), 0)},
              {1, std::complex<double>(cos(params[0] / 2), 0)}}}};
  case (qc::RZ):
    return {{0, {{0, exp(std::complex<double>(0, -params[0] / 2))}}},
            {1, {{1, exp(std::complex<double>(0, params[0] / 2))}}}};
  case (qc::R):
    return {{0,
             {{0, std::complex<double>(cos(params[0] / 2), 0)},
              {1, exp(std::complex<double>(0, params[1])) *
                      std::complex<double>(0, -sin(params[0] / 2))}}},
            {1,
             {{0, exp(std::complex<double>(0, -params[1])) *
                      std::complex<double>(0, -sin(params[0] / 2))},
              {1, std::complex<double>(cos(params[0] / 2), 0)}}}};
  case (qc::SWAP): // If target qubits are in one QubitState, we can simply
                   // apply the matrix representation
    return {{0, {{0, std::complex<double>(1, 0)}}},
            {1, {{2, std::complex<double>(1, 0)}}},
            {2, {{1, std::complex<double>(1, 0)}}},
            {3, {{3, std::complex<double>(1, 0)}}}};
  case (qc::iSWAP):
    return {{0, {{0, std::complex<double>(1, 0)}}},
            {1, {{2, std::complex<double>(0, -1)}}},
            {2, {{1, std::complex<double>(0, -1)}}},
            {3, {{3, std::complex<double>(1, 0)}}}};
  case (qc::iSWAPdg):
    return {{0, {{0, std::complex<double>(1, 0)}}},
            {1, {{2, std::complex<double>(0, 1)}}},
            {2, {{1, std::complex<double>(0, 1)}}},
            {3, {{3, std::complex<double>(1, 0)}}}};
  case (qc::Peres):
    return {{0, {{2, std::complex<double>(1, 0)}}},
            {1, {{3, std::complex<double>(1, 0)}}},
            {2, {{1, std::complex<double>(1, 0)}}},
            {3, {{0, std::complex<double>(1, 0)}}}};
  case (qc::Peresdg):
    return {{0, {{3, std::complex<double>(1, 0)}}},
            {1, {{2, std::complex<double>(1, 0)}}},
            {2, {{0, std::complex<double>(1, 0)}}},
            {3, {{1, std::complex<double>(1, 0)}}}};
  case (qc::DCX):
    return {{0, {{0, std::complex<double>(1, 0)}}},
            {1, {{2, std::complex<double>(1, 0)}}},
            {2, {{3, std::complex<double>(1, 0)}}},
            {3, {{1, std::complex<double>(1, 0)}}}};
  case (qc::ECR):
    return {{0,
             {{2, std::complex<double>(1 / sqrt(2), 0)},
              {3, std::complex<double>(0, -1 / sqrt(2))}}},
            {1,
             {{2, std::complex<double>(0, -1 / sqrt(2))},
              {3, std::complex<double>(1 / sqrt(2), 0)}}},
            {2,
             {{0, std::complex<double>(1 / sqrt(2), 0)},
              {1, std::complex<double>(0, 1 / sqrt(2))}}},
            {3,
             {{0, std::complex<double>(0, 1 / sqrt(2))},
              {1, std::complex<double>(1 / sqrt(2), 0)}}}};
  case (qc::RXX):
    return {{0,
             {{0, std::complex<double>(cos(params[0] / 2), 0)},
              {3, std::complex<double>(0, -sin(params[0] / 2))}}},
            {1,
             {{1, std::complex<double>(cos(params[0] / 2), 0)},
              {2, std::complex<double>(0, -sin(params[0] / 2))}}},
            {2,
             {{1, std::complex<double>(0, -sin(params[0] / 2))},
              {2, std::complex<double>(cos(params[0] / 2), 0)}}},
            {3,
             {{0, std::complex<double>(0, -sin(params[0] / 2))},
              {3, std::complex<double>(cos(params[0] / 2), 0)}}}};
  case (qc::RYY):
    return {{0,
             {{0, std::complex<double>(cos(params[0] / 2), 0)},
              {3, std::complex<double>(0, sin(params[0] / 2))}}},
            {1,
             {{1, std::complex<double>(cos(params[0] / 2), 0)},
              {2, std::complex<double>(0, -sin(params[0] / 2))}}},
            {2,
             {{1, std::complex<double>(0, -sin(params[0] / 2))},
              {2, std::complex<double>(cos(params[0] / 2), 0)}}},
            {3,
             {{0, std::complex<double>(0, sin(params[0] / 2))},
              {3, std::complex<double>(cos(params[0] / 2), 0)}}}};
  case (qc::RZZ):
    return {{0, {{0, exp(std::complex<double>(0, -params[0] / 2))}}},
            {1, {{1, exp(std::complex<double>(0, params[0] / 2))}}},
            {2, {{2, exp(std::complex<double>(0, params[0] / 2))}}},
            {3, {{3, exp(std::complex<double>(0, -params[0] / 2))}}}};
  case (qc::RZX):
    return {{0,
             {{0, std::complex<double>(cos(params[0] / 2), 0)},
              {1, std::complex<double>(0, -sin(params[0] / 2))}}},
            {1,
             {{0, std::complex<double>(0, -sin(params[0] / 2))},
              {1, std::complex<double>(cos(params[0] / 2), 0)}}},
            {2,
             {{2, std::complex<double>(cos(params[0] / 2), 0)},
              {3, std::complex<double>(0, sin(params[0] / 2))}}},
            {3,
             {{2, std::complex<double>(0, sin(params[0] / 2))},
              {3, std::complex<double>(cos(params[0] / 2), 0)}}}};
  case (qc::XXminusYY):
    return {{0,
             {{0, std::complex<double>(cos(params[0] / 2), 0)},
              {3, std::complex<double>(0, -sin(params[0] / 2)) *
                      exp(std::complex<double>(0, params[1]))}}},
            {1, {{1, std::complex<double>(1, 0)}}},
            {2, {{2, std::complex<double>(1, 0)}}},
            {3,
             {{0, std::complex<double>(0, -sin(params[0] / 2)) *
                      exp(std::complex<double>(0, -params[1]))},
              {3, std::complex<double>(cos(params[0] / 2), 0)}}}};
  case (qc::XXplusYY):
    return {{0, {{0, std::complex<double>(1, 0)}}},
            {1,
             {{1, std::complex<double>(cos(params[0] / 2), 0)},
              {2, std::complex<double>(0, -sin(params[0] / 2)) *
                      exp(std::complex<double>(0, params[1]))}}},
            {2,
             {{1, std::complex<double>(0, -sin(params[0] / 2)) *
                      exp(std::complex<double>(0, -params[1]))},
              {2, std::complex<double>(cos(params[0] / 2), 0)}}},
            {3, {{3, std::complex<double>(1, 0)}}}};
  default:
    throw std::runtime_error(
        "Unsupported gate in mqt::ir::opt::qcp::getQubitMappingOfGates");
  }
}
} // namespace mqt::ir::opt::qcp

#endif // MQT_CORE_GATETOMAP_H
