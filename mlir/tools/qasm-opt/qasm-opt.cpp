/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/QuantumComputation.hpp"
#include "mlir/Conversion/MQTOptToMQTRef/MQTOptToMQTRef.h" // IWYU pragma: keep
#include "mlir/Conversion/MQTRefToMQTOpt/MQTRefToMQTOpt.h" // IWYU pragma: keep
#include "mlir/Conversion/MQTRefToQIR/MQTRefToQIR.h"       // IWYU pragma: keep
#include "mlir/Conversion/QIRToMQTRef/QIRToMQTRef.h"       // IWYU pragma: keep
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"          // IWYU pragma: keep
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"         // IWYU pragma: keep
#include "mlir/Dialect/MQTRef/IR/MQTRefDialect.h"          // IWYU pragma: keep
#include "mlir/Dialect/MQTRef/Translation/ImportQuantumComputation.h"
#include "qasm3/Importer.hpp"

#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>

// Suppress warnings about implicit captures of `this` in lambdas
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-this-capture"
#endif

#include <mlir/Dialect/Func/Extensions/AllExtensions.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#ifdef __clang__
#pragma clang diagnostic pop
#endif

std::string getOutputString(mlir::OwningOpRef<mlir::ModuleOp>* module) {
  std::string outputString;
  llvm::raw_string_ostream os(outputString);
  (*module)->print(os);
  os.flush();
  return outputString;
}

int main(const int argc, char** argv) {
  mlir::registerAllPasses();
  mqt::ir::opt::registerMQTOptPasses();
  mqt::ir::registerMQTRefToMQTOptPasses();
  mqt::ir::registerMQTOptToMQTRefPasses();
  mqt::ir::registerQIRToMQTRefPasses();
  mqt::ir::registerMQTRefToQIRPass();

  std::unique_ptr<mlir::MLIRContext> context;

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  registry.insert<mqt::ir::opt::MQTOptDialect>();
  registry.insert<mqt::ir::ref::MQTRefDialect>();

  context = std::make_unique<mlir::MLIRContext>();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();

  qc::QuantumComputation qc = qasm3::Importer::importf("simple.qasm");
  auto module = translateQuantumComputationToMLIR(context.get(), qc);

  const auto mqtRefString = getOutputString(&module);

  mlir::MlirOptMainConfig config =
      mlir::MlirOptMainConfig::createFromCLOptions();

  std::string toolName = "Quantum optimizer driver\n";
  std::string outputFile = "-";

  std::string inputFilename, outputFilename;
  std::tie(inputFilename, outputFilename) =
      registerAndParseCLIOptions(argc, argv, toolName, registry);

  int ac = argc;
  llvm::InitLLVM y(ac, argv);

  std::unique_ptr<llvm::MemoryBuffer> buffer =
      llvm::MemoryBuffer::getMemBuffer(mqtRefString);

  llvm::raw_ostream& ostr = llvm::outs();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(ostr, std::move(buffer), registry, config));
}
