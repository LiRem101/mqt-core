/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/MQTOptToMQTRef/MQTOptToMQTRef.h" // IWYU pragma: keep
#include "mlir/Conversion/MQTRefToMQTOpt/MQTRefToMQTOpt.h" // IWYU pragma: keep
#include "mlir/Conversion/MQTRefToQIR/MQTRefToQIR.h"       // IWYU pragma: keep
#include "mlir/Conversion/QIRToMQTRef/QIRToMQTRef.h"       // IWYU pragma: keep
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"          // IWYU pragma: keep
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"         // IWYU pragma: keep
#include "mlir/Dialect/MQTRef/IR/MQTRefDialect.h"          // IWYU pragma: keep

#include <filesystem>
#include <fstream>
#include <iostream>
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

  mlir::MlirOptMainConfig config =
      mlir::MlirOptMainConfig::createFromCLOptions();

  std::string toolName = "Quantum optimizer driver\n";

  std::string inputFilename, outputFilename;
  std::tie(inputFilename, outputFilename) =
      registerAndParseCLIOptions(argc, argv, toolName, registry);

  int ac = argc;
  llvm::InitLLVM y(ac, argv);

  std::filesystem::path inputRoot = "Benchmarks/MLIRCollection";
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(inputRoot)) {
    if (entry.is_regular_file()) {
      std::filesystem::path relative =
          std::filesystem::relative(entry.path(), inputRoot);
      std::ifstream in(entry.path());
      std::string content((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
      llvm::raw_ostream& ostr = llvm::outs();
      std::unique_ptr<llvm::MemoryBuffer> buffer =
          llvm::MemoryBuffer::getMemBuffer(content);

      auto res = mlir::MlirOptMain(ostr, std::move(buffer), registry, config);

      if (res.failed()) {
        std::cerr << "Error while processing file " << entry.path()
                  << std::endl;
        return -1;
      } else {
        std::cout << "Processed: " << entry.path() << std::endl;
      }
    }
  }
}
