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

#include <filesystem>
#include <fstream>
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
  std::ifstream file("/home/lian/DLR/mqt-core/mqt-core/ErrorQasm");
  std::set<std::string> errorLines; // or std::unordered_set<std::string>
  std::string line;
  while (std::getline(file, line)) {
    line.erase(line.size() - 1);
    line.erase(0, 1);
    errorLines.insert(line);
  }

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

  std::string errorFile = "ErrorQasm";
  std::filesystem::path inputRoot = "MQTBench";
  std::filesystem::path outputRoot = "MLIRCollection/MQTBench";
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(inputRoot)) {
    if (entry.is_regular_file()) {
      std::filesystem::path relative =
          std::filesystem::relative(entry.path(), inputRoot);
      std::filesystem::path outputFile = outputRoot / relative;
      std::string outputFileString = outputFile.string();
      std::size_t pos = outputFileString.find(".qasm");
      if (pos != std::string::npos) {
        outputFileString.replace(pos, 5, ".mlir"); // 5 = length of ".qasm"
      }

      if (errorLines.contains(entry.path()) ||
          std::filesystem::exists(outputFileString)) {
        std::cout << "Skipping " << entry.path() << std::endl;
        continue;
      }
      std::filesystem::create_directories(outputFile.parent_path());
      std::ifstream in(entry.path());
      std::string content((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());

      qc::QuantumComputation qc;
      try {
        qc = qasm3::Importer::importf(entry.path());
      } catch (const std::exception& e) {
        std::cerr << "Error while processing file " << entry.path()
                  << std::endl;
        std::ofstream outFile(errorFile, std::ios::app);
        if (outFile.is_open()) {
          outFile << entry.path() << std::endl;
          outFile.close();
        } else {
          std::cerr << "Unable to open error file";
        }
        continue;
      }

      auto module = translateQuantumComputationToMLIR(context.get(), qc);

      const auto mqtRefString = getOutputString(&module);

      std::unique_ptr<llvm::MemoryBuffer> buffer =
          llvm::MemoryBuffer::getMemBuffer(mqtRefString);

      std::error_code ec;
      llvm::raw_fd_ostream file(outputFileString, ec);

      auto res = mlir::MlirOptMain(file, std::move(buffer), registry, config);

      if (res.failed()) {
        std::cerr << "Error while processing file " << entry.path()
                  << std::endl;
        std::ofstream outFile(errorFile, std::ios::app);
        if (outFile.is_open()) {
          outFile << entry.path() << std::endl;
          outFile.close();
        } else {
          std::cerr << "Unable to open error file";
        }
      } else {
        std::cout << "Processed: " << entry.path() << std::endl;
      }
    }
  }
}
