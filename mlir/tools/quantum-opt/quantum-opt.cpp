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
#include <iostream>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <mlir/IR/AsmState.h>

// Suppress warnings about implicit captures of `this` in lambdas
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-this-capture"
#endif

#include <fstream>
#include <mlir/Dialect/Func/Extensions/AllExtensions.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#ifdef __clang__
#pragma clang diagnostic pop
#endif

std::string slurp(std::ifstream& in) {
  std::ostringstream sstr;
  sstr << in.rdbuf();
  return sstr.str();
}

int main(const int argc, char** argv) {
  std::ifstream file("ErrorMQTBenchMeasLiftHadamardLift");
  std::set<std::string> errorLines; // or std::unordered_set<std::string>
  std::string line;
  while (std::getline(file, line)) {
    line.erase(line.size() - 1);
    line.erase(0, 1);
    errorLines.insert(line);
  }

  int ac = argc;
  llvm::InitLLVM y(ac, argv);

  mlir::registerAllPasses();
  mqt::ir::opt::registerMQTOptPasses();
  mqt::ir::registerMQTRefToMQTOptPasses();
  mqt::ir::registerMQTOptToMQTRefPasses();
  mqt::ir::registerQIRToMQTRefPasses();
  mqt::ir::registerMQTRefToQIRPass();

  std::unique_ptr<mlir::MLIRContext> context;

  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  registry.insert<mqt::ir::opt::MQTOptDialect>();
  registry.insert<mqt::ir::ref::MQTRefDialect>();

  std::string toolName = "Quantum optimizer driver\n";

  std::string inputFilename, outputFilename;
  std::tie(inputFilename, outputFilename) =
      registerAndParseCLIOptions(argc, argv, toolName, registry);

  context = std::make_unique<mlir::MLIRContext>();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();

  mlir::MlirOptMainConfig config =
      mlir::MlirOptMainConfig::createFromCLOptions();

  std::string errorFile = "ErrorQCPMQTBench";
  std::ofstream timeOut("EvalQCP.csv", std::ios::app);
  std::filesystem::path inputRoot =
      "/home/lian/DLR/Benchmarks/MLIRCollection/MQTBench";
  std::filesystem::path outputRoot =
      "/home/lian/DLR/Benchmarks/MQTBenchAllsQCP";

  timeOut << "Filename;QCPDuration[ms];LiftMeasurementsAboveControls;"
             "LiftMeasurementsAbovePhaseGates;"
             "AdaptCtrldPauliZToLifting;"
             "LiftHadamardsAbovePauliGates;LiftHadamardAboveCNOT;"
             "LiftHadamardsAboveControlledPauliGates;"
             "RemoveClassicalCondValueIsTrueFalse;RemoveGatesNeverExecutable;"
             "RemoveDiagonalGates;RemoveSuperfluousQCtrls;RemoveImpliedQCtrls;"
             "ExchangeQCtrlsToCCtrls"
          << std::endl;

  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(inputRoot)) {
    if (entry.is_regular_file()) {
      std::filesystem::path relative =
          std::filesystem::relative(entry.path(), inputRoot);
      std::filesystem::path outputFile = outputRoot / relative;
      std::string outputFileString = outputFile.string();

      if (errorLines.contains(entry.path()) ||
          std::filesystem::exists(outputFileString) ||
          entry.path().extension() != ".mlir") {
        std::cout << "Skipping " << entry.path() << std::endl;
        continue;
      }
      std::cout << "Processing: " << entry.path() << std::endl;
      std::filesystem::create_directories(outputFile.parent_path());
      std::ifstream in(entry.path());
      std::string content((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());

      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufferOrErr =
          llvm::MemoryBuffer::getFile(entry.path().string());
      if (!bufferOrErr) {
        std::cerr << "Error while reading file " << entry.path() << std::endl;
        continue;
      }
      std::unique_ptr<llvm::MemoryBuffer> mlirInput = std::move(*bufferOrErr);

      std::error_code ec;
      llvm::raw_fd_ostream outStream(outputFileString, ec);

      auto start = std::chrono::high_resolution_clock::now();

      std::ostringstream oss;
      std::streambuf* old = std::cout.rdbuf(oss.rdbuf());

      auto res =
          mlir::MlirOptMain(outStream, std::move(mlirInput), registry, config);

      auto end = std::chrono::high_resolution_clock::now();

      std::cout.rdbuf(old); // restore
      auto appliedTrafos = oss.str();

      auto duration = duration_cast<std::chrono::milliseconds>(end - start);

      int numberAdapatCtrldPauliZToLifting = static_cast<unsigned int>(
          std::count(appliedTrafos.begin(), appliedTrafos.end(), 'A'));
      int numberLiftHadamardsAbovePauliGates = static_cast<unsigned int>(
          std::count(appliedTrafos.begin(), appliedTrafos.end(), 'B'));
      int numberLiftHadamardsAboveCNOT = static_cast<unsigned int>(
          std::count(appliedTrafos.begin(), appliedTrafos.end(), 'C'));
      int numberLiftMeasurementsAbovePhaseGates = static_cast<unsigned int>(
          std::count(appliedTrafos.begin(), appliedTrafos.end(), 'D'));
      int numberLiftMeasurementsAboveControls = static_cast<unsigned int>(
          std::count(appliedTrafos.begin(), appliedTrafos.end(), 'E'));
      int numberLiftHadamardsAboveControlledPauliGates =
          static_cast<unsigned int>(
              std::count(appliedTrafos.begin(), appliedTrafos.end(), 'F'));
      int numberRemoveClassicalCondValueIsTrueFalse = static_cast<unsigned int>(
          std::count(appliedTrafos.begin(), appliedTrafos.end(), 'G'));
      int numberRemoveGatesNeverExecutable = static_cast<unsigned int>(
          std::count(appliedTrafos.begin(), appliedTrafos.end(), 'H'));
      int numberRemoveDiagonalGates = static_cast<unsigned int>(
          std::count(appliedTrafos.begin(), appliedTrafos.end(), 'I'));
      int numberRemoveSuperfluousQCtrls = static_cast<unsigned int>(
          std::count(appliedTrafos.begin(), appliedTrafos.end(), 'J'));
      int numberRemoveImpliedQCtrls = static_cast<unsigned int>(
          std::count(appliedTrafos.begin(), appliedTrafos.end(), 'K'));
      int numberExchangeQCtrlsToCCtrls = static_cast<unsigned int>(
          std::count(appliedTrafos.begin(), appliedTrafos.end(), 'L'));

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
        std::filesystem::path relativeInput =
            std::filesystem::relative(entry.path(), inputRoot);
        timeOut << relativeInput << ";" << duration.count() << ";"
                << numberLiftMeasurementsAboveControls << ";"
                << numberLiftMeasurementsAbovePhaseGates << ";"
                << numberAdapatCtrldPauliZToLifting << ";"
                << numberLiftHadamardsAbovePauliGates << ";"
                << numberLiftHadamardsAboveCNOT << ";"
                << numberLiftHadamardsAboveControlledPauliGates << ";"
                << numberRemoveClassicalCondValueIsTrueFalse << ";"
                << numberRemoveGatesNeverExecutable << ";"
                << numberRemoveDiagonalGates << ";"
                << numberRemoveSuperfluousQCtrls << ";"
                << numberRemoveImpliedQCtrls << ";"
                << numberExchangeQCtrlsToCCtrls << std::endl;
        std::cout << "Processed: " << entry.path() << std::endl;
      }
    }
  }
}