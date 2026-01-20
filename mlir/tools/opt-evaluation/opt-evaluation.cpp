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
#include <llvm/Support/SourceMgr.h>
#include <mlir/Dialect/Func/Extensions/AllExtensions.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

int main(int argc, char** argv) {
  mlir::registerAllPasses();
  mqt::ir::opt::registerMQTOptPasses();

  std::unique_ptr<mlir::MLIRContext> context;

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  registry.insert<mqt::ir::opt::MQTOptDialect>();
  registry.insert<mqt::ir::ref::MQTRefDialect>();

  context = std::make_unique<mlir::MLIRContext>();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();
  context->allowUnregisteredDialects();

  std::string errorFile =
      "/home/lian/DLR/mqt-core/mqt-core/LiftEvaluationError";
  std::filesystem::path originalInput = "/home/lian/DLR/Benchmarks/programs";
  std::filesystem::path measurementLiftInput =
      "/home/lian/DLR/Benchmarks/programsMeasurementLift";
  std::filesystem::path hadamardMeasurementLiftInput =
      "/home/lian/DLR/Benchmarks/programsHadamardMeasurementLift";
  std::ofstream out("/home/lian/DLR/mqt-core/mqt-core/EvaluationOfLifting.csv",
                    std::ios::app);

  out << "Filename;OriginalCountOfGates;OriginalCountOfUncontrolledGates;"
         "OriginalCountOfControllingQubits;AfterMeasLiftingCountOfGates;"
         "AfterMeasLiftingCountOfUncontrolledGates;"
         "AfterMeasLiftingCountOfControllingQubits;"
         "AfterHadamardAndMeasLiftingCountOfGates;"
         "AfterHadamardAndMeasLiftingCountOfUncontrolledGates;"
         "AfterHadamardAndMeasLiftingCountOfControllingQubits;\n";

  std::ifstream csvFile(
      "/home/lian/DLR/mqt-core/mqt-core/EvaluationOfLifting.csv");
  std::set<std::string> processedLines; // or std::unordered_set<std::string>
  std::string line;
  while (std::getline(csvFile, line)) {
    std::size_t pos = line.find(";");
    std::string res = line =
        (pos == std::string::npos) ? line : line.substr(0, pos);
    res.erase(res.size() - 1);
    res.erase(0, 1);
    processedLines.insert(res);
  }

  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(originalInput)) {
    if (entry.is_regular_file()) {
      std::filesystem::path relative =
          std::filesystem::relative(entry.path(), originalInput);
      std::filesystem::path mLFile = measurementLiftInput / relative;
      std::filesystem::path hmLFile = hadamardMeasurementLiftInput / relative;
      std::vector<std::string> files = {entry.path().string(), mLFile.string(),
                                        hmLFile.string()};
      // if (processedLines.contains(relative)) {
      //   std::cout << "Skipping " << relative << std::endl;
      //   continue;
      // }
      out << relative << ";";

      for (const auto currentFile : files) {
        llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
            llvm::MemoryBuffer::getFile(currentFile);
        if (std::error_code ec = fileOrErr.getError()) {
          std::cerr << "Error reading file: " << ec.message() << "\n";
          out << ";;;";
          std::ofstream outFile(errorFile, std::ios::app);
          if (outFile.is_open()) {
            outFile << currentFile << std::endl;
            outFile.close();
          } else {
            std::cerr << "Unable to open error file";
          }
          continue;
        }
        fileOrErr = llvm::MemoryBuffer::getFile(currentFile);
        if (std::error_code ec = fileOrErr.getError()) {
          std::cerr << "Error reading file: " << ec.message() << "\n";
          out << ";;;";
          std::ofstream outFile(errorFile, std::ios::app);
          if (outFile.is_open()) {
            outFile << currentFile << std::endl;
            outFile.close();
          } else {
            std::cerr << "Unable to open error file";
          }
          continue;
        }

        llvm::SourceMgr sourceMgr;
        sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

        mlir::OwningOpRef<mlir::ModuleOp> module =
            mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &*context);

        if (!module) {
          std::cerr << "Failed to parse MLIR file\n";
          out << ";;;";
          std::ofstream outFile(errorFile, std::ios::app);
          if (outFile.is_open()) {
            outFile << currentFile << std::endl;
            outFile.close();
          } else {
            std::cerr << "Unable to open error file";
          }
          continue;
        }

        int countOfGates = 0;
        int countOfSingleGates = 0;
        int countOfCtrlQubits = 0;
        module->walk([&](mqt::ir::opt::UnitaryInterface op) {
          if (op.isControlled()) {
            countOfCtrlQubits += op.getAllCtrlInQubits().size();
          } else {
            countOfSingleGates++;
          }
          countOfGates++;
        });
        out << countOfGates << ";" << countOfSingleGates << ";"
            << countOfCtrlQubits << ";";
        // std::cout << "Found " << countOfGates
        //           << " occurrences of quantum instructions\n";
      }
      out << '\n';
      std::cout << "Processed " << entry.path() << '\n';
    }
  }
  return 0;
}
