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
  std::string filename = "simpleqcp.mlir";

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

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFile(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    std::cerr << "Error reading file: " << ec.message() << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &*context);

  if (!module) {
    std::cerr << "Failed to parse MLIR file\n";
    return 1;
  }

  int count = 0;
  module->walk([&](mqt::ir::opt::UnitaryInterface) { count++; });
  /*More robust:
   * module->walk([&](mlir::arith::AddIOp addOp) { count++; });
   */

  std::cout << "Found " << count << " occurrences of quantum instructions\n";
  return 0;
}
