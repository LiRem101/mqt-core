#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include <iostream>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: count_ops <file.mlir> <op-name>\n";
    return 1;
  }

  std::string filename = argv[1];
  std::string targetOpName = argv[2];

  mlir::MLIRContext context;
  context.allowUnregisteredDialects();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFile(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    std::cerr << "Error reading file: " << ec.message() << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);

  if (!module) {
    std::cerr << "Failed to parse MLIR file\n";
    return 1;
  }

  int count = 0;
  module->walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == targetOpName)
      count++;
  });
  /*More robust:
   * module->walk([&](mlir::arith::AddIOp addOp) { count++; });
   */

  std::cout << "Found " << count << " occurrences of op '" << targetOpName
            << "'\n";
  return 0;
}
