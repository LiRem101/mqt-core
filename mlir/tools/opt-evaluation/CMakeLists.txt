# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        ${transform_libs}
        ${passes_libs}
        MLIROptLib
        MLIRMQTOptTransforms
        MQTRefToMQTOpt
        MQTOptToMQTRef
        MQTRefToQIR
        QIRToMQTRef
        MQT::CoreQASM
        MLIRMQTRefTranslation)

add_mlir_tool(opt-evaluation opt-evaluation.cpp DEPENDS ${LIBS} SUPPORT_PLUGINS)
target_compile_options(opt-evaluation PRIVATE -fexceptions)
target_link_libraries(opt-evaluation PUBLIC ${LIBS})
llvm_update_compile_flags(opt-evaluation)
mlir_check_all_link_libraries(opt-evaluation)
export_executable_symbols_for_plugins(opt-evaluation)
