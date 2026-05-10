// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#include "approx/Dialect.h"
#include "approx/Ops.h"
#include "approx/Passes/Passes.h"

namespace mlir::iree_compiler::approx {
namespace {

struct ApproxOptions {
  void bindOptions(OptionsBinder &binder) {}
};

struct ApproxSession
    : public PluginSession<ApproxSession, ApproxOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {
    mlir::registerEmitApproxPass();
    mlir::registerEmitManagementPass();
    mlir::registerConfigApproxPass();
    mlir::registerTransformApproxPass();
    mlir::registerPreEmitTransformationPass();
    mlir::registerFinalizeApproxPass();
    mlir::registerLegalizeToStablehloPass();
  }

  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<mlir::approx::approxDialect>();
  }
};

} // namespace
} // namespace mlir::iree_compiler::approx

IREE_DEFINE_COMPILER_OPTION_FLAGS(
    ::mlir::iree_compiler::approx::ApproxOptions);

extern "C" bool iree_register_compiler_plugin_input_approx(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<::mlir::iree_compiler::approx::ApproxSession>(
      "input_approx");
  return true;
}
