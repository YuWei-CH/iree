// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_FOLDMEMREFCOPYINTODPSOPSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

static Value getSingleMemRefRoot(Value value) {
  while (Operation *definingOp = value.getDefiningOp()) {
    if (isa<memref::AllocOp, memref::AllocaOp,
            IREE::HAL::InterfaceBindingSubspanOp>(definingOp)) {
      break;
    }

    Value memrefOperand;
    int memrefOperandCount = 0;
    for (Value operand : definingOp->getOperands()) {
      if (isa<MemRefType>(operand.getType())) {
        memrefOperand = operand;
        ++memrefOperandCount;
      }
    }
    if (memrefOperandCount != 1) {
      break;
    }
    value = memrefOperand;
  }
  return value;
}

static bool isDerivedFromLocalAlloc(Value value) {
  Operation *root = getSingleMemRefRoot(value).getDefiningOp();
  return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(root);
}

static std::optional<IREE::HAL::InterfaceBindingSubspanOp>
getSourceSubspan(Value value) {
  auto typedValue = dyn_cast<TypedValue<MemRefType>>(value);
  if (!typedValue) {
    return std::nullopt;
  }
  return getSourceSubspanMemref(typedValue);
}

static bool mayAlias(Value lhs, Value rhs) {
  if (lhs == rhs) {
    return true;
  }

  std::optional<IREE::HAL::InterfaceBindingSubspanOp> lhsSubspan =
      getSourceSubspan(lhs);
  std::optional<IREE::HAL::InterfaceBindingSubspanOp> rhsSubspan =
      getSourceSubspan(rhs);
  if (lhsSubspan && rhsSubspan) {
    // Different interface bindings cannot alias. Treat different subspans of
    // the same binding conservatively because offsets may be dynamic.
    return lhsSubspan->getBinding() == rhsSubspan->getBinding();
  }

  if (lhsSubspan || rhsSubspan) {
    Value other = lhsSubspan ? rhs : lhs;
    return !isDerivedFromLocalAlloc(other);
  }

  Value lhsRoot = getSingleMemRefRoot(lhs);
  Value rhsRoot = getSingleMemRefRoot(rhs);
  if (isDerivedFromLocalAlloc(lhsRoot) && isDerivedFromLocalAlloc(rhsRoot)) {
    return lhsRoot == rhsRoot;
  }

  return true;
}

static bool opMayAccessTarget(Operation *op, Value target) {
  for (Value operand : op->getOperands()) {
    if (isa<MemRefType>(operand.getType()) && mayAlias(operand, target)) {
      return true;
    }
  }
  return false;
}

static bool hasInterveningTargetAccess(Operation *first, Operation *last,
                                       Operation *allowedOp, Value target) {
  for (Operation *op = first->getNextNode(); op && op != last;
       op = op->getNextNode()) {
    if (op == allowedOp) {
      continue;
    }
    if (opMayAccessTarget(op, target)) {
      return true;
    }
  }
  return false;
}

static bool targetMayAliasDpsReads(DestinationStyleOpInterface dpsOp,
                                   OpOperand *forwardedInitOperand,
                                   Value copySource, Value target) {
  if (mayAlias(copySource, target)) {
    return true;
  }

  for (OpOperand &operand : dpsOp->getOpOperands()) {
    if (&operand == forwardedInitOperand) {
      continue;
    }
    if (isa<MemRefType>(operand.get().getType()) &&
        mayAlias(operand.get(), target)) {
      return true;
    }
  }
  return false;
}

struct FoldTemporaryCopyIntoDpsOp final : OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOut,
                                PatternRewriter &rewriter) const override {
    auto allocOp = copyOut.getSource().getDefiningOp<memref::AllocOp>();
    if (!allocOp) {
      return failure();
    }

    memref::CopyOp copyIn;
    DestinationStyleOpInterface dpsOp;
    OpOperand *forwardedInitOperand = nullptr;
    for (Operation *user : allocOp->getUsers()) {
      if (user == copyOut.getOperation()) {
        continue;
      }
      if (auto copy = dyn_cast<memref::CopyOp>(user)) {
        if (copy.getTarget() == allocOp.getResult() && !copyIn) {
          copyIn = copy;
          continue;
        }
        return failure();
      }
      if (auto candidate = dyn_cast<DestinationStyleOpInterface>(user)) {
        if (candidate.getNumDpsInits() != 1 || dpsOp) {
          return failure();
        }
        OpOperand *initOperand = candidate.getDpsInitOperand(0);
        if (initOperand->get() != allocOp.getResult()) {
          return failure();
        }
        dpsOp = candidate;
        forwardedInitOperand = initOperand;
        continue;
      }
      return failure();
    }

    if (!copyIn || !dpsOp || !forwardedInitOperand) {
      return failure();
    }
    if (copyIn->getBlock() != dpsOp->getBlock() ||
        dpsOp->getBlock() != copyOut->getBlock()) {
      return failure();
    }
    if (!copyIn->isBeforeInBlock(dpsOp) ||
        !dpsOp->isBeforeInBlock(copyOut)) {
      return failure();
    }

    Value finalTarget = copyOut.getTarget();
    if (targetMayAliasDpsReads(dpsOp, forwardedInitOperand, copyIn.getSource(),
                               finalTarget)) {
      return failure();
    }
    if (hasInterveningTargetAccess(copyIn, copyOut, dpsOp, finalTarget)) {
      return failure();
    }

    rewriter.setInsertionPoint(copyIn);
    memref::CopyOp::create(rewriter, copyIn.getLoc(), copyIn.getSource(),
                           finalTarget);
    forwardedInitOperand->set(finalTarget);

    rewriter.eraseOp(copyOut);
    rewriter.eraseOp(copyIn);
    if (allocOp->use_empty()) {
      rewriter.eraseOp(allocOp);
    }
    return success();
  }
};

struct FoldMemRefCopyIntoDPSOpsPass final
    : impl::FoldMemRefCopyIntoDPSOpsPassBase<
          FoldMemRefCopyIntoDPSOpsPass> {
  using FoldMemRefCopyIntoDPSOpsPassBase::
      FoldMemRefCopyIntoDPSOpsPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldTemporaryCopyIntoDpsOp>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
