#pragma once

#include "Scheduler/ASTNode.h"
#include "Scheduler/Operation.h"

#include <memory>
#include <vector>

namespace fhenomenon {
namespace scheduler {

/// A fused operation groups multiple elementary operations into a single
/// composite kernel (e.g., matrix-multiply = many rotate + multiply + add).
template <typename T> class FusedOperation : public OperationBase {
public:
  FusedOperation(OperationType type,
                 std::vector<std::shared_ptr<Compuon<T>>> inputs,
                 std::vector<std::shared_ptr<Compuon<T>>> outputs)
      : type_(type), inputs_(std::move(inputs)), outputs_(std::move(outputs)) {}

  void execute() override {
    // Default: no-op; subclasses or backend delegate provide implementation.
  }

  OperationType getType() const override { return type_; }

  void setBackendDelegate(const Backend *backend) {
    backend_delegate_ = backend;
  }

  const std::vector<std::shared_ptr<Compuon<T>>> &getInputs() const {
    return inputs_;
  }

  const std::vector<std::shared_ptr<Compuon<T>>> &getOutputs() const {
    return outputs_;
  }

private:
  OperationType type_;
  std::vector<std::shared_ptr<Compuon<T>>> inputs_;
  std::vector<std::shared_ptr<Compuon<T>>> outputs_;
  const Backend *backend_delegate_ = nullptr;
};

/// AST node that wraps a FusedOperation for evaluation in the planner.
template <typename T> class FusedKernelNode : public ASTNode {
public:
  FusedKernelNode(std::shared_ptr<FusedOperation<T>> op,
                  std::vector<std::shared_ptr<ASTNode>> dependencies,
                  std::vector<std::shared_ptr<Compuon<T>>> outputs)
      : op_(std::move(op)), dependencies_(std::move(dependencies)),
        outputs_(std::move(outputs)), evaluated_(false) {}

  void evaluate() override {
    if (evaluated_)
      return;
    for (auto &dep : dependencies_) {
      dep->evaluate();
    }
    op_->execute();
    evaluated_ = true;
  }

  void print(int depth = 0) const override {
    std::string indent(static_cast<std::string::size_type>(depth * 2), ' ');
    std::cout << indent << "FusedKernel" << std::endl;
    for (const auto &dep : dependencies_) {
      dep->print(depth + 1);
    }
  }

private:
  std::shared_ptr<FusedOperation<T>> op_;
  std::vector<std::shared_ptr<ASTNode>> dependencies_;
  std::vector<std::shared_ptr<Compuon<T>>> outputs_;
  bool evaluated_;
};

} // namespace scheduler
} // namespace fhenomenon
