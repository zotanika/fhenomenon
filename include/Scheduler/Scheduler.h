#pragma once

#include "Common.h"
#include "Scheduler/ASTNode.h"
#include "Scheduler/FusedOperation.h"
#include "Scheduler/LowerToFhnProgram.h"
#include "Scheduler/Operation.h"
#include "Scheduler/Planner.h"
#include "Scheduler/PreASTPass.h"
#include "Scheduler/ASTPass.h"
#include "FHN/FhnDefaultExecutor.h"
#include "FHN/ToyFheKernels.h"

#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fhenomenon {

// Forward declaration
class Backend;

namespace scheduler {

class Scheduler {
  private:
  std::vector<std::shared_ptr<ASTPass>> astPasses_;
  std::vector<std::shared_ptr<PreASTPass>> preASTPasses_;
  const Backend *backend_delegate_; // Backend delegate for executing operations
  std::unordered_set<std::string> registeredPreASTPasses_;
  std::unordered_set<std::string> registeredASTPasses_;

  void validateDependencies(const std::string &passName,
                            const std::vector<std::string> &deps,
                            const std::unordered_set<std::string> &registered) {
    for (const auto &dep : deps) {
      if (!registered.count(dep)) {
        throw std::runtime_error(
          "Pass '" + passName + "' requires '" + dep +
          "' to be registered first");
      }
    }
  }

  public:
  // Constructor - takes backend as delegate
  explicit Scheduler(const Backend &backend) : backend_delegate_(&backend) {}

  // Set backend delegate
  void setBackendDelegate(const Backend &backend) { backend_delegate_ = &backend; }

  const Backend &getBackendDelegate() const {
    if (!backend_delegate_) {
      throw std::runtime_error("Scheduler: Backend delegate not set");
    }
    return *backend_delegate_;
  }
  template <typename T>
  void buildAST(const std::vector<std::shared_ptr<scheduler::OperationBase>> &operations, Planner<T> &plan) {
    std::unordered_map<std::shared_ptr<Compuon<T>>, std::shared_ptr<ASTNode>> entityToNodeMap;
    std::vector<std::shared_ptr<ASTNode>> roots;
    for (const auto &op : operations) {
      // Handle FusedOperation (from pre-AST passes like MatMul recognition)
      auto fusedOp = std::dynamic_pointer_cast<FusedOperation<T>>(op);
      if (fusedOp) {
        fusedOp->setBackendDelegate(&getBackendDelegate());

        // Create dependency nodes for all inputs
        std::vector<std::shared_ptr<ASTNode>> deps;
        for (const auto &input : fusedOp->getInputs()) {
          if (entityToNodeMap.count(input)) {
            deps.push_back(entityToNodeMap[input]);
          } else {
            auto node = std::make_shared<OperandNode<T>>(input);
            entityToNodeMap[input] = node;
            deps.push_back(node);
          }
        }

        // Create FusedKernelNode
        auto fusedNode = std::make_shared<FusedKernelNode<T>>(
          fusedOp, std::move(deps),
          std::vector<std::shared_ptr<Compuon<T>>>(fusedOp->getOutputs()));

        // Map all outputs to this node
        for (const auto &output : fusedOp->getOutputs()) {
          entityToNodeMap[output] = fusedNode;
        }

        plan.addRoot(fusedNode);
        continue;
      }

      auto operation = std::dynamic_pointer_cast<scheduler::Operation<T>>(op);
      if (!operation) {
        throw std::runtime_error("Invalid operation cast.");
      }

      // Set backend delegate on operation (scheduler orchestrates, backend executes)
      operation->setBackendDelegate(&getBackendDelegate());

      auto operand1 = operation->getOperand1();
      auto operand2 = operation->getOperand2();
      auto result = operation->getResult();

      std::shared_ptr<ASTNode> left;
      std::shared_ptr<ASTNode> right;

      if (entityToNodeMap.count(operand1)) {
        left = entityToNodeMap[operand1];
      } else {
        left = std::make_shared<OperandNode<T>>(operand1);
        entityToNodeMap[operand1] = left;
      }
      if (operand2->isScalar()) {
        right = std::make_shared<OperandNode<T>>(operand2);
      } else {
        if (entityToNodeMap.count(operand2)) {
          right = entityToNodeMap[operand2];
        } else {
          right = std::make_shared<OperandNode<T>>(operand2);
          entityToNodeMap[operand2] = right;
        }
      }
      if (operation->getType() == OperationType::Add || operation->getType() == OperationType::Multiply) {
        auto operatorNode = std::make_shared<OperatorNode<T>>(operation, operation->getType(), left, right, result);
        entityToNodeMap[result] = operatorNode;
        plan.addRoot(operatorNode);
      } else if (operation->getType() == OperationType::Assignment) {
        auto assignmentNode =
          std::make_shared<OperatorNode<T>>(operation, OperationType::Assignment, left, right, operand1);
        entityToNodeMap[operand1] = assignmentNode;
        plan.addRoot(assignmentNode);
      } else {
        throw std::runtime_error("Unsupported operation type.");
      }
    }
  }

  template <typename T>
  Planner<T> &buildGraph(std::vector<std::shared_ptr<OperationBase>> &operationsMap, Planner<T> &plan) {
    LOG_MESSAGE("Build Graph");
    // Run pre-AST passes before building the AST
    for (auto &pass : preASTPasses_) {
      pass->apply(operationsMap, getBackendDelegate());
    }
    this->buildAST(operationsMap, plan);
    return plan;
  }

  template <typename T> void optimizeGraph(Planner<T> &plan) {
    LOG_MESSAGE("Optimize");
    // Apply custom scheduling strategies
    for (auto &pass : astPasses_) {
      pass->apply(plan);
    }
    // Further optimizations can be added here
  }

  template <typename T> void evaluateGraph(Planner<T> &plan) {
    const auto &roots = plan.getRoots();
    if (roots.empty()) {
      throw std::runtime_error("No roots available for evaluation.");
    }

    // Evaluate graph using backend delegate
    for (const auto &root : roots) {
      root->evaluate();
    }
  }

  /// Evaluate graph using FHN pipeline: lower AST to FhnProgram, dispatch via executor.
  /// This is the new execution path alongside the old evaluateGraph().
  /// buffers_out is populated with FhnBuffer pointers indexed by result_id.
  /// The caller is responsible for managing buffer lifecycle.
  template <typename T>
  FhnProgram *lowerGraph(Planner<T> &plan) {
    LowerToFhnProgram lowering;
    return lowering.lower(plan);
  }

  void addASTPass(std::shared_ptr<ASTPass> pass) {
    validateDependencies(pass->name(), pass->dependencies(), registeredASTPasses_);
    registeredASTPasses_.insert(pass->name());
    astPasses_.push_back(std::move(pass));
  }
  void addPreASTPass(std::shared_ptr<PreASTPass> pass) {
    validateDependencies(pass->name(), pass->dependencies(), registeredPreASTPasses_);
    registeredPreASTPasses_.insert(pass->name());
    preASTPasses_.push_back(std::move(pass));
  }
};

} // namespace scheduler
} // namespace fhenomenon
