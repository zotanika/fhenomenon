#pragma once

#include "Common.h"
#include "Scheduler/ASTNode.h"
#include "Scheduler/Operation.h"
#include "Scheduler/Planner.h"
#include "Scheduler/Strategy.h"

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace fhenomenon {

// Forward declaration
class Backend;

namespace scheduler {

class Scheduler {
  private:
  std::vector<std::shared_ptr<Strategy>> strategies;
  const Backend *backend_delegate_; // Backend delegate for executing operations

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
    this->buildAST(operationsMap, plan);
    return plan;
  }

  template <typename T> void optimizeGraph(Planner<T> &plan) {
    LOG_MESSAGE("Optimize");
    // Apply custom scheduling strategies
    for (auto &strategy : strategies) {
      strategy->apply(plan);
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

  void addStrategy(const std::shared_ptr<Strategy> &strategy) { strategies.push_back(strategy); }
};

} // namespace scheduler
} // namespace fhenomenon
