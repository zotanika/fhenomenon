#pragma once

#include "Operation.h"
#include <vector>

namespace fhenomenon {
namespace scheduler {

class Strategy {
  public:
  virtual ~Strategy() = default;
  virtual void apply(Planner<int> &plan) const = 0;
};

class PrintOperationsStrategy final : public Strategy {
  public:
  void apply(Planner<int> &plan) const override {
    LOG_MESSAGE("Tree Structure of Operations");
    for (const auto &root : plan.getRoots()) {
      LOG_MESSAGE("Root");
      root->print(0);
    }
  }
};

/*
class ConstantFoldingStrategy final: public Strategy {
  public:
  void apply(Planner<int> &plan) const override{
    LOG_MESSAGE("Constant Folding Strategy...");
    const auto& roots = plan.getRoots();
  }
}
*/
/*
class FuseOperationsStrategy final : public Strategy {
  public:
  void apply(std::vector<Operation<int>> &ops) const override  {
    LOG_MESSAGE("Constant Folding Strategy...");
    // Example strategy: Fusing consecutive additions
    std::vector<Operation<int>> fusedOps;
    std::unordered_map<std::shared_ptr<Compuon<int>>, std::vector<Operation<int>>> pending_operations;

    auto flushPending = [&](const std::shared_ptr<Compuon<int>> &entity_ptr) {
        if (pending_operations.find(entity_ptr) != pending_operations.end()) {
            for (const auto &tmp : pending_operations[entity_ptr]) {
                fusedOps.push_back(Operation<int>(tmp.getType(), entity_ptr, tmp.getOperand2(), entity_ptr));
            }
            pending_operations[entity_ptr].clear();
        }
    };

    for(std::size_t i=0;i<ops.size();i++){
      const auto& op = ops[i];
      const auto& operand1 = op.getOperand1();
      const auto& operand2 = op.getOperand2();
      const auto& result = op.getResult();
      const auto& operationType = op.getType();

      // if(a = b + 2 -> res = b+2 , a = res)
      if(i<ops.size()-1 && operand1 != ops[i+1].getOperand1() && operand2->isScalar()){
        if(ops[i+1].getType() == OperationType::Assignment){
          flushPending(operand1);
          fusedOps.push_back(op);
          continue;
        }
      }

      if(i>0 && operand1 == ops[i-1].getOperand1() && ops[i-1].getOperand2()->isScalar()){
        if(operationType == OperationType::Assignment){
            continue;
        }

        flushPending(operand1);
        fusedOps.push_back(op);
        continue;
      }

      //if(a = b + c)
      if(!operand2->isScalar()){
        flushPending(operand1);
        flushPending(operand2);
        fusedOps.push_back(op);
        pending_operations[operand1].clear();
        pending_operations[operand2].clear();
      }

      //if(a = a + 2)
      else if((pending_operations.find(operand1) == pending_operations.end()) && operand2->isScalar()){
        pending_operations[operand1] = {op};
      }
      else if((pending_operations.find(operand1) != pending_operations.end()) && operand2->isScalar()){
        pending_operations[operand1].push_back(op);
      }
      else fusedOps.push_back(op);
    }

    // Flush all remainder
    for(auto const &pair:pending_operations){
      flushPending(pair.first);
    }

    ops.swap(fusedOps);
  }
};

class AdditionToMultiplicationStrategy : public Strategy {
public:
    void apply(std::vector<Operation<int>> &ops) const override {
        LOG_MESSAGE("Addition To Multiplication for same constant...");
        std::vector<Operation<int>> fusedOps;
        std::size_t len = 1;

        auto canFuse = [](const Operation<int> &a, const Operation<int> &b) {
            return a.getOperand1() == b.getOperand1() &&
                   a.getOperand2()->getValue() == b.getOperand2()->getValue() &&
                   a.getType() == OperationType::Add && b.getType() == OperationType::Add;
        };

        auto fuseOperations = [&](std::size_t len, const Operation<int> &op) {
            auto size = std::make_shared<Compuon<int>>(len);
            auto entity = std::make_shared<Compuon<int>>(1234);
            size->belong(op.getOperand1()->getProfile());
            entity->belong(op.getOperand1()->getProfile());
            fusedOps.push_back(Operation<int>(OperationType::Multiply, op.getOperand2(), size, entity));
            fusedOps.push_back(Operation<int>(OperationType::Add, op.getOperand1(), entity, op.getOperand1()));
        };

        for (std::size_t i = 1; i < ops.size(); i++) {
            if (canFuse(ops[i - 1], ops[i])) {
                len++;
            }
            else {
                if (ops[i - 1].getType() == OperationType::Add && len > 1)
                  fuseOperations(len, ops[i - 1]);

                else
                  fusedOps.push_back(ops[i - 1]);

                len = 1;
            }
        }

        // last operation
        const auto &lastOp = ops.back();
        if (lastOp.getType() == OperationType::Add && len > 1) {
            fuseOperations(len, lastOp);
        } else {
            fusedOps.push_back(lastOp);
        }

        ops.swap(fusedOps);
    }
};
*/

} // namespace scheduler
} // namespace fhenomenon
