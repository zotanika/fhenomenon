#pragma once

#include "scheduler/Operation.h"
#include <unordered_map>
#include <vector>

namespace fhenomenon {

template <typename T>
class Receiver {
  private:
  std::unordered_map<void *, std::vector<Operation<T>>> operationsMap;

  public:
  template <typename T> void receive(const Operation<T> &op) {
    void *key = op.operand1.encrypted_data.get();
    operationsMap[key].push_back(op);
  }

  const std::unordered_map<void *, std::vector<Operation<int>>> &getOperationsMap() const { return operationsMap; }
};

} // namespace fhenomenon
