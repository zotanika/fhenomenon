#pragma once

#include "Scheduler/ASTNode.h"
#include "Scheduler/Operation.h"
#include <unordered_set>
#include <vector>

namespace fhenomenon {
namespace scheduler {

template <typename T> class Planner {
  private:
  std::unordered_map<std::shared_ptr<Compuon<T>>, std::shared_ptr<ASTNode>> entityToNodeMap_;
  std::unordered_set<std::shared_ptr<ASTNode>> rootSet_;
  std::vector<std::shared_ptr<ASTNode>> roots_;

  public:
  const std::vector<std::shared_ptr<ASTNode>> &getRoots() const { return roots_; }
  std::vector<std::shared_ptr<ASTNode>> &getRoots() { return roots_; }
  const std::unordered_map<std::shared_ptr<Compuon<T>>, std::shared_ptr<ASTNode>> &getEntityToNodeMap() const {
    return entityToNodeMap_;
  }
  std::unordered_map<std::shared_ptr<Compuon<T>>, std::shared_ptr<ASTNode>> &getEntityToNodeMap() {
    return entityToNodeMap_;
  }

  void addRoot(const std::shared_ptr<ASTNode> &node) {
    if (rootSet_.insert(node).second)
      roots_.push_back(node);
  }
};

} // namespace scheduler
} // namespace fhenomenon
