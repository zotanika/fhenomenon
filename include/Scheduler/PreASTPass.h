#pragma once

#include "Scheduler/Operation.h"

#include <memory>
#include <string>
#include <vector>

namespace fhenomenon {

class Backend;

namespace scheduler {

class PreASTPass {
  public:
  virtual ~PreASTPass() = default;
  virtual void apply(std::vector<std::shared_ptr<OperationBase>> &operations, const Backend &backend) = 0;

  virtual std::string name() const = 0;
  virtual std::vector<std::string> dependencies() const { return {}; }
};

} // namespace scheduler
} // namespace fhenomenon
