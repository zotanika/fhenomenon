#pragma once

namespace fhenomenon {
namespace scheduler {

class Dispatcher {
  public:
  void dispatch(const Planner &planner) {
    // Execute the operations in the plan
    for (const auto &op : planner.getOperations()) {
      op.execute();
    }
  }
};

} // namespace scheduler
} // namespace fhenomenon
