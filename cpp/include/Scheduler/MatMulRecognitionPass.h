#pragma once

#include "Scheduler/PreASTPass.h"

namespace fhenomenon {
namespace scheduler {

class MatMulRecognitionPass : public PreASTPass {
  public:
  void apply(std::vector<std::shared_ptr<OperationBase>> &operations, const Backend &backend) override;
  std::string name() const override { return "MatMulRecognition"; }
};

} // namespace scheduler
} // namespace fhenomenon
