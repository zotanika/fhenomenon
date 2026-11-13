#include "Scheduler/LowerToFhnProgram.h"

namespace fhenomenon {
namespace scheduler {

FhnOpCode LowerToFhnProgram::mapOpType(OperationType type) {
  switch (type) {
  case OperationType::Add:
    return FHN_ADD_CC;
  case OperationType::Sub:
    return FHN_SUB_CC;
  case OperationType::Multiply:
    return FHN_HMULT;
  case OperationType::Conjugate:
    return FHN_CONJUGATE;
  case OperationType::LeftRotate:
  case OperationType::RightRotate:
    return FHN_ROTATE;
  default:
    return FHN_NOP;
  }
}

} // namespace scheduler
} // namespace fhenomenon
