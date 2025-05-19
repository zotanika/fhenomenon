#include "Scheduler/Operation.h"
#include "Session/Session.h"

namespace fhenomenon {
namespace scheduler {

template <typename T> void Operation<T>::execute() {
  if (type_ == OperationType::Assignment) {
    auto op1_ref = Session::getSession()->getEntity(*operand1_);
    LOG_MESSAGE("OperationType::Assignment start (" << operand1_->getValue() << ", " << operand2_->getValue() << ")");
    // Session::getSession()->getEntity(*operand1_)->setValue(operand2_->getValue());
    op1_ref->setValue(operand2_->getValue());
    LOG_MESSAGE("OperationType::Assignment end (" << op1_ref->getValue() << ", " << operand2_->getValue() << ")");
  } else if (type_ == OperationType::Refresh) {
    // operand1.bootstrap();
  } else if (type_ == OperationType::Add) {
    LOG_MESSAGE("OperationType::Add start (" << result_->getValue() << " = " << operand1_->getValue() << " + "
                                             << operand2_->getValue() << ")");

    auto result_add = Backend::getInstance().add(static_cast<const CompuonBase &>(*operand1_),
                                                 static_cast<const CompuonBase &>(*operand2_));

    if (!result_add) {
      throw std::runtime_error("Backend::add returned nullptr");
    }

    auto result = std::dynamic_pointer_cast<Compuon<T>>(result_add);

    if (!result) {
      throw std::runtime_error("Backend::add returned nullptr");
    }

    Session::getSession()->getEntity(*result_)->setValue(result->getValue());
    LOG_MESSAGE("OperationType::Add end (" << result_->getValue() << " = " << operand1_->getValue() << " + "
                                           << operand2_->getValue() << ")");
  } else if (type_ == OperationType::Multiply) {
    LOG_MESSAGE("OperationType::Multiply start (" << result_->getValue() << " = " << operand1_->getValue() << " * "
                                             << operand2_->getValue() << ")");
    auto result_multiply = Backend::getInstance().multiply(static_cast<const CompuonBase &>(*operand1_),
                                                  static_cast<const CompuonBase &>(*operand2_));
    if(!result_multiply){
      throw std::runtime_error("Backend::multiply returned nullptr");
    }

    auto result = std::dynamic_pointer_cast<Compuon<T>>(result_multiply);

    Session::getSession()->getEntity(*result_)->setValue(result->getValue());
    LOG_MESSAGE("OperationType::Multiply end (" << result_->getValue() << " = " << operand1_->getValue() << " * "
                                           << operand2_->getValue() << ")");
  } else {
    throw std::runtime_error("Invalid operation");
  }
}

template class Operation<int>;

} // namespace scheduler
} // namespace fhenomenon
