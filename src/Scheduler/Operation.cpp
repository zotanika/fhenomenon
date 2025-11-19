#include "Scheduler/Operation.h"
#include "Session/Session.h"
#include "Backend/Backend.h"

namespace fhenomenon {
namespace scheduler {

template <typename T> void Operation<T>::execute() {
  // Use backend delegate instead of Backend::getInstance()
  // Scheduler orchestrates, backend executes
  if (!backend_delegate_) {
    throw std::runtime_error("Operation::execute: Backend delegate not set");
  }
  
  const Backend& backend = *backend_delegate_;
  
  if (type_ == OperationType::Assignment) {
    auto op1_ref = Session::getSession()->getEntity(*operand1_);
    LOG_MESSAGE("OperationType::Assignment start (" << operand1_->getValue() << ", " << operand2_->getValue() << ")");
    
    // Copy value
    op1_ref->setValue(operand2_->getValue());
    
    // Copy ciphertext if operand2 is encrypted
    if (operand2_->isEncrypted_ && operand2_->ciphertext_) {
      op1_ref->ciphertext_ = operand2_->ciphertext_->clone();
      op1_ref->isEncrypted_ = true;
      LOG_MESSAGE("OperationType::Assignment: Copied encrypted ciphertext");
    } else if (operand2_->isEncrypted_) {
      // If operand2 should be encrypted but ciphertext is missing, encrypt it using backend delegate
      if (operand2_->getProfile()) {
        backend.transform(*op1_ref, *(operand2_->getProfile()->getParam()));
      }
    }
    
    LOG_MESSAGE("OperationType::Assignment end (" << op1_ref->getValue() << ", " << operand2_->getValue() << ")");
  } else if (type_ == OperationType::Refresh) {
    // operand1.bootstrap();
    LOG_MESSAGE("OperationType::Refresh not yet implemented");
  } else if (type_ == OperationType::Add) {
    LOG_MESSAGE("OperationType::Add start (" << result_->getValue() << " = " << operand1_->getValue() << " + "
                                             << operand2_->getValue() << ")");

    // Use backend delegate to execute addition
    auto result_add = backend.add(static_cast<const CompuonBase &>(*operand1_),
                                 static_cast<const CompuonBase &>(*operand2_));

    if (!result_add) {
      throw std::runtime_error("Backend::add returned nullptr");
    }

    auto result = std::dynamic_pointer_cast<Compuon<T>>(result_add);

    if (!result) {
      throw std::runtime_error("Backend::add returned wrong type");
    }

    // Get the result entity from session and update both value and ciphertext
    auto result_ref = Session::getSession()->getEntity(*result_);
    if (result_ref) {
      result_ref->setValue(result->getValue());
      
      // Copy ciphertext from backend result
      if (result->isEncrypted_ && result->ciphertext_) {
        result_ref->ciphertext_ = result->ciphertext_->clone();
        result_ref->isEncrypted_ = true;
        result_ref->setProfile(result->getProfile());
        LOG_MESSAGE("OperationType::Add: Copied encrypted ciphertext to result");
      }
    }
    
    LOG_MESSAGE("OperationType::Add end (" << result_ref->getValue() << " = " << operand1_->getValue() << " + "
                                           << operand2_->getValue() << ")");
  } else if (type_ == OperationType::Multiply) {
    LOG_MESSAGE("OperationType::Multiply start (" << result_->getValue() << " = " << operand1_->getValue() << " * "
                                             << operand2_->getValue() << ")");
    
    // Use backend delegate to execute multiplication
    auto result_multiply = backend.multiply(static_cast<const CompuonBase &>(*operand1_),
                                           static_cast<const CompuonBase &>(*operand2_));
    if(!result_multiply){
      throw std::runtime_error("Backend::multiply returned nullptr");
    }

    auto result = std::dynamic_pointer_cast<Compuon<T>>(result_multiply);
    
    if (!result) {
      throw std::runtime_error("Backend::multiply returned wrong type");
    }

    // Get the result entity from session and update both value and ciphertext
    auto result_ref = Session::getSession()->getEntity(*result_);
    if (result_ref) {
      result_ref->setValue(result->getValue());
      
      // Copy ciphertext from backend result
      if (result->isEncrypted_ && result->ciphertext_) {
        result_ref->ciphertext_ = result->ciphertext_->clone();
        result_ref->isEncrypted_ = true;
        result_ref->setProfile(result->getProfile());
        LOG_MESSAGE("OperationType::Multiply: Copied encrypted ciphertext to result");
      }
    }
    
    LOG_MESSAGE("OperationType::Multiply end (" << result_ref->getValue() << " = " << operand1_->getValue() << " * "
                                           << operand2_->getValue() << ")");
  } else {
    throw std::runtime_error("Invalid operation");
  }
}

template class Operation<int>;

} // namespace scheduler
} // namespace fhenomenon
