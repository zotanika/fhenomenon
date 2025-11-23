#include "Compuon.h"
#include "Backend/Builtin.h"
#include "Profile.h"
#include "Session/Session.h"

namespace fhenomenon {
std::shared_ptr<Profile> Profile::profile_ = nullptr;
// Forward declaration of Operation class to prevent circular dependency
namespace scheduler {

template <typename T> class Operation;

} // namespace scheduler

template <typename T> class Compuon;

template <typename T>
Compuon<T>::Compuon(const Compuon<T> &other)
  : std::enable_shared_from_this<Compuon<T>>(other), val_(other.val_), profile_(other.profile_) {
  LOG_MESSAGE("Copy constructor with value: " << val_ << " (" << this << ")");
  if (Session::getSession()->isActive()) {
    Session::getSession()->setEntity<T>(this, const_cast<Compuon<T> &>(other));
    Session::getSession()->saveEntity(this, const_cast<Compuon<T> &>(other));
    // Session::getSession()->saveEntity(&other, *this);
  }
}

template <typename T> void Compuon<T>::belong(std::shared_ptr<Profile> newProfile) {
  profile_.swap(newProfile);
  Backend::getInstance().transform(*this, *(profile_->getParam()));
  if (Session::getSession())
    Session::getSession()->saveEntity(*this);
}

template <typename T> Compuon<T> &Compuon<T>::operator=(const T &scalar) {
  LOG_MESSAGE("Assignment of scalar");
  if (Session::getSession()->isActive()) {
    Session::getSession()->saveEntity(*this);

    auto op1_ptr = Session::getSession()->trackEntity(*this);
    auto tmp = std::make_shared<Compuon<T>>(scalar);
    tmp->belong(op1_ptr->getProfile());

    Session::getSession()->saveOp(
      std::make_shared<scheduler::Operation<T>>(scheduler::OperationType::Assignment, op1_ptr, tmp));

    return *op1_ptr;
  } else {
    this->setValue(scalar);
    return *this;
  }
}

template <typename T> Compuon<T> &Compuon<T>::operator=(const Compuon<T> &other) {
  LOG_MESSAGE("(TODO) Copy Assignment (" << this->getValue() << ", " << other.getValue() << ")");
#if 0
    if (this == &other)
        return *this;

    val_ = other.val_;
    profile_ = other.profile_;

    return *this;
#endif
  if (Session::getSession()->isActive()) {
    std::cout << "other: " << &other << " " << Session::getSession()->getEntity<T>(&other) << std::endl;
    auto op1_ptr = Session::getSession()->trackEntity(*this);
    auto op2_ptr = Session::getSession()->trackEntity(const_cast<Compuon<T> &>(other));

    Session::getSession()->saveOp(
      std::make_shared<scheduler::Operation<T>>(scheduler::OperationType::Assignment, op1_ptr, op2_ptr));

    LOG_MESSAGE("========");

    return *this;

    return *this;
  } else {
    this->setValue(other.getValue());

    return *this;
  }
}

template <typename T> Compuon<T> &Compuon<T>::operator=(Compuon &&other) noexcept {
  LOG_MESSAGE("Move Assignment (" << this->getValue() << ", " << other.getValue() << ")");

  // self assignment
  if (this == &other)
    return *this;

  if (Session::getSession()->isActive()) {
    LOG_MESSAGE("(Move) Assignment on session");
    std::cout << "other: " << &other << " " << Session::getSession()->getEntity<T>(&other) << std::endl;
    auto op1_ptr = Session::getSession()->trackEntity(*this);
    auto op2_ptr = Session::getSession()->trackEntity(other);

    Session::getSession()->saveOp(
      std::make_shared<scheduler::Operation<T>>(scheduler::OperationType::Assignment, op1_ptr, op2_ptr));

    LOG_MESSAGE("========");

    return *this;
  } else {
    LOG_MESSAGE("(Move) Assignment without session");
    this->setValue(std::move(other.getValue()));
    return *this;
  }
}

template <typename T> Compuon<T> Compuon<T>::operator+(const T &scalar) const {
  LOG_MESSAGE("=====Operator Add of Compuon with scalar=====");
  auto entity = std::make_shared<Compuon<T>>(scalar);
  entity->setScalar();
  entity->belong(Profile::getProfile());
  LOG_MESSAGE("=============================");
  return *this + *entity;
}

template <typename T> Compuon<T> Compuon<T>::operator+(const Compuon<T> &other) const {
  LOG_MESSAGE("=====Operator Add of Compuon=====");

  std::shared_ptr<Compuon<T>> result = nullptr;

  if (Session::getSession()->isActive()) {
    LOG_MESSAGE("Add on session (" << this->getValue() << ", " << other.getValue() << ")");

    // Track operands and result in the Session
    auto op1_ptr = Session::getSession()->trackEntity(const_cast<Compuon<T> &>(*this));
    auto op2_ptr = Session::getSession()->trackEntity(const_cast<Compuon<T> &>(other));
    auto result_ptr = std::make_shared<Compuon<T>>(1234);
    Session::getSession()->saveEntity(*result_ptr);
    Session::getSession()->trackEntity(*result_ptr);

    // Save the add operation for later execution
    Session::getSession()->saveOp(
      std::make_shared<scheduler::Operation<T>>(scheduler::OperationType::Add, op1_ptr, op2_ptr, result_ptr));

    result = result_ptr;
  } else {
    LOG_MESSAGE("Add without session");
    auto result_add =
      Backend::getInstance().add(static_cast<const CompuonBase &>(*this), static_cast<const CompuonBase &>(other));

    if (!result_add) {
      throw std::runtime_error("Backend::add returned nullptr");
    }

    result = std::dynamic_pointer_cast<Compuon<T>>(result_add);

    if (!result) {
      throw std::runtime_error("Backend::add returned nullptr");
    }
  }

  LOG_MESSAGE("=============================");
  return *result;
}

template <typename T> Compuon<T> Compuon<T>::operator*(const Compuon<T> &other) const {
  LOG_MESSAGE("=====Operator Multiply of Compuon=====");
  std::shared_ptr<Compuon<T>> result = nullptr;

  if (Session::getSession()->isActive()) {
    LOG_MESSAGE("Multiplication on session");

    // Track operands and result in the Session
    auto op1_ptr = Session::getSession()->trackEntity(const_cast<Compuon<T> &>(*this));
    auto op2_ptr = Session::getSession()->trackEntity(const_cast<Compuon<T> &>(other));
    auto result_ptr = std::make_shared<Compuon<T>>(1234);
    Session::getSession()->saveEntity(*result_ptr);
    Session::getSession()->trackEntity(*result_ptr);

    Session::getSession()->saveOp(
      std::make_shared<scheduler::Operation<T>>(scheduler::OperationType::Multiply, op1_ptr, op2_ptr, result_ptr));
    result = result_ptr;
  } else {
    LOG_MESSAGE("Multiplication without session");
    LOG_MESSAGE(this->getValue() << " x " << other.getValue());
    auto result_mul =
      Backend::getInstance().multiply(static_cast<const CompuonBase &>(*this), static_cast<const CompuonBase &>(other));

    if (!result_mul) {
      throw std::runtime_error("Backend::add returned nullptr");
    }

    result = std::dynamic_pointer_cast<Compuon<T>>(result_mul);

    if (!result) {
      throw std::runtime_error("Backend::add returned nullptr");
    }
  }
  LOG_MESSAGE("=============================");
  return *result;
}

template <typename T> Compuon<T> Compuon<T>::operator*(const T &scalar) const {
  auto entity = std::make_shared<Compuon<T>>(scalar);
  entity->setScalar();
  entity->belong(Profile::getProfile());
  return *this * *entity;
}

template <typename T> T Compuon<T>::decrypt() const { return std::any_cast<T>(Backend::getInstance().decrypt(*this)); }

template class Compuon<int>;

} // namespace fhenomenon
