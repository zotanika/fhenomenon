#pragma once

#include "Compuon.h"

#include <optional>

namespace fhenomenon {

// Forward declaration of Compuon class to avoid circular dependency
template <typename T> class Compuon;

namespace scheduler {

enum class OperationType {
  // (XXX) Do we need to distinguish copy / move assignment?
  Assignment,
  Refresh,
  LeftRotate,
  RightRotate,
  Conjugate,
  Add,
  Sub,
  Multiply
};

class OperationBase {
  public:
  virtual ~OperationBase() = default;
  virtual void execute() = 0;
  virtual OperationType getType() const = 0;
};

template <typename T> class Operation final : public OperationBase {
  public:
  Operation(OperationType type, std::shared_ptr<Compuon<T>> op1, std::shared_ptr<Compuon<T>> op2,
            std::shared_ptr<Compuon<T>> tmp = nullptr)
    : type_(type), operand1_(op1), operand2_(op2), result_(tmp) {}

  void execute() override;

  OperationType getType() const override { return type_; }
  std::shared_ptr<Compuon<T>> getOperand1() const {return operand1_;}
  std::shared_ptr<Compuon<T>> getOperand2() const {return operand2_;}
  std::shared_ptr<Compuon<T>> getResult() const { return result_; }

  private:
  OperationType type_;
  std::shared_ptr<Compuon<T>> operand1_;
  std::shared_ptr<Compuon<T>> operand2_;
  std::shared_ptr<Compuon<T>> result_;
};

} // namespace scheduler
} // namespace fhenomenon
