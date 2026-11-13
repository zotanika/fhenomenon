#pragma once

#include "Fhenon.h"

#include <optional>

namespace fhenomenon {

// Forward declaration of Fhenon class to avoid circular dependency
template <typename T> class Fhenon;
class Backend; // Forward declaration

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
  Operation(OperationType type, std::shared_ptr<Fhenon<T>> op1, std::shared_ptr<Fhenon<T>> op2,
            std::shared_ptr<Fhenon<T>> tmp = nullptr, const Backend *backend_delegate = nullptr, int64_t param = 0)
    : type_(type), operand1_(op1), operand2_(op2), result_(tmp), backend_delegate_(backend_delegate), param_(param) {}

  void execute() override;

  // Set backend delegate (can be set later if not provided in constructor)
  void setBackendDelegate(const Backend *backend) { backend_delegate_ = backend; }

  OperationType getType() const override { return type_; }
  std::shared_ptr<Fhenon<T>> getOperand1() const { return operand1_; }
  std::shared_ptr<Fhenon<T>> getOperand2() const { return operand2_; }
  std::shared_ptr<Fhenon<T>> getResult() const { return result_; }

  // Operation-specific integer parameter (e.g. rotation distance for
  // LeftRotate/RightRotate). Mirrors FhnInstruction::params[0].
  int64_t getParam() const { return param_; }
  void setParam(int64_t param) { param_ = param; }

  private:
  OperationType type_;
  std::shared_ptr<Fhenon<T>> operand1_;
  std::shared_ptr<Fhenon<T>> operand2_;
  std::shared_ptr<Fhenon<T>> result_;
  const Backend *backend_delegate_; // Backend delegate for executing operations
  int64_t param_ = 0;
};

} // namespace scheduler
} // namespace fhenomenon
