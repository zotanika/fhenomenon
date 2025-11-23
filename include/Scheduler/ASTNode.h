#pragma once
#include "Scheduler/Operation.h"

namespace fhenomenon {
namespace scheduler {

class ASTNode {
  public:
  virtual ~ASTNode() = default;
  virtual void evaluate() = 0;
  virtual void print(int depth = 0) const = 0;
};

template <typename T> class OperatorNode : public ASTNode {
  public:
  OperatorNode(std::shared_ptr<Operation<T>> operation, OperationType type, std::shared_ptr<ASTNode> left,
               std::shared_ptr<ASTNode> right, std::shared_ptr<Compuon<T>> result)
    : operation_(operation), type_(type), left_(left), right_(right), result_(result), evaluated_(false) {}

  OperationType getType() const { return type_; }
  std::shared_ptr<ASTNode> getRight() const { return right_; }
  std::shared_ptr<ASTNode> getLeft() const { return left_; }
  std::shared_ptr<Compuon<T>> getResult() const { return result_; }

  void evaluate() override {
    if (evaluated_)
      return;
    if (left_)
      left_->evaluate();
    if (right_)
      right_->evaluate();
    operation_->execute();
    evaluated_ = true;
  }

  void print(int depth = 0) const override {
    std::string indent(static_cast<std::string::size_type>(depth * 2), ' ');
    std::cout << indent << "Operation: "
              << (type_ == OperationType::Add        ? "ADD"
                  : type_ == OperationType::Multiply ? "Multiply"
                                                     : "Assignment")
              << std::endl;
    if (left_)
      left_->print(depth + 1);
    if (right_)
      right_->print(depth + 1);
  }

  private:
  std::shared_ptr<Operation<T>> operation_;
  OperationType type_;
  std::shared_ptr<ASTNode> left_;
  std::shared_ptr<ASTNode> right_;
  std::shared_ptr<Compuon<T>> result_;
  bool evaluated_;
};

template <typename T> class OperandNode : public ASTNode {
  public:
  explicit OperandNode(std::shared_ptr<Compuon<T>> entity) : entity_(entity), evaluated_(false) {}

  std::shared_ptr<Compuon<T>> getEntity() const { return entity_; }

  void evaluate() override {
    if (evaluated_)
      return;
    value_ = entity_->getValue();
    evaluated_ = true;
  }

  void print(int depth) const override {
    std::string indent(static_cast<std::string::size_type>(depth * 2), ' ');
    std::cout << indent << "Operand: " << entity_->getValue() << std::endl;
  }

  private:
  std::shared_ptr<Compuon<T>> entity_;
  T value_;
  bool evaluated_;
};

} // namespace scheduler
} // namespace fhenomenon
