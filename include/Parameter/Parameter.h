#pragma once

#include "Common.h"
#include "ParameterPreset.h"

namespace fhenomenon {

class Parameter {
  public:
  virtual ~Parameter() = default;

  virtual void printParams() const = 0;
  virtual void init() = 0;

  protected:
  int key_size_;
  int degree_;

  void setKeySize(int size) { key_size_ = size; }
  void setDegree(int degree) { degree_ = degree; }
};

} // namespace fhenomenon
