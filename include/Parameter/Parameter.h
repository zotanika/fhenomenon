#pragma once

#include "Common.h"
#include "ParameterPreset.h"

namespace fhenomenon {

class Parameter {
  public:
  virtual ~Parameter() = default;

  virtual void printParams() const = 0;
  virtual void init() = 0;

  int getKeySize() const { return key_size_; }
  int getDegree() const { return degree_; }

  protected:
  // Zero-initialized: init() reads these to decide whether defaults are
  // needed, and an uninitialized read is UB that Apple clang compiles into
  // a trap at -O2 (the macOS-only SIGTRAP in every test constructing a
  // CKKSParameter).
  int key_size_ = 0;
  int degree_ = 0;

  void setKeySize(int size) { key_size_ = size; }
  void setDegree(int degree) { degree_ = degree; }
};

} // namespace fhenomenon
