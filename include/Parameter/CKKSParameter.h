#pragma once

#include "Parameter/Parameter.h"

namespace fhenomenon {

class CKKSParameter : public Parameter {
  public:
  CKKSParameter(CKKSParamPreset preset) {
    init();
    setParams(preset);
  }

  CKKSParameter(int size, int degree) {
    setKeySize(size);
    setDegree(degree);
  }

  void printParams() const override { std::cout << "(TODO) CKKS Parameters" << std::endl; }

  void init() override { std::cout << "(TODO) Initialize CKKS parameters" << std::endl; }

  private:
  void setParams(CKKSParamPreset preset);
};

} // namespace fhenomenon
