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

  void printParams() const override {
    std::cout << "CKKS Parameters:" << std::endl;
    std::cout << "  Key Size: " << key_size_ << " bits" << std::endl;
    std::cout << "  Polynomial Modulus Degree: " << degree_ << std::endl;
  }

  void init() override {
    // Initialize default values if not set
    if (key_size_ == 0) {
      setKeySize(128);
    }
    if (degree_ == 0) {
      setDegree(8192);
    }
  }

  private:
  void setParams(CKKSParamPreset preset);
};

} // namespace fhenomenon
