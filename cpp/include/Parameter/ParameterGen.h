#pragma once

#include "Parameter/CKKSParameter.h"

namespace fhenomenon {

class ParameterGen {
  public:
  // Static factory methods
  static std::shared_ptr<Parameter> createCKKSParam(CKKSParamPreset preset) {
    return std::make_shared<CKKSParameter>(preset);
  }
};

} // namespace fhenomenon
