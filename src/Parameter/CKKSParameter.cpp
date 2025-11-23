#include "Parameter/CKKSParameter.h"
#include "Utils/log.h"

namespace fhenomenon {

void CKKSParameter::setParams([[maybe_unused]] CKKSParamPreset preset) {
  switch (preset) {
  case CKKSParamPreset::FGb:
    // Fast Galois (FGb) preset: Good balance between security and performance
    // poly_modulus_degree = 8192, coeff_modulus = [60, 40, 40, 60]
    setKeySize(128); // Security level
    setDegree(8192); // Polynomial modulus degree
    LOG_MESSAGE("CKKSParameter: Set FGb preset (poly_modulus_degree=8192, security=128-bit)");
    break;
  default:
    throw std::runtime_error("CKKSParameter: Unknown preset");
  }
}

} // namespace fhenomenon
