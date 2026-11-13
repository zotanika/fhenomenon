#pragma once
#include "engine.hpp"
#include <complex>
#include <vector>

namespace fhenomenon::tensor_ckks {
using ComplexMatrix = std::vector<std::vector<std::complex<double>>>;

// Native canonical slots evaluate at exp(pi*i*5^j/(2*M)). Coefficient
// slot k stores f[k]+i*f[k+M] in the degree-2M invariant subring.
ComplexMatrix coefficientToSlotMatrix(unsigned period);
ComplexMatrix slotToCoefficientMatrix(unsigned period);

// One level for public diagonal multiplication, with baby/giant rotations.
// These internal transforms operate on slot-valued ciphertexts; the caller
// assigns any external semantic representation tag only at its boundary.
Value linearTransform(Engine &, const Value &, const ComplexMatrix &, double multiplier = 1.0);
Value coefficientToSlot(Engine &, const Value &, double multiplier = 1.0);
Value slotToCoefficient(Engine &, const Value &, double multiplier = 1.0);

// Unnormalized relative trace, using PHYSICAL rotations M,2M,...,N/4.
// For r=N/(2M), it keeps coefficient indices divisible by r and multiplies
// them by r. In particular T(integer polynomial)/r remains integral.
Value traceToPeriod(Engine &, const Value &);
} // namespace fhenomenon::tensor_ckks
