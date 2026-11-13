#pragma once

#include "engine.hpp"
#include <complex>
#include <vector>

namespace fhenomenon::tensor_ckks {

// Deterministic dense ternary secret bound: |I_k| <= N/2+2 after a
// centered one-prime lift. No sparse-secret or probabilistic wrap assumption.
unsigned refreshSquarings(unsigned log_degree);
unsigned refreshDepth(unsigned log_degree);
// Largest power-of-two amplification admitted by B*S[0]/q0 <= 1/16.
// Depends exclusively on public parameters and B; consumes no levels.
std::int64_t refreshGain(const Engine &, double public_bound);

// Coefficients in ascending monomial order. Exposed for independent clear
// polynomial verification; encrypted refresh evaluates these same polynomials.
std::vector<std::complex<double>> refreshExponentialCoefficients();
std::vector<std::complex<double>> refreshArcsineCoefficients();

// True encrypted modular refresh. The source is immutable, and only the prefix
// ending at target+refreshDepth(logN) is lifted and evaluated. public_bound is a
// public precondition on complex slot magnitude, not a measured secret range.
// The fractional domain is public_bound*S[0]/q0 <= 1/16. No decryption occurs.
Value refresh(Engine &, const Value &, unsigned target, double public_bound = 1.0);

} // namespace fhenomenon::tensor_ckks
