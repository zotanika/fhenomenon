#pragma once
#include "CKKS/Storage.h"
#include <complex>
#include <cstddef>
#include <cstdint>
namespace fhenomenon::ckks {
class Arena;
class Params;
class EvaluationTables;
namespace encoding {
// Canonical CKKS packing at exponents 5^j modulo 2N. A shorter
// power-of-two vector repeats periodically across all N/2 physical slots.
// The caller supplies a valid Q-prefix NTT output layout. Encoding rounds
// scaled coefficients to int64_t and rejects nonfinite input, int64 overflow,
// or rounded coefficients outside the centered active-Q range [-floor(Q/2), floor(Q/2)].
bool encode(RnsPolyView &, const double *values, uint32_t count, double scale, const EvaluationTables &, Arena &);
bool encode(RnsPolyView &, const std::complex<double> *values, uint32_t count, double scale, const EvaluationTables &,
            Arena &);
// Reconstructs centered coefficients from every active Q limb using exact
// mixed-radix CRT, then returns the requested leading physical slots. The real overload returns
// their real projection.
bool decode(double *values, uint32_t count, ConstRnsPolyView, double scale, const EvaluationTables &, Arena &);
bool decode(std::complex<double> *values, uint32_t count, ConstRnsPolyView, double scale, const EvaluationTables &,
            Arena &);
std::size_t scratchBytes(const Params &) noexcept;
} // namespace encoding
} // namespace fhenomenon::ckks
