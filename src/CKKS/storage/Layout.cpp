#include "CKKS/Layout.h"

#include "CKKS/Params.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace fhenomenon::ckks {

namespace {

bool fail(std::string *error, std::string reason) {
  if (error != nullptr) {
    *error = std::move(reason);
  }
  return false;
}

// FNV-1a over bytes. Chosen for being short enough to read and verify in one
// sitting, and for having a fixed published definition — the fingerprint is a
// compatibility surface, so an implementation whose behaviour could drift with
// a library version would defeat the point of not using std::hash.
constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

void mixByte(uint64_t &state, uint8_t byte) {
  state ^= uint64_t{byte};
  state *= kFnvPrime;
}

void mixWord(uint64_t &state, uint64_t value) {
  for (uint32_t i = 0; i < 8; ++i) {
    mixByte(state, static_cast<uint8_t>((value >> (i * 8U)) & 0xFFU));
  }
}

// Length-prefixed, so that {1, 2} and {1} followed by {2} cannot collide. A
// concatenation-based fingerprint over several vectors has exactly that hole.
void mixWords(uint64_t &state, const std::vector<uint64_t> &values) {
  mixWord(state, values.size());
  for (uint64_t value : values) {
    mixWord(state, value);
  }
}

} // namespace

// The field order below is a wire-compatibility surface and may never change.
// It covers every field Params::operator== compares, in the order that
// operator== lists them; if a field is added to Params::Spec, it must be added
// here too, at the end.
ParamsId ParamsId::of(const Params &params) {
  uint64_t state = kFnvOffset;
  mixWord(state, params.logDegree());
  mixWords(state, params.mainPrimes());
  mixWords(state, params.auxPrimes());
  mixWords(state, params.terminalPrimes());
  mixWord(state, params.dnum());
  mixWord(state, params.logDefaultScale());
  mixWord(state, static_cast<uint64_t>(params.secret()));
  mixWord(state, params.sparseHammingWeight());
  // Zero is reserved for "unidentified", which RnsBasis::wellFormed rejects.
  // Folding it to 1 costs one value out of 2^64 and removes a case where a
  // legitimate parameter set would be indistinguishable from a default-
  // constructed basis.
  return ParamsId{state == 0 ? 1U : state};
}

bool RnsBasis::wellFormed(std::string *error) const {
  if (params_id.empty()) {
    return fail(error, "basis names no params: an unidentified basis compares equal to every other "
                       "unidentified basis, so residues under one parameter set would be accepted under another");
  }
  if (main_end < main_begin) {
    return fail(error, "main range [" + std::to_string(main_begin) + ", " + std::to_string(main_end) + ") is inverted");
  }
  if (aux_end < aux_begin) {
    return fail(error, "aux range [" + std::to_string(aux_begin) + ", " + std::to_string(aux_end) + ") is inverted");
  }
  if (limbCount() == 0) {
    return fail(error, "basis has no limbs");
  }
  if (limbCount() > kMaxLimbs) {
    return fail(error,
                "basis has " + std::to_string(limbCount()) + " limbs, over the bound of " + std::to_string(kMaxLimbs));
  }
  return true;
}

bool RnsBasis::operator==(const RnsBasis &other) const {
  return params_id == other.params_id && main_begin == other.main_begin && main_end == other.main_end &&
         aux_begin == other.aux_begin && aux_end == other.aux_end;
}

bool PolyLayout::wellFormed(std::string *error) const {
  if (!basis.wellFormed(error)) {
    return false;
  }
  if (log_degree < kMinLogDegree || log_degree > kMaxLogDegree) {
    return fail(error, "log_degree " + std::to_string(log_degree) + " outside [" + std::to_string(kMinLogDegree) +
                         ", " + std::to_string(kMaxLogDegree) + "]");
  }
  return true;
}

bool PolyLayout::operator==(const PolyLayout &other) const {
  return basis == other.basis && log_degree == other.log_degree && form == other.form;
}

bool CiphertextLayout::wellFormed(std::string *error) const {
  if (!poly.wellFormed(error)) {
    return false;
  }
  if (num_polys == 0 || num_polys > kMaxPolys) {
    return fail(error, "num_polys " + std::to_string(num_polys) + " outside [1, " + std::to_string(kMaxPolys) + "]");
  }
  if (!poly.basis.isChainPrefix()) {
    return fail(error, "a ciphertext basis must be a chain prefix, but this one starts at main index " +
                         std::to_string(poly.basis.main_begin) +
                         "; level() is main_end - 1 and means nothing "
                         "otherwise, so a key-switching digit is a "
                         "PolyLayout and never a CiphertextLayout");
  }
  if (!std::isfinite(scale) || scale <= 0.0) {
    return fail(error, "scale must be finite and positive; decryption divides by it");
  }
  return true;
}

bool CiphertextLayout::operator==(const CiphertextLayout &other) const {
  return poly == other.poly && num_polys == other.num_polys && scale == other.scale;
}

} // namespace fhenomenon::ckks
