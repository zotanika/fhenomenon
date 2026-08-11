#include "CKKS/Params.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace fhenomenon::ckks {

namespace {

// Sum of log2 of a half-open prime range. Used instead of a big-integer
// product because every check here is a magnitude comparison, and the primes
// are bounded well inside double's 53-bit mantissa.
double log2Product(const std::vector<uint64_t> &primes, uint32_t begin, uint32_t end) {
  double bits = 0.0;
  for (uint32_t i = begin; i < end; ++i) {
    bits += std::log2(static_cast<double>(primes[i]));
  }
  return bits;
}

bool fail(std::string *error, std::string reason) {
  if (error != nullptr) {
    *error = std::move(reason);
  }
  return false;
}

// Every prime must be NTT-friendly for this ring: q = 1 (mod 2N) is what makes
// a primitive 2N-th root of unity exist mod q, which the negacyclic transform
// at L1 needs. A typo'd prime that fails this would otherwise surface as
// silent garbage after the first multiplication.
bool checkPrimeList(const std::vector<uint64_t> &primes, const char *what, uint64_t two_n,
                    std::unordered_set<uint64_t> &seen, std::string *error) {
  for (uint64_t p : primes) {
    if (p < 2) {
      return fail(error, std::string(what) + ": " + std::to_string(p) + " is not a modulus");
    }
    if (p >> Params::kMaxPrimeBits != 0) {
      return fail(error, std::string(what) + ": " + std::to_string(p) + " exceeds " +
                           std::to_string(Params::kMaxPrimeBits) + " bits");
    }
    if (p % two_n != 1) {
      return fail(error, std::string(what) + ": " + std::to_string(p) + " is not 1 mod 2N (" + std::to_string(two_n) +
                           "), so it admits no primitive 2N-th root of unity");
    }
    if (!seen.insert(p).second) {
      return fail(error, std::string(what) + ": " + std::to_string(p) + " appears more than once");
    }
  }
  return true;
}

void hashCombine(std::size_t &seed, std::size_t value) {
  seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

} // namespace

std::optional<Params> Params::create(Spec spec, std::string *error) {
  if (spec.log_degree < kMinLogDegree || spec.log_degree > kMaxLogDegree) {
    fail(error, "log_degree " + std::to_string(spec.log_degree) + " outside [" + std::to_string(kMinLogDegree) + ", " +
                  std::to_string(kMaxLogDegree) + "]");
    return std::nullopt;
  }
  if (spec.main_primes.empty()) {
    fail(error, "main_primes is empty: a parameter set needs at least one level");
    return std::nullopt;
  }
  if (spec.aux_primes.empty()) {
    fail(error, "aux_primes is empty: hybrid key switching needs at least one special prime");
    return std::nullopt;
  }

  const uint64_t two_n = uint64_t{2} << spec.log_degree;
  std::unordered_set<uint64_t> seen;
  if (!checkPrimeList(spec.main_primes, "main_primes", two_n, seen, error) ||
      !checkPrimeList(spec.aux_primes, "aux_primes", two_n, seen, error) ||
      !checkPrimeList(spec.terminal_primes, "terminal_primes", two_n, seen, error)) {
    return std::nullopt;
  }

  const uint32_t levels = static_cast<uint32_t>(spec.main_primes.size());
  if (spec.dnum == 0 || spec.dnum > levels) {
    fail(error, "dnum " + std::to_string(spec.dnum) + " outside [1, " + std::to_string(levels) +
                  "]: it is the number of key-switching digits the main chain is split into");
    return std::nullopt;
  }

  if (spec.log_default_scale == 0 || spec.log_default_scale >= kMaxPrimeBits) {
    fail(error, "log_default_scale " + std::to_string(spec.log_default_scale) + " outside (0, " +
                  std::to_string(kMaxPrimeBits) + ")");
    return std::nullopt;
  }

  if (spec.secret == SecretDistribution::SparseTernary) {
    if (spec.sparse_hamming_weight == 0 || spec.sparse_hamming_weight >= (uint32_t{1} << spec.log_degree)) {
      fail(error, "sparse_hamming_weight " + std::to_string(spec.sparse_hamming_weight) +
                    " must be in (0, N) for a sparse ternary secret");
      return std::nullopt;
    }
  }

  // Hybrid key switching raises a digit to P * (that digit's primes) before
  // coming back down, so P must exceed the largest digit. dnum == 1 makes the
  // single digit the whole chain, which is why plain GHS needs a P as large as
  // Q and why this check rejects it unless the caller actually supplied one.
  Params params(std::move(spec));
  const std::vector<uint64_t> &aux = params.spec_.aux_primes;
  const double aux_bits = log2Product(aux, 0, static_cast<uint32_t>(aux.size()));
  double widest_digit_bits = 0.0;
  uint32_t widest_digit = 0;
  for (uint32_t digit = 0; digit < params.dnum(); ++digit) {
    const auto range = params.digitRange(digit);
    const double bits = log2Product(params.spec_.main_primes, range.first, range.second);
    if (bits > widest_digit_bits) {
      widest_digit_bits = bits;
      widest_digit = digit;
    }
  }
  if (aux_bits <= widest_digit_bits) {
    fail(error, "aux_primes carry " + std::to_string(aux_bits) + " bits but digit " + std::to_string(widest_digit) +
                  " of " + std::to_string(params.dnum()) + " carries " + std::to_string(widest_digit_bits) +
                  " bits: P must exceed the widest key-switching digit (raise dnum, or add special primes)");
    return std::nullopt;
  }

  if (error != nullptr) {
    error->clear();
  }
  return params;
}

uint32_t Params::alpha() const {
  const uint32_t levels = levelCount();
  return (levels + spec_.dnum - 1) / spec_.dnum;
}

std::pair<uint32_t, uint32_t> Params::digitRange(uint32_t digit) const {
  if (digit >= spec_.dnum) {
    return {0, 0};
  }
  const uint32_t levels = levelCount();
  const uint32_t width = alpha();
  const uint32_t begin = std::min(digit * width, levels);
  const uint32_t end = std::min(begin + width, levels);
  return {begin, end};
}

uint64_t Params::ciphertextBytes(int64_t level, uint32_t num_polys) const {
  if (level < 0 || level > freshLevel() || num_polys == 0) {
    return 0;
  }
  const uint64_t primes = static_cast<uint64_t>(level) + 1;
  return num_polys * primes * degree() * sizeof(uint64_t);
}

bool Params::operator==(const Params &other) const {
  return spec_.log_degree == other.spec_.log_degree && spec_.main_primes == other.spec_.main_primes &&
         spec_.aux_primes == other.spec_.aux_primes && spec_.terminal_primes == other.spec_.terminal_primes &&
         spec_.dnum == other.spec_.dnum && spec_.log_default_scale == other.spec_.log_default_scale &&
         spec_.secret == other.spec_.secret && spec_.sparse_hamming_weight == other.spec_.sparse_hamming_weight;
}

std::size_t Params::hash() const {
  std::size_t seed = 0;
  hashCombine(seed, std::hash<uint32_t>{}(spec_.log_degree));
  hashCombine(seed, std::hash<uint32_t>{}(spec_.dnum));
  hashCombine(seed, std::hash<uint32_t>{}(spec_.log_default_scale));
  hashCombine(seed, std::hash<uint32_t>{}(static_cast<uint32_t>(spec_.secret)));
  hashCombine(seed, std::hash<uint32_t>{}(spec_.sparse_hamming_weight));
  for (const std::vector<uint64_t> *list : {&spec_.main_primes, &spec_.aux_primes, &spec_.terminal_primes}) {
    hashCombine(seed, std::hash<std::size_t>{}(list->size()));
    for (uint64_t p : *list) {
      hashCombine(seed, std::hash<uint64_t>{}(p));
    }
  }
  return seed;
}

} // namespace fhenomenon::ckks
