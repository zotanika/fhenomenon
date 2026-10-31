#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace fhenomenon::ckks {

// Exact mixed-radix reconstruction; separate from approximate BConvTables.
// Construction allocates all tables; conversion and reconstruction allocate nothing.
class CrtTables {
  public:
  // Distinct primes in [2, 2^62), not necessarily NTT-friendly.
  static std::optional<CrtTables> create(std::vector<uint64_t> moduli, std::string *error = nullptr);
  const std::vector<uint64_t> &moduli() const { return moduli_; }
  // Fully reduced residues are required. Exact in-place conversion is allowed;
  // partial overlap is rejected. Invalid input leaves digits unchanged.
  bool toMixedRadix(const uint64_t *residues, uint64_t *digits) const;
  // Remaining methods require valid mixed-radix digits (digit[i] < moduli[i]).
  // A representative above floor(Q/2) is negative. For even Q, Q/2 is positive.
  bool isNegative(const uint64_t *digits) const;
  // Exact centered residue, also for composite target moduli. Zero modulus
  // returns zero rather than dividing by zero; it is not a usable residue.
  uint64_t centeredMod(const uint64_t *digits, uint64_t modulus) const;
  // Floating approximation of the signed integer, possibly infinite if the
  // basis exceeds long double's range. Small negatives avoid Q - x cancellation.
  long double centeredValue(const uint64_t *digits) const;

  private:
  CrtTables() = default;
  std::vector<uint64_t> moduli_;
  std::vector<uint64_t> inverses_;
  std::vector<uint64_t> half_;
};

} // namespace fhenomenon::ckks
