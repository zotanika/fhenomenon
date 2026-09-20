#pragma once

#include <cstddef>
#include <cstdint>

// L2 — the durable owner of residue words.
//
// A slab owns a fixed number of 64-bit words and knows nothing else: not how
// many limbs they form, not which primes, not how many polynomials. Shape is
// the layout's business (Layout.h) and binding the two together is the
// handle's (RnsPoly.h, Ciphertext.h). Keeping the owner shapeless is what
// lets the same bytes be reinterpreted as a ciphertext shrinks with level or
// grows by a polynomial, without any memory moving.
//
// Invariant I2 as this layer reads it: the size is an explicit constructor
// argument, the slab never grows, there is no default size and no allocator
// to configure. That is the same reading under which Arena (L3) satisfies I2
// — what the invariant forbids is hidden policy about how much and when, not
// the act of calling operator new.
//
// Contents are uninitialised, as an Arena allocation is. A slab is written
// by whoever produces the residues it holds, and zeroing a multi-megabyte
// buffer on every allocation would be exactly the kind of cost a caller did
// not ask for and cannot switch off.

namespace fhenomenon::ckks {

class PolySlab {
  public:
  // Cache-line alignment, promised to every kernel that reads a limb. The
  // same number as Arena::kAlignment, stated independently because L2 does
  // not see L3; Layout.h's minimum degree guarantees every limb is a whole
  // number of these lines, so limb pointers inherit the alignment for free.
  static constexpr std::size_t kAlignment = 64;

  // Owns nothing. data() is nullptr and words() is 0.
  PolySlab() = default;

  // Owns exactly `words` residues. A request of zero words owns nothing and
  // is indistinguishable from the default.
  explicit PolySlab(std::size_t words);

  PolySlab(const PolySlab &) = delete;
  PolySlab &operator=(const PolySlab &) = delete;

  // Move leaves the source owning nothing, so a moved-from slab cannot hand
  // out a pointer into memory it no longer owns.
  PolySlab(PolySlab &&other) noexcept;
  PolySlab &operator=(PolySlab &&other) noexcept;
  ~PolySlab();

  // Duplication is a distinctly named operation (invariant I4): a copy
  // constructor would let an allocation hide behind an `=`. Copies every
  // word, including any the caller has not written.
  PolySlab clone() const;

  uint64_t *data() { return data_; }
  const uint64_t *data() const { return data_; }
  std::size_t words() const { return words_; }
  std::size_t bytes() const { return words_ * sizeof(uint64_t); }
  bool empty() const { return words_ == 0; }

  private:
  uint64_t *data_ = nullptr;
  std::size_t words_ = 0;
};

} // namespace fhenomenon::ckks
