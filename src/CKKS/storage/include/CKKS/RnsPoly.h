#pragma once

#include "CKKS/Layout.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

// L2 — a polynomial layout bound to the residues it describes.
//
// BasicRnsPoly is a non-owning handle: a PolyLayout and a pointer, nothing
// more. It never allocates, never frees, and is trivially copyable, so it is
// the operand type an L4 kernel takes whether the residues sit in a
// Ciphertext's slab, in an Arena allocation, or in a buffer that arrived over
// the wire. That is the point of splitting the handle from the owner: the
// kernel cannot tell, and therefore cannot care, which one it was given.
//
// The template parameter is uint64_t or const uint64_t and nothing else, and
// the two spellings are the read/write split from invariant I4 made visible
// in the signature: RnsPolyView cannot be written through, RnsPolyRef can,
// and a Ref converts to a View implicitly, never the other way.
//
// Residues are limb-major: limb i occupies degree() contiguous words starting
// at data() + i * degree(). Main limbs come first in basis order, then aux
// limbs. Contiguity per limb is what the L1 transforms need — ntt::forward
// takes one prime's residues as a single span — and it is why a slice of the
// main range is a view rather than a copy.
//
// Two kinds of function live here and they follow one rule. Functions that
// address WITHIN a valid layout — limb(), mainLimb(), auxLimb() — are
// unchecked, because they are what an inner loop calls and the layout they
// address was validated when it was bound. Functions that produce a NEW
// layout — mainSlice() — validate it and refuse rather than hand back a
// handle over the wrong words.

namespace fhenomenon::ckks {

template <typename T> class BasicRnsPoly {
  static_assert(std::is_same_v<std::remove_const_t<T>, uint64_t>, "BasicRnsPoly binds uint64_t residues only");

  public:
  using value_type = T;

  // Binds nothing: layout default, data nullptr.
  constexpr BasicRnsPoly() = default;

  // Precondition: `data` addresses at least layout.wordCount() words that
  // stay valid for as long as the handle is used. Unchecked here because
  // there is nothing to check against — the owner that produced `data` is
  // where the size was validated.
  constexpr BasicRnsPoly(const PolyLayout &layout, T *data) : layout_(layout), data_(data) {}

  // Ref -> View. Enabled only when T is the const flavour and U the mutable
  // one, so the copy constructor still handles same-type copies.
  template <typename U, typename = std::enable_if_t<std::is_const_v<T> && std::is_same_v<std::remove_const_t<T>, U>>>
  constexpr BasicRnsPoly(const BasicRnsPoly<U> &other) : layout_(other.layout()), data_(other.data()) {}

  constexpr const PolyLayout &layout() const { return layout_; }
  constexpr const RnsBasis &basis() const { return layout_.basis; }
  constexpr T *data() const { return data_; }
  constexpr std::size_t words() const { return layout_.wordCount(); }
  constexpr uint64_t degree() const { return layout_.degree(); }
  constexpr uint32_t limbCount() const { return layout_.basis.limbCount(); }
  constexpr NttForm form() const { return layout_.form; }
  constexpr bool empty() const { return data_ == nullptr; }

  // Limb `i` in storage order, i < limbCount(). Main limbs first, then aux.
  constexpr T *limb(uint32_t i) const { return data_ + (std::size_t{i} << layout_.log_degree); }

  // The same limbs addressed by the prime index the basis names, which is
  // what key switching speaks: a digit at main [b, e) has mainLimb(b) as its
  // first limb, not limb(b). Preconditions: main_begin <= index < main_end,
  // respectively aux_begin <= index < aux_end.
  constexpr T *mainLimb(uint32_t index) const { return limb(index - layout_.basis.main_begin); }
  constexpr T *auxLimb(uint32_t index) const {
    return limb(layout_.basis.mainCount() + (index - layout_.basis.aux_begin));
  }

  // A view over main limbs [begin, end) of this polynomial and no aux limbs
  // — the shape of a key-switching digit. No copy: the range is contiguous
  // in storage, so the result addresses the same words. Refuses a range that
  // is empty, inverted, or not inside this polynomial's own main range.
  std::optional<BasicRnsPoly> mainSlice(uint32_t begin, uint32_t end) const {
    if (begin < layout_.basis.main_begin || end > layout_.basis.main_end || begin >= end) {
      return std::nullopt;
    }
    PolyLayout sliced = layout_;
    sliced.basis.main_begin = begin;
    sliced.basis.main_end = end;
    sliced.basis.aux_begin = 0;
    sliced.basis.aux_end = 0;
    return BasicRnsPoly(sliced, mainLimb(begin));
  }

  private:
  PolyLayout layout_{};
  T *data_ = nullptr;
};

using RnsPolyView = BasicRnsPoly<const uint64_t>;
using RnsPolyRef = BasicRnsPoly<uint64_t>;

// Copy every live word of `src` into `dst`. Refuses, writing nothing, unless
// the two layouts are identical — a copy between different layouts is a
// conversion, and conversions are L4 operations with names of their own.
// Tolerates dst and src addressing the same words.
bool copy(RnsPolyRef dst, RnsPolyView src);

} // namespace fhenomenon::ckks
