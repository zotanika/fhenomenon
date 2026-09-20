#pragma once

#include "CKKS/Layout.h"
#include "CKKS/PolySlab.h"
#include "CKKS/RnsPoly.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>

// L2 — a ciphertext layout bound to the residues it describes, as a
// non-owning handle (BasicCiphertext) and as the owner behind it
// (Ciphertext).
//
// Polynomials sit at a UNIFORM STRIDE from one base pointer: polynomial p
// starts at data() + p * polyStride(). No array of pointers, so a handle is
// three words whatever kMaxPolys is, and a stride wider than the live
// polynomial is a per-polynomial reserve at each polynomial's own tail. That
// reserve is what makes the two shape changes CKKS performs most often free
// of any memmove: RESCALE drops the last live limb of every polynomial by
// decrementing main_end, and ModUp appends aux limbs at each live tail.
//
// A stride equal to the live polynomial is the tight case, which is how a
// temporary carved from an Arena is bound: no reserve, because a temporary
// never changes shape. A kernel takes the handle and cannot tell the two
// apart, which is the whole reason the handle and the owner are different
// types.
//
// A plaintext is a one-polynomial layout over this same type. Nothing about
// storage distinguishes it; whether an operation accepts one is for L4 to
// say in the operation's signature.

namespace fhenomenon::ckks {

template <typename T> class BasicCiphertext {
  static_assert(std::is_same_v<std::remove_const_t<T>, uint64_t>, "BasicCiphertext binds uint64_t residues only");

  public:
  using value_type = T;

  constexpr BasicCiphertext() = default;

  // Preconditions: poly_stride >= layout.poly.wordCount(), and `data`
  // addresses layout.num_polys * poly_stride words that outlive the handle.
  // Unchecked for the same reason BasicRnsPoly's constructor is: the owner
  // that produced `data` is where the size was validated.
  constexpr BasicCiphertext(const CiphertextLayout &layout, T *data, std::size_t poly_stride)
    : layout_(layout), data_(data), poly_stride_(poly_stride) {}

  template <typename U, typename = std::enable_if_t<std::is_const_v<T> && std::is_same_v<std::remove_const_t<T>, U>>>
  constexpr BasicCiphertext(const BasicCiphertext<U> &other)
    : layout_(other.layout()), data_(other.data()), poly_stride_(other.polyStride()) {}

  constexpr const CiphertextLayout &layout() const { return layout_; }
  constexpr T *data() const { return data_; }
  constexpr std::size_t polyStride() const { return poly_stride_; }
  constexpr uint32_t numPolys() const { return layout_.num_polys; }
  constexpr int64_t level() const { return layout_.level(); }
  constexpr double scale() const { return layout_.scale; }
  constexpr bool empty() const { return data_ == nullptr; }

  // Polynomial p < numPolys(), as a handle over its live limbs. Unchecked:
  // addressing within a bound layout.
  constexpr BasicRnsPoly<T> poly(uint32_t p) const {
    return BasicRnsPoly<T>(layout_.poly, data_ + std::size_t{p} * poly_stride_);
  }

  private:
  CiphertextLayout layout_{};
  T *data_ = nullptr;
  std::size_t poly_stride_ = 0;
};

using CiphertextView = BasicCiphertext<const uint64_t>;
using CiphertextRef = BasicCiphertext<uint64_t>;

// The owner. A PolySlab, a layout, and two capacities the layout may grow
// into without the slab changing.
//
// Capacity is a separate, checked quantity from the live shape because the
// FHN ABI forces it to be: FhnBufferAllocFn takes no size, so a buffer is
// sized once, at the largest shape it will ever hold, and then reshaped as
// the ciphertext moves through the program — down a level on every rescale,
// up to three polynomials on FHN_MULT_CC, back to two on relinearisation.
// The bytes never move; only the layout the handle carries changes.
//
// Ownership follows invariant I4: move transfers, duplicate() allocates and
// is named for it, ref()/view() borrow, and there is no copy constructor.
class Ciphertext {
  public:
  // Owns nothing and has no shape. reshape() refuses it: a shape needs
  // memory behind it, and there is none.
  Ciphertext() = default;

  // Allocates exactly slabWords(...) words and binds `layout` over them. The
  // capacities must cover the layout and stay within the layout bounds
  // (RnsBasis::kMaxLimbs, CiphertextLayout::kMaxPolys); when `error` is
  // non-null it receives the first reason for a refusal.
  static std::optional<Ciphertext> create(const CiphertextLayout &layout, uint32_t limb_capacity,
                                          uint32_t poly_capacity, std::string *error = nullptr);

  // Binds `layout` over a slab the caller already owns — bytes from the wire,
  // or a slab being reused. The slab must hold at least slabWords(...) words;
  // any excess is owned but never addressed. On refusal the slab is dropped
  // with the rest of the argument, so pass a clone() if it must survive.
  static std::optional<Ciphertext> adopt(PolySlab slab, const CiphertextLayout &layout, uint32_t limb_capacity,
                                         uint32_t poly_capacity, std::string *error = nullptr);

  // Words a slab must hold for these capacities: poly_capacity polynomials
  // at a stride of limb_capacity limbs. Pure, so a planner can ask before any
  // slab exists. Inputs are expected within the layout bounds; the product
  // fits size_t comfortably at every representable degree.
  static std::size_t slabWords(uint32_t log_degree, uint32_t limb_capacity, uint32_t poly_capacity) {
    return std::size_t{poly_capacity} * (std::size_t{limb_capacity} << log_degree);
  }

  Ciphertext(const Ciphertext &) = delete;
  Ciphertext &operator=(const Ciphertext &) = delete;
  Ciphertext(Ciphertext &&other) noexcept;
  Ciphertext &operator=(Ciphertext &&other) noexcept;
  ~Ciphertext() = default;

  // A new owner with the same capacities and the same live words. Words
  // outside the live shape are not copied and are not defined in the result,
  // exactly as they are not defined in a fresh create().
  Ciphertext duplicate() const;

  // Change the live shape without moving a byte. Refuses, leaving the
  // current layout in place, unless the new layout is well formed, in the
  // same ring (parameter set and degree — the stride depends on the degree,
  // and a buffer does not change parameter sets), and within both
  // capacities. This is the one mutator, so a shape change is atomic: basis,
  // polynomial count and scale move together or not at all.
  //
  // Division of labour with kernels: a kernel computes through handles whose
  // layout is fixed for the call, and whoever holds the owner calls
  // reshape() to declare the shape the kernel produced. Because reshape()
  // moves no word, it is safe before or after the kernel runs — a RESCALE may
  // reshape first and let the kernel address the new live range, or compute
  // first and reshape to publish the result.
  bool reshape(const CiphertextLayout &layout, std::string *error = nullptr);

  const CiphertextLayout &layout() const { return layout_; }
  uint32_t limbCapacity() const { return limb_capacity_; }
  uint32_t polyCapacity() const { return poly_capacity_; }
  std::size_t polyStride() const { return std::size_t{limb_capacity_} << layout_.poly.log_degree; }
  const PolySlab &slab() const { return slab_; }
  bool empty() const { return slab_.empty(); }
  int64_t level() const { return layout_.level(); }
  double scale() const { return layout_.scale; }

  CiphertextRef ref() { return CiphertextRef(layout_, slab_.data(), polyStride()); }
  CiphertextView view() const { return CiphertextView(layout_, slab_.data(), polyStride()); }
  RnsPolyRef poly(uint32_t p) { return ref().poly(p); }
  RnsPolyView poly(uint32_t p) const { return view().poly(p); }

  private:
  Ciphertext(PolySlab slab, const CiphertextLayout &layout, uint32_t limb_capacity, uint32_t poly_capacity);

  PolySlab slab_;
  CiphertextLayout layout_{};
  uint32_t limb_capacity_ = 0;
  uint32_t poly_capacity_ = 0;
};

// Copy every live word of `src` into `dst`, polynomial by polynomial. Refuses,
// writing nothing, unless the two layouts are identical; the strides may
// differ, which is how an Arena temporary lands in a buffer.
//
// Aliasing: the same handle (same base, same stride) is a no-op and returns
// true — the ABI lets a kernel's result be one of its operands. The same base
// under different strides is refused, because copying polynomial p over the
// words polynomial p+1 has not been read from yet is a re-striding in place,
// and nothing at this layer asks for one. Any other overlap is undefined, as
// it is for memcpy.
bool copy(CiphertextRef dst, CiphertextView src);

} // namespace fhenomenon::ckks
