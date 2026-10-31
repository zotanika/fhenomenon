#pragma once

#include "CKKS/Layout.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>

namespace fhenomenon::ckks {

namespace storage_detail {
// Layout's diagnostic path constructs strings on invalid input. These checks
// deliberately avoid it: borrowing storage must be noexcept and allocate nothing.
inline bool validLayout(const PolyLayout &layout) noexcept {
  const auto &basis = layout.basis;
  if (basis.params_id.empty() || basis.main_end < basis.main_begin || basis.aux_end < basis.aux_begin)
    return false;
  const uint64_t limbs = uint64_t{basis.main_end - basis.main_begin} + basis.aux_end - basis.aux_begin;
  return limbs != 0 && limbs <= RnsBasis::kMaxLimbs && layout.log_degree >= PolyLayout::kMinLogDegree &&
         layout.log_degree <= PolyLayout::kMaxLogDegree &&
         (layout.form == NttForm::Coefficient || layout.form == NttForm::Evaluation);
}

inline bool validMemory(const void *data, std::size_t capacity_words) noexcept {
  const auto address = reinterpret_cast<uintptr_t>(data);
  return data != nullptr && address % 64U == 0 && capacity_words != 0 &&
         capacity_words <= (std::numeric_limits<uintptr_t>::max() - address) / sizeof(uint64_t);
}
} // namespace storage_detail

// Explicit durable allocation. Views below never own or grow this storage.
class PolySlab {
  public:
  explicit PolySlab(std::size_t word_capacity);
  ~PolySlab();
  PolySlab(const PolySlab &) = delete;
  PolySlab &operator=(const PolySlab &) = delete;
  PolySlab(PolySlab &&other) noexcept;
  PolySlab &operator=(PolySlab &&other) noexcept;

  uint64_t *data() noexcept { return data_; }
  const uint64_t *data() const noexcept { return data_; }
  std::size_t capacityWords() const noexcept { return capacity_words_; }

  private:
  uint64_t *data_ = nullptr;
  std::size_t capacity_words_ = 0;
};

template <typename Word> struct BasicRnsPoly {
  static_assert(std::is_same_v<Word, uint64_t> || std::is_same_v<Word, const uint64_t>);
  Word *data = nullptr;
  std::size_t capacity_words = 0;
  PolyLayout layout{};

  bool wellFormed() const noexcept {
    return storage_detail::validLayout(layout) && storage_detail::validMemory(data, capacity_words) &&
           layout.wordCount() <= capacity_words;
  }
  Word *limb(uint32_t index) const noexcept {
    return wellFormed() && index < layout.basis.limbCount() ? data + std::size_t{index} * layout.degreeWords()
                                                            : nullptr;
  }
};
using RnsPolyView = BasicRnsPoly<uint64_t>;
using ConstRnsPolyView = BasicRnsPoly<const uint64_t>;
inline ConstRnsPolyView asConst(RnsPolyView view) noexcept { return {view.data, view.capacity_words, view.layout}; }

template <typename Word> struct BasicCiphertext {
  static_assert(std::is_same_v<Word, uint64_t> || std::is_same_v<Word, const uint64_t>);
  Word *data = nullptr;
  std::size_t capacity_words = 0;
  std::size_t poly_stride_words = 0;
  CiphertextLayout layout{};

  bool wellFormed() const noexcept {
    if (!storage_detail::validLayout(layout.poly) || !storage_detail::validMemory(data, capacity_words) ||
        layout.num_polys == 0 || layout.num_polys > CiphertextLayout::kMaxPolys || layout.poly.basis.main_begin != 0 ||
        layout.poly.basis.mainCount() == 0 || !std::isfinite(layout.scale) || layout.scale <= 0.0)
      return false;
    const std::size_t live = layout.poly.wordCount();
    // Each polynomial begins on a cache line, and its live words cannot
    // overlap the next polynomial. A final polynomial need not have padding.
    if (poly_stride_words < live || poly_stride_words % 8U != 0 || capacity_words < live)
      return false;
    return std::size_t{layout.num_polys - 1U} <= (capacity_words - live) / poly_stride_words;
  }
  BasicRnsPoly<Word> poly(uint32_t index) const noexcept {
    if (!wellFormed() || index >= layout.num_polys)
      return {};
    const std::size_t offset = std::size_t{index} * poly_stride_words;
    const std::size_t remaining = capacity_words - offset;
    return {data + offset, remaining < poly_stride_words ? remaining : poly_stride_words, layout.poly};
  }
};
using CiphertextView = BasicCiphertext<uint64_t>;
using ConstCiphertextView = BasicCiphertext<const uint64_t>;
inline ConstCiphertextView asConst(CiphertextView view) noexcept {
  return {view.data, view.capacity_words, view.poly_stride_words, view.layout};
}

enum class SwitchKeyKind { Relinearization, Galois };
struct EvaluationKeyView {
  const ConstCiphertextView *digits = nullptr;
  uint32_t digit_count = 0;
  ParamsId params_id{};
  uint32_t log_degree = 0;
  uint32_t main_count = 0;
  uint32_t aux_count = 0;
  SwitchKeyKind kind = SwitchKeyKind::Relinearization;
  uint64_t galois_element = 1;
};

} // namespace fhenomenon::ckks
