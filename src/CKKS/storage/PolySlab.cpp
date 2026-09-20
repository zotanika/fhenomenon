#include "CKKS/PolySlab.h"

#include <cstdint>
#include <cstring>
#include <new>
#include <utility>

namespace fhenomenon::ckks {

namespace {

// Over-aligned allocation has to be released with the matching over-aligned
// delete, so both sides live here next to each other.
uint64_t *allocateWords(std::size_t words) {
  if (words == 0) {
    return nullptr;
  }
  // A word count whose byte size wraps would otherwise become a small
  // allocation that reports itself as enormous. operator new already throws
  // bad_alloc for a request it cannot meet; this is the same failure mode
  // for a request that cannot even be expressed.
  if (words > SIZE_MAX / sizeof(uint64_t)) {
    throw std::bad_alloc();
  }
  return static_cast<uint64_t *>(::operator new(words * sizeof(uint64_t), std::align_val_t{PolySlab::kAlignment}));
}

void releaseWords(uint64_t *data) {
  if (data != nullptr) {
    ::operator delete(data, std::align_val_t{PolySlab::kAlignment});
  }
}

} // namespace

PolySlab::PolySlab(std::size_t words) : data_(allocateWords(words)), words_(words) {}

PolySlab::PolySlab(PolySlab &&other) noexcept : data_(other.data_), words_(other.words_) {
  other.data_ = nullptr;
  other.words_ = 0;
}

PolySlab &PolySlab::operator=(PolySlab &&other) noexcept {
  if (this != &other) {
    releaseWords(data_);
    data_ = other.data_;
    words_ = other.words_;
    other.data_ = nullptr;
    other.words_ = 0;
  }
  return *this;
}

PolySlab::~PolySlab() { releaseWords(data_); }

PolySlab PolySlab::clone() const {
  PolySlab copy(words_);
  if (words_ != 0) {
    std::memcpy(copy.data_, data_, bytes());
  }
  return copy;
}

} // namespace fhenomenon::ckks
