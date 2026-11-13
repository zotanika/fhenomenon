#include "CKKS/Storage.h"

#include <limits>
#include <utility>

namespace fhenomenon::ckks {
PolySlab::PolySlab(std::size_t word_capacity) {
  if (word_capacity > std::numeric_limits<std::size_t>::max() / sizeof(uint64_t))
    throw std::bad_array_new_length{};
  if (word_capacity != 0) {
    data_ = static_cast<uint64_t *>(::operator new(word_capacity * sizeof(uint64_t), std::align_val_t{64}));
    capacity_words_ = word_capacity;
  }
}

PolySlab::~PolySlab() { ::operator delete(data_, std::align_val_t{64}); }

PolySlab::PolySlab(PolySlab &&other) noexcept
  : data_(std::exchange(other.data_, nullptr)), capacity_words_(std::exchange(other.capacity_words_, 0)) {}

PolySlab &PolySlab::operator=(PolySlab &&other) noexcept {
  if (this != &other) {
    ::operator delete(data_, std::align_val_t{64});
    data_ = std::exchange(other.data_, nullptr);
    capacity_words_ = std::exchange(other.capacity_words_, 0);
  }
  return *this;
}
} // namespace fhenomenon::ckks
