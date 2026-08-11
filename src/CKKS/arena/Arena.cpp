#include "CKKS/Arena.h"

namespace fhenomenon::ckks {

Arena::Arena(std::size_t capacity_bytes) : capacity_(roundUp(capacity_bytes)) {
  if (capacity_ != 0) {
    slab_.reset(static_cast<std::byte *>(::operator new(capacity_, std::align_val_t{kAlignment})));
  }
}

void *Arena::allocateBytes(std::size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  const std::size_t padded = roundUp(bytes);
  // roundUp saturates, so a request near SIZE_MAX fails here instead of
  // wrapping into a small offset.
  if (padded < bytes || padded > capacity_ - offset_) {
    return nullptr;
  }
  std::byte *storage = slab_.get() + offset_;
  offset_ += padded;
  if (offset_ > high_water_) {
    high_water_ = offset_;
  }
  return storage;
}

void Arena::reset() { offset_ = 0; }

std::size_t Arena::roundUp(std::size_t bytes) {
  const std::size_t remainder = bytes % kAlignment;
  if (remainder == 0) {
    return bytes;
  }
  const std::size_t padding = kAlignment - remainder;
  if (bytes > SIZE_MAX - padding) {
    return SIZE_MAX;
  }
  return bytes + padding;
}

} // namespace fhenomenon::ckks
