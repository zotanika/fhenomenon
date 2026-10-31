#include "CKKS/Arena.h"

namespace fhenomenon::ckks {

Arena::Arena(std::size_t capacity_bytes) : capacity_(roundUp(capacity_bytes)) {
  if (capacity_ != 0) {
    slab_.reset(static_cast<std::byte *>(::operator new(capacity_, std::align_val_t{kAlignment})));
  }
}

Arena::~Arena() { invalidateScopes(); }

Arena::Arena(Arena &&other) noexcept
  : slab_(std::move(other.slab_)), capacity_(other.capacity_), offset_(other.offset_),
    high_water_(other.high_water_) {
  other.invalidateScopes();
  other.capacity_ = 0;
  other.offset_ = 0;
  other.high_water_ = 0;
}

Arena &Arena::operator=(Arena &&other) noexcept {
  if (this != &other) {
    invalidateScopes();
    other.invalidateScopes();
    slab_ = std::move(other.slab_);
    capacity_ = other.capacity_;
    offset_ = other.offset_;
    high_water_ = other.high_water_;
    other.capacity_ = 0;
    other.offset_ = 0;
    other.high_water_ = 0;
  }
  return *this;
}

Arena::Scope::Scope(Arena &arena) : arena_(&arena), saved_(arena.offset_), previous_(arena.scope_) {
  arena.scope_ = this;
}

Arena::Scope::~Scope() {
  if (arena_ != nullptr) {
    arena_->offset_ = saved_;
    arena_->invalidateScopes(previous_);
  }
}

void Arena::invalidateScopes(Scope *stop) noexcept {
  while (scope_ != stop) {
    Scope *scope = scope_;
    scope_ = scope->previous_;
    scope->arena_ = nullptr;
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

void Arena::reset() {
  invalidateScopes();
  offset_ = 0;
}

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
