#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace fhenomenon::ckks {

// A caller-owned scratch allocator — layer L3.
//
// Invariant I2: allocation is a parameter, never ambient. There is no global
// arena, no getter that returns one, and no default. A function that needs
// temporaries takes an `Arena &`, or it does not allocate. This file
// deliberately exports no free function that produces an Arena.
//
// The arena is a bump allocator over a single slab acquired once at
// construction. That is only viable because of requirement R1: an FhnProgram
// is straight-line and data-oblivious, so its scratch requirement is exactly
// known before execution. Sizing is therefore the caller's job, and running
// out is a planning bug rather than a runtime condition to recover from —
// allocate() reports it by returning nullptr instead of growing, because
// growing would silently invalidate the footprint the caller already
// reported.
//
// Arena is deliberately not thread-safe. Threads get one arena each; a lock
// here would reintroduce the shared-allocator contention the design exists to
// avoid.
class Arena {
  public:
  // Cache-line alignment. Every allocation is aligned to this, so a caller
  // never has to pad by hand to keep SIMD loads aligned.
  static constexpr std::size_t kAlignment = 64;

  explicit Arena(std::size_t capacity_bytes);

  Arena(const Arena &) = delete;
  Arena &operator=(const Arena &) = delete;
  ~Arena();

  // Move transfers the slab, cursor and high water, leaving the source empty.
  // Outstanding Scope guards on BOTH arenas are invalidated: they never
  // rewind the transferred or replaced storage. Self-move is a no-op.
  Arena(Arena &&other) noexcept;
  Arena &operator=(Arena &&other) noexcept;

  // Bump-allocate space for `count` objects of type T, kAlignment-aligned.
  // Returns nullptr if the arena cannot satisfy the request. The storage is
  // uninitialised; T must be trivially destructible because the arena never
  // runs destructors.
  template <typename T> T *allocate(std::size_t count) {
    static_assert(std::is_trivially_destructible_v<T>, "Arena never runs destructors");
    static_assert(alignof(T) <= kAlignment, "type over-aligned for the arena");
    if (count != 0 && count > (SIZE_MAX / sizeof(T))) {
      return nullptr;
    }
    void *storage = allocateBytes(count * sizeof(T));
    // Through void* on purpose: a direct reinterpret_cast from the byte slab
    // trips -Wcast-align even though the slab is over-aligned already.
    return static_cast<T *>(storage);
  }

  // Raw form of allocate(). Returns nullptr when exhausted, and for a
  // zero-byte request.
  void *allocateBytes(std::size_t bytes);

  // Rewind to empty. Frees nothing and keeps the slab, so the next program
  // reuses the same memory. highWater() survives a reset. Existing Scope
  // guards are invalidated so they cannot rewind allocations made afterward.
  void reset();

  std::size_t capacity() const { return capacity_; }
  std::size_t used() const { return offset_; }
  std::size_t remaining() const { return capacity_ - offset_; }
  // Largest `used()` ever reached. This is how a caller checks that the size
  // it planned was actually the size it needed — an arena whose high water is
  // far under capacity was mis-planned just as surely as one that overflowed.
  std::size_t highWater() const { return high_water_; }

  // RAII rewind for nested temporaries: an operation takes a Scope, allocates
  // freely, and everything it took is reclaimed when the Scope dies. This is
  // what lets composite kernels nest without the caller sizing each level.
  // A rewind also invalidates younger guards that are still alive, making
  // out-of-order destruction safe. Reset, move and Arena destruction likewise
  // detach guards; destroying a detached guard has no effect. Tracking uses
  // links in the guards themselves and never allocates.
  class Scope {
public:
    explicit Scope(Arena &arena);
    ~Scope();

    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
    Scope(Scope &&) = delete;
    Scope &operator=(Scope &&) = delete;

private:
    friend class Arena;
    Arena *arena_;
    std::size_t saved_;
    Scope *previous_;
  };

  private:
  struct SlabDeleter {
    void operator()(std::byte *slab) const noexcept { ::operator delete(slab, std::align_val_t{kAlignment}); }
  };

  static std::size_t roundUp(std::size_t bytes);
  // Detach guards down to (but excluding) the specified older guard.
  void invalidateScopes(Scope *stop = nullptr) noexcept;

  std::unique_ptr<std::byte, SlabDeleter> slab_;
  std::size_t capacity_ = 0;
  std::size_t offset_ = 0;
  std::size_t high_water_ = 0;
  Scope *scope_ = nullptr;
};

} // namespace fhenomenon::ckks
