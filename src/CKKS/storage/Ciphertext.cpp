#include "CKKS/Ciphertext.h"

#include <utility>

namespace fhenomenon::ckks {

namespace {

bool fail(std::string *error, std::string reason) {
  if (error != nullptr) {
    *error = std::move(reason);
  }
  return false;
}

// The capacity checks shared by create() and adopt(). Bounds, not policy:
// the layout types already say how many limbs and polynomials are
// representable, and a capacity beyond that would let a stride be computed
// that no layout could ever address.
bool checkCapacities(const CiphertextLayout &layout, uint32_t limb_capacity, uint32_t poly_capacity,
                     std::string *error) {
  if (!layout.wellFormed(error)) {
    return false;
  }
  const uint32_t live_limbs = layout.poly.basis.limbCount();
  if (limb_capacity < live_limbs) {
    return fail(error, "limb capacity " + std::to_string(limb_capacity) + " is below the " +
                         std::to_string(live_limbs) + " live limbs of the layout");
  }
  if (limb_capacity > RnsBasis::kMaxLimbs) {
    return fail(error, "limb capacity " + std::to_string(limb_capacity) + " is over the bound of " +
                         std::to_string(RnsBasis::kMaxLimbs));
  }
  if (poly_capacity < layout.num_polys) {
    return fail(error, "polynomial capacity " + std::to_string(poly_capacity) + " is below the " +
                         std::to_string(layout.num_polys) + " live polynomials of the layout");
  }
  if (poly_capacity > CiphertextLayout::kMaxPolys) {
    return fail(error, "polynomial capacity " + std::to_string(poly_capacity) + " is over the bound of " +
                         std::to_string(CiphertextLayout::kMaxPolys));
  }
  return true;
}

} // namespace

Ciphertext::Ciphertext(PolySlab slab, const CiphertextLayout &layout, uint32_t limb_capacity, uint32_t poly_capacity)
  : slab_(std::move(slab)), layout_(layout), limb_capacity_(limb_capacity), poly_capacity_(poly_capacity) {}

std::optional<Ciphertext> Ciphertext::create(const CiphertextLayout &layout, uint32_t limb_capacity,
                                             uint32_t poly_capacity, std::string *error) {
  if (!checkCapacities(layout, limb_capacity, poly_capacity, error)) {
    return std::nullopt;
  }
  return Ciphertext(PolySlab(slabWords(layout.poly.log_degree, limb_capacity, poly_capacity)), layout, limb_capacity,
                    poly_capacity);
}

std::optional<Ciphertext> Ciphertext::adopt(PolySlab slab, const CiphertextLayout &layout, uint32_t limb_capacity,
                                            uint32_t poly_capacity, std::string *error) {
  if (!checkCapacities(layout, limb_capacity, poly_capacity, error)) {
    return std::nullopt;
  }
  const std::size_t needed = slabWords(layout.poly.log_degree, limb_capacity, poly_capacity);
  if (slab.words() < needed) {
    fail(error,
         "slab holds " + std::to_string(slab.words()) + " words but these capacities need " + std::to_string(needed));
    return std::nullopt;
  }
  return Ciphertext(std::move(slab), layout, limb_capacity, poly_capacity);
}

// Move leaves the source shapeless as well as empty. Carrying the layout
// across would let a moved-from ciphertext report a level and a scale for
// residues it no longer holds.
Ciphertext::Ciphertext(Ciphertext &&other) noexcept
  : slab_(std::move(other.slab_)), layout_(other.layout_), limb_capacity_(other.limb_capacity_),
    poly_capacity_(other.poly_capacity_) {
  other.layout_ = CiphertextLayout{};
  other.limb_capacity_ = 0;
  other.poly_capacity_ = 0;
}

Ciphertext &Ciphertext::operator=(Ciphertext &&other) noexcept {
  if (this != &other) {
    slab_ = std::move(other.slab_);
    layout_ = other.layout_;
    limb_capacity_ = other.limb_capacity_;
    poly_capacity_ = other.poly_capacity_;
    other.layout_ = CiphertextLayout{};
    other.limb_capacity_ = 0;
    other.poly_capacity_ = 0;
  }
  return *this;
}

Ciphertext Ciphertext::duplicate() const {
  if (empty()) {
    return Ciphertext{};
  }
  Ciphertext out(PolySlab(slabWords(layout_.poly.log_degree, limb_capacity_, poly_capacity_)), layout_, limb_capacity_,
                 poly_capacity_);
  // Cannot refuse: the layouts are the same object.
  copy(out.ref(), view());
  return out;
}

bool Ciphertext::reshape(const CiphertextLayout &layout, std::string *error) {
  if (!layout.wellFormed(error)) {
    return false;
  }
  if (!layout.poly.sameRing(layout_.poly)) {
    return fail(error, "reshape would change the ring: a slab is bound to one parameter set and one degree, "
                       "and the stride depends on the degree");
  }
  const uint32_t limbs = layout.poly.basis.limbCount();
  if (limbs > limb_capacity_) {
    return fail(error, "reshape to " + std::to_string(limbs) + " limbs exceeds the capacity of " +
                         std::to_string(limb_capacity_));
  }
  if (layout.num_polys > poly_capacity_) {
    return fail(error, "reshape to " + std::to_string(layout.num_polys) + " polynomials exceeds the capacity of " +
                         std::to_string(poly_capacity_));
  }
  layout_ = layout;
  return true;
}

bool copy(CiphertextRef dst, CiphertextView src) {
  if (dst.layout() != src.layout()) {
    return false;
  }
  for (uint32_t p = 0; p < src.numPolys(); ++p) {
    copy(dst.poly(p), src.poly(p));
  }
  return true;
}

} // namespace fhenomenon::ckks
