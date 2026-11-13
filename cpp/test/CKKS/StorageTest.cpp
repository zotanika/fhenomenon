#include "CKKS/Storage.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace {
using namespace fhenomenon::ckks;

PolyLayout polyLayout() { return {{ParamsId{123}, 0, 2, 0, 0}, 3, NttForm::Evaluation}; }
CiphertextLayout ciphertextLayout() { return {polyLayout(), 1024.0, 2}; }

static_assert(std::is_aggregate_v<RnsPolyView> && std::is_aggregate_v<CiphertextView>);
static_assert(std::is_trivially_copyable_v<RnsPolyView> && std::is_trivially_copyable_v<CiphertextView>);
static_assert(!std::is_copy_constructible_v<PolySlab> && std::is_nothrow_move_constructible_v<PolySlab>);
static_assert(std::is_same_v<decltype(asConst(RnsPolyView{}).limb(0)), const uint64_t *>);
static_assert(std::is_same_v<decltype(asConst(CiphertextView{}).poly(0).data), const uint64_t *>);

TEST(Storage, SlabOwnershipMovesWithoutMovingResidues) {
  PolySlab slab(40);
  ASSERT_NE(slab.data(), nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(slab.data()) % 64U, 0U);
  slab.data()[39] = 0x12345678;
  auto *original = slab.data();
  PolySlab moved(std::move(slab));
  EXPECT_EQ(moved.data(), original);
  EXPECT_EQ(moved.capacityWords(), 40U);
  EXPECT_EQ(moved.data()[39], 0x12345678U);
  EXPECT_EQ(slab.data(), nullptr);
  EXPECT_EQ(slab.capacityWords(), 0U);
  PolySlab assigned(8);
  assigned = std::move(moved);
  EXPECT_EQ(assigned.data(), original);
  EXPECT_EQ(moved.data(), nullptr);
  auto *self = &assigned;
  assigned = std::move(*self);
  EXPECT_EQ(assigned.data(), original);
  EXPECT_EQ(assigned.capacityWords(), 40U);
}

TEST(Storage, EmptyAndOverflowingSlabsHaveDefinedBehavior) {
  PolySlab empty(0);
  EXPECT_EQ(empty.data(), nullptr);
  EXPECT_EQ(empty.capacityWords(), 0U);
  EXPECT_THROW(PolySlab{std::numeric_limits<size_t>::max()}, std::bad_array_new_length);
}

TEST(Storage, PolyViewsValidateLiveCapacityAndAlignment) {
  PolySlab slab(24);
  RnsPolyView view{slab.data(), 24, polyLayout()};
  ASSERT_TRUE(view.wellFormed());
  view.limb(1)[7] = 77;
  EXPECT_EQ(asConst(view).limb(1)[7], 77U);
  EXPECT_EQ(view.limb(2), nullptr);
  view.capacity_words = 15;
  EXPECT_FALSE(view.wellFormed());
  EXPECT_EQ(view.limb(0), nullptr);
  view = {slab.data() + 1, 16, polyLayout()};
  EXPECT_FALSE(view.wellFormed());
  EXPECT_FALSE(RnsPolyView{}.wellFormed());
}

TEST(Storage, CiphertextUsesStrideAndProtectsOtherPolynomials) {
  PolySlab slab(40);
  CiphertextView view{slab.data(), 40, 24, ciphertextLayout()};
  ASSERT_TRUE(view.wellFormed());
  EXPECT_EQ(view.poly(1).data, slab.data() + 24);
  EXPECT_EQ(view.poly(0).capacity_words, 24U);
  EXPECT_EQ(view.poly(1).capacity_words, 16U);
  view.poly(0).limb(1)[0] = 3;
  view.poly(1).limb(1)[0] = 9;
  EXPECT_EQ(asConst(view).poly(0).limb(1)[0], 3U);
  EXPECT_EQ(asConst(view).poly(1).limb(1)[0], 9U);
  EXPECT_EQ(view.poly(2).data, nullptr);
  // Shrinking the live basis leaves polynomial 1 at its reserved offset.
  view.layout.poly.basis.main_end = 1;
  EXPECT_TRUE(view.wellFormed());
  EXPECT_EQ(view.poly(1).data, slab.data() + 24);
}

TEST(Storage, CiphertextRejectsOverlapMisalignmentAndShortTail) {
  PolySlab slab(40);
  CiphertextView view{slab.data(), 40, 8, ciphertextLayout()};
  EXPECT_FALSE(view.wellFormed());
  view.poly_stride_words = 17;
  EXPECT_FALSE(view.wellFormed());
  view.poly_stride_words = 24;
  view.capacity_words = 39;
  EXPECT_FALSE(view.wellFormed());
  EXPECT_EQ(view.poly(0).data, nullptr);
  view.capacity_words = 40;
  view.poly_stride_words = std::numeric_limits<size_t>::max() - 7U;
  EXPECT_FALSE(view.wellFormed());
}

TEST(Storage, MalformedMetadataFailsWithoutThrowing) {
  PolySlab slab(40);
  const CiphertextView valid{slab.data(), 40, 24, ciphertextLayout()};
  auto view = valid;
  view.layout.poly.basis.params_id = {};
  EXPECT_FALSE(view.wellFormed());
  view = valid;
  view.layout.poly.basis.main_begin = 3;
  EXPECT_FALSE(view.wellFormed());
  view = valid;
  view.layout.poly.basis.aux_end = std::numeric_limits<uint32_t>::max();
  EXPECT_FALSE(view.wellFormed());
  view = valid;
  view.layout.poly.log_degree = 64;
  EXPECT_FALSE(view.wellFormed());
  view = valid;
  view.layout.poly.form = static_cast<NttForm>(99);
  EXPECT_FALSE(view.wellFormed());
  view = valid;
  view.layout.scale = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(view.wellFormed());
  view = valid;
  view.layout.num_polys = 5;
  EXPECT_FALSE(view.wellFormed());
  view = valid;
  view.capacity_words = std::numeric_limits<size_t>::max();
  EXPECT_FALSE(view.wellFormed());
  view = valid;
  view.layout.poly.basis.main_begin = 1;
  EXPECT_FALSE(view.wellFormed());
}
} // namespace
