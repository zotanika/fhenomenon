#include "corpus_shapes.h"

#include "corpus_builder.h"

namespace fhenomenon {
namespace corpus {

namespace {

// Deterministic small values: |v| <= 8.
int64_t val(uint32_t i) { return static_cast<int64_t>((i * 7 + 3) % 17) - 8; }
Slots vec(uint32_t n, uint32_t seed) {
  Slots v(n);
  for (uint32_t i = 0; i < n; ++i)
    v[i] = val(seed * 131 + i);
  return v;
}

// Rotate-accumulate reduction: s = sum over all slots, in every slot.
uint32_t reduceTree(ShapeBuilder &b, uint32_t x, uint32_t n) {
  uint32_t s = x;
  for (uint32_t k = n / 2; k >= 1; k /= 2)
    s = b.cc(FHN_ADD_CC, s, b.rot(s, static_cast<int64_t>(k)));
  return s;
}

// Diagonal-packed matvec skeleton: one rotate + scalar-mult per diagonal,
// accumulated. Stresses fan-out reuse of the single input ciphertext.
uint32_t matvecOnce(ShapeBuilder &b, uint32_t x, uint32_t n, uint32_t seed) {
  uint32_t acc = 0;
  for (uint32_t d = 0; d < n; ++d) {
    const uint32_t r = (d == 0) ? x : b.rot(x, static_cast<int64_t>(d));
    const uint32_t m = b.cs(FHN_MULT_CS, r, ((static_cast<int64_t>(d) * 3 + seed) % 7) - 3);
    acc = (d == 0) ? m : b.cc(FHN_ADD_CC, acc, m);
  }
  return acc;
}

Shape shapeMatvec() {
  ShapeBuilder b;
  const uint32_t x = b.input(vec(64, 1));
  const uint32_t y = matvecOnce(b, x, 64, 0);
  return b.finish("matvec", "fan-out reuse of one input ct", 64, 0, {y});
}

Shape shapeMatmulTile() {
  ShapeBuilder b;
  const uint32_t x = b.input(vec(32, 2));
  const uint32_t y = matvecOnce(b, x, 32, 1);
  const uint32_t z = matvecOnce(b, y, 32, 2);
  return b.finish("matmul-tile", "chained matvecs, deep and wide", 32, 0, {z});
}

Shape shapeHorner15() {
  ShapeBuilder b;
  const uint32_t x = b.input({2});
  uint32_t acc = b.input({3}); // c15 as a ciphertext input
  for (int k = 14; k >= 0; --k) {
    acc = b.cc(FHN_MULT_CC, acc, x);
    acc = b.cs(FHN_ADD_CS, acc, ((static_cast<int64_t>(k) * 5) % 9) - 4);
  }
  return b.finish("horner15", "long sequential ct*ct chain", 1, 15, {acc});
}

Shape shapeReduceTree() {
  ShapeBuilder b;
  uint32_t combined = 0;
  for (uint32_t j = 0; j < 4; ++j) {
    const uint32_t x = b.input(vec(64, 10 + j));
    const uint32_t s = reduceTree(b, x, 64);
    combined = (j == 0) ? s : b.cc(FHN_ADD_CC, combined, s);
  }
  return b.finish("reduce-tree", "log-depth rotate-add fan-in x4", 64, 0, {combined});
}

Shape shapeWideFront() {
  ShapeBuilder b;
  std::vector<uint32_t> parts;
  for (uint32_t i = 0; i < 24; ++i) {
    const uint32_t a = b.input({val(100 + i)});
    const uint32_t c = b.input({val(200 + i)});
    const uint32_t d = b.input({val(300 + i)});
    parts.push_back(b.cc(FHN_ADD_CC, b.cc(FHN_MULT_CC, a, c), d));
  }
  while (parts.size() > 1) {
    std::vector<uint32_t> next;
    for (size_t i = 0; i + 1 < parts.size(); i += 2)
      next.push_back(b.cc(FHN_ADD_CC, parts[i], parts[i + 1]));
    if (parts.size() % 2 == 1)
      next.push_back(parts.back());
    parts = next;
  }
  return b.finish("wide-front", "maximum simultaneous live set", 1, 1, {parts[0]});
}

Shape shapeLogreg() {
  ShapeBuilder b;
  const uint32_t w = b.input(vec(32, 20)); // shared weights: fan-out x3
  uint32_t combined = 0;
  for (uint32_t j = 0; j < 3; ++j) {
    const uint32_t x = b.input(vec(32, 21 + j));
    const uint32_t d = reduceTree(b, b.cc(FHN_MULT_CC, x, w), 32);
    const uint32_t d3 = b.cc(FHN_MULT_CC, b.cc(FHN_MULT_CC, d, d), d);
    const uint32_t r = b.cc(FHN_ADD_CC, b.cs(FHN_MULT_CS, d, 3), b.cs(FHN_MULT_CS, d3, -1));
    combined = (j == 0) ? r : b.cc(FHN_ADD_CC, combined, r);
  }
  return b.finish("logreg", "shared-weight dot products + polynomial", 32, 3, {combined});
}

Shape shapeConv1d() {
  ShapeBuilder b;
  uint32_t p = b.input(vec(64, 30));
  for (uint32_t layer = 0; layer < 3; ++layer) {
    uint32_t acc = b.cs(FHN_MULT_CS, p, 2); // center tap
    const int64_t taps[4] = {-2, -1, 1, 2};
    for (int64_t t : taps) {
      const uint32_t m = b.cs(FHN_MULT_CS, b.rot(p, t), (t % 3) + 1);
      acc = b.cc(FHN_ADD_CC, acc, m);
    }
    p = acc;
  }
  return b.finish("conv1d", "overlapping stencil window reuse x3 layers", 64, 0, {p});
}

Shape shapeStats() {
  ShapeBuilder b;
  std::vector<uint32_t> outs;
  for (uint32_t j = 0; j < 2; ++j) {
    const uint32_t x = b.input(vec(64, 40 + j));
    const uint32_t s = reduceTree(b, x, 64);
    const uint32_t s2 = reduceTree(b, b.cc(FHN_MULT_CC, x, x), 64);
    // 64*Var proxy: 64*sum(x^2) - sum(x)^2
    const uint32_t v = b.cc(FHN_SUB_CC, b.cs(FHN_MULT_CS, s2, 64), b.cc(FHN_MULT_CC, s, s));
    outs.push_back(s);
    outs.push_back(v);
  }
  return b.finish("stats", "two datasets, four live outputs", 64, 2, outs);
}

Shape shapePsiEq() {
  ShapeBuilder b;
  std::vector<uint32_t> bits;
  for (uint32_t i = 0; i < 32; ++i) {
    const uint32_t a = b.input({val(500 + i)});
    const uint32_t c = b.input({(i % 4 == 0) ? val(500 + i) : val(700 + i)});
    bits.push_back(b.cc(FHN_EQ, a, c));
  }
  while (bits.size() > 1) {
    std::vector<uint32_t> next;
    for (size_t i = 0; i + 1 < bits.size(); i += 2)
      next.push_back(b.cc(FHN_AND, bits[i], bits[i + 1]));
    if (bits.size() % 2 == 1)
      next.push_back(bits.back());
    bits = next;
  }
  return b.finish("psi-eq", "boolean equality-AND tree (plan-only today)", 1, 0, {bits[0]});
}

Shape shapeIterUpdate() {
  ShapeBuilder b;
  uint32_t x = b.input({1});
  for (uint32_t k = 0; k < 32; ++k)
    x = (k % 2 == 0) ? b.cs(FHN_MULT_CS, x, 2) : b.cs(FHN_ADD_CS, x, 3);
  return b.finish("iter-update", "rebinding-heavy sequential chain", 1, 0, {x});
}

Shape shapeWeightedSum() {
  ShapeBuilder b;
  uint32_t acc = 0;
  for (uint32_t i = 0; i < 24; ++i) {
    const uint32_t x = b.input({val(600 + i)});
    const uint32_t m = b.cs(FHN_MULT_CS, x, ((static_cast<int64_t>(i) * 5) % 7) - 3);
    acc = (i == 0) ? m : b.cc(FHN_ADD_CC, acc, m);
  }
  return b.finish("weighted-sum", "scalar-multiply accumulate", 1, 0, {acc});
}

Shape shapeDiamond() {
  ShapeBuilder b;
  const uint32_t root = b.input({5});
  std::vector<uint32_t> tips;
  for (uint32_t d = 0; d < 2; ++d) {
    uint32_t p = root;
    for (uint32_t level = 0; level < 8; ++level) {
      const uint32_t l = b.cs(FHN_ADD_CS, p, static_cast<int64_t>(level) + 1);
      const uint32_t r = b.cc(FHN_ADD_CC, p, root); // long-gap root reuse
      p = b.cc(FHN_ADD_CC, l, r);
    }
    tips.push_back(p);
  }
  return b.finish("diamond", "fork-join with long-gap root reuse", 1, 0, {b.cc(FHN_ADD_CC, tips[0], tips[1])});
}

} // namespace

std::vector<Shape> allShapes() {
  std::vector<Shape> shapes;
  shapes.push_back(shapeMatvec());
  shapes.push_back(shapeMatmulTile());
  shapes.push_back(shapeHorner15());
  shapes.push_back(shapeReduceTree());
  shapes.push_back(shapeWideFront());
  shapes.push_back(shapeLogreg());
  shapes.push_back(shapeConv1d());
  shapes.push_back(shapeStats());
  shapes.push_back(shapePsiEq());
  shapes.push_back(shapeIterUpdate());
  shapes.push_back(shapeWeightedSum());
  shapes.push_back(shapeDiamond());
  return shapes;
}

} // namespace corpus
} // namespace fhenomenon
