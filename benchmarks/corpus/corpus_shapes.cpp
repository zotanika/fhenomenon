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

} // namespace

std::vector<Shape> allShapes() {
  std::vector<Shape> shapes;
  shapes.push_back(shapeMatvec());
  shapes.push_back(shapeMatmulTile());
  shapes.push_back(shapeHorner15());
  shapes.push_back(shapeReduceTree());
  shapes.push_back(shapeWideFront());
  shapes.push_back(shapeLogreg());
  return shapes;
}

} // namespace corpus
} // namespace fhenomenon
