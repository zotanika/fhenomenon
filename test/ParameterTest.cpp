#include "Parameter/ParameterGen.h"

#include <gtest/gtest.h>

using namespace fhenomenon;

// The preset constructor must yield the documented defaults. init() used to
// read the then-uninitialized key_size_/degree_ members before setParams
// overwrote them — undefined behavior that Apple clang at -O2 compiled into
// a trap, the macOS-only SIGTRAP that killed every test constructing a
// CKKSParameter. This pins the constructor path end-to-end on all platforms.
TEST(CKKSParameterTest, PresetConstructionYieldsDocumentedDefaults) {
  auto param = ParameterGen::createCKKSParam(CKKSParamPreset::FGb);
  auto *ckks = dynamic_cast<CKKSParameter *>(param.get());
  ASSERT_NE(ckks, nullptr);
  EXPECT_EQ(ckks->getKeySize(), 128);
  EXPECT_EQ(ckks->getDegree(), 8192);
}

TEST(CKKSParameterTest, ExplicitSizeConstructorKeepsGivenValues) {
  CKKSParameter param(256, 16384);
  EXPECT_EQ(param.getKeySize(), 256);
  EXPECT_EQ(param.getDegree(), 16384);
}
