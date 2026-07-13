#include "Backend/External.h"
#include "FHN/FhnDefaultExecutor.h"
#include "FHN/fhn_program.h"
#include "Fhenon.h"
#include "Parameter/ParameterGen.h"

#include <any>
#include <dlfcn.h>
#include <gtest/gtest.h>

// We test ExternalBackend by loading a shared library that exports fhn_* symbols.
// The test build produces libtoyfhe_fhn.so from ToyFheKernels.cpp for this purpose.

using namespace fhenomenon;

static std::string getTestLibPath() {
#ifdef __APPLE__
  return std::string(TEST_LIB_DIR) + "/libtoyfhe_fhn.dylib";
#else
  return std::string(TEST_LIB_DIR) + "/libtoyfhe_fhn.so";
#endif
}

TEST(FhnExternalBackend, LoadAndQueryInfo) {
  ExternalBackend backend(getTestLibPath(), nullptr, "toyfhe_");

  auto *info = backend.getInfo();
  ASSERT_NE(info, nullptr);
  EXPECT_STREQ(info->name, "toyfhe-reference");
  EXPECT_EQ(info->device_type, FHN_DEVICE_CPU);
}

TEST(FhnExternalBackend, ExecutorSupportsComputeOpsOnly) {
  ExternalBackend backend(getTestLibPath(), nullptr, "toyfhe_");

  auto *exec = backend.getFhnExecutor();
  ASSERT_NE(exec, nullptr);
  EXPECT_TRUE(exec->supports(FHN_ADD_CC));
  EXPECT_TRUE(exec->supports(FHN_HMULT));
  EXPECT_TRUE(exec->supports(FHN_ROTATE));
  EXPECT_FALSE(exec->supports(FHN_CONJUGATE));
}

TEST(FhnExternalBackend, ResolvesDataPlane) {
  ExternalBackend backend(getTestLibPath(), nullptr, "toyfhe_");

  auto &vtable = backend.getVTable();
  // The ABI handshake resolved and matched, or the constructor would throw.
  ASSERT_NE(vtable.get_abi_version, nullptr);
  EXPECT_EQ(vtable.get_abi_version(), FHN_ABI_VERSION);
  ASSERT_NE(vtable.buffer_alloc, nullptr);
  ASSERT_NE(vtable.buffer_free, nullptr);
  // ToyFHE holds its own key material, so the optional key operations exist.
  EXPECT_NE(vtable.encrypt_i64, nullptr);
  EXPECT_NE(vtable.encrypt_f64, nullptr);
  EXPECT_NE(vtable.decrypt_i64, nullptr);
  EXPECT_NE(vtable.decrypt_f64, nullptr);
}

TEST(FhnExternalBackend, EncryptAddDecryptViaDlopen) {
  ExternalBackend backend(getTestLibPath(), nullptr, "toyfhe_");

  auto &vtable = backend.getVTable();
  auto *ctx = backend.getFhnCtx();
  auto *exec = backend.getFhnExecutor();
  ASSERT_NE(vtable.encrypt_i64, nullptr);
  ASSERT_NE(vtable.decrypt_i64, nullptr);

  // Inputs enter through the host-side data plane; the program is
  // compute-only: bufs[3] = bufs[1] + bufs[2]
  auto *prog = fhn_program_alloc(1, 2, 1);
  prog->input_ids[0] = 1;
  prog->input_ids[1] = 2;
  prog->output_ids[0] = 3;
  prog->instructions[0] = {FHN_ADD_CC, 3, {1, 2, 0, 0}, {}, {}};

  FhnBuffer *bufs[4];
  for (int i = 0; i < 4; i++)
    bufs[i] = vtable.buffer_alloc(ctx);
  ASSERT_EQ(vtable.encrypt_i64(ctx, bufs[1], 10), 0);
  ASSERT_EQ(vtable.encrypt_i64(ctx, bufs[2], 20), 0);

  int err = exec->execute(ctx, prog, bufs);
  EXPECT_EQ(err, 0);

  int64_t result = 0;
  ASSERT_EQ(vtable.decrypt_i64(ctx, bufs[3], &result), 0);
  EXPECT_EQ(result, 30);

  for (int i = 0; i < 4; i++)
    vtable.buffer_free(ctx, bufs[i]);
  fhn_program_free(prog);
}

// The Backend-interface path: transform() encrypts and decrypt() decrypts
// through the data plane instead of silently returning 0.
TEST(FhnExternalBackend, BackendInterfaceEncryptDecrypt) {
  ExternalBackend backend(getTestLibPath(), nullptr, "toyfhe_");

  Fhenon<int> a = 123;
  auto param = ParameterGen::createCKKSParam(CKKSParamPreset::FGb);
  backend.transform(a, *param);
  EXPECT_TRUE(a.isEncrypted_);
  EXPECT_EQ(std::any_cast<int>(backend.decrypt(a)), 123);
}

TEST(FhnExternalBackend, InvalidLibraryThrows) {
  EXPECT_THROW(ExternalBackend("/nonexistent/libfoo.so"), std::runtime_error);
}

// A ciphertext produced by one backend must be rejected by another: the
// bytes behind an FhnBuffer are backend-specific, and the owner tag is the
// firewall that keeps them from reaching foreign kernels.
TEST(FhnExternalBackend, RejectsForeignCiphertext) {
  ExternalBackend backend(getTestLibPath(), nullptr, "toyfhe_");
  auto param = ParameterGen::createCKKSParam(CKKSParamPreset::FGb);

  Fhenon<int> a = 7;
  Backend::getInstance().transform(a, *param); // encrypted by the singleton (Builtin)

  EXPECT_THROW(backend.decrypt(a), std::runtime_error);
}

// Buffers must outlive the backend that allocated them: the library and
// context are released only after the last buffer is freed.
TEST(FhnExternalBackend, BufferOutlivesBackend) {
  auto param = ParameterGen::createCKKSParam(CKKSParamPreset::FGb);
  Fhenon<int> a = 41;
  {
    ExternalBackend backend(getTestLibPath(), nullptr, "toyfhe_");
    backend.transform(a, *param);
    EXPECT_EQ(std::any_cast<int>(backend.decrypt(a)), 41);
  }
  // Backend destroyed; releasing the entity's buffer must not crash (the
  // deleter calls into the library, which the keepalive holds loaded).
  a.ciphertext_.reset();
  a.isEncrypted_ = false;
  SUCCEED();
}

// ToyFHE has a single memory space and exports no movement hooks; the
// runtime must surface them as null so plan execution skips the actions.
TEST(FhnExternalBackend, MovementHooksAbsentResolveToNull) {
  ExternalBackend backend(getTestLibPath(), nullptr, "toyfhe_");
  const FhnRuntime *rt = backend.fhnRuntime();
  ASSERT_NE(rt, nullptr);
  EXPECT_EQ(rt->prefetch, nullptr);
  EXPECT_EQ(rt->evict, nullptr);
}

// ToyFHE declares a flat level model: the trio resolves (positive dlsym
// coverage), fresh level 0, a nonzero flat size, PRESERVE everywhere.
TEST(FhnExternalBackend, LevelModelResolvesFlatForToyFhe) {
  ExternalBackend backend(getTestLibPath(), nullptr, "toyfhe_");
  const FhnRuntime *rt = backend.fhnRuntime();
  ASSERT_NE(rt, nullptr);
  ASSERT_NE(rt->fresh_level, nullptr);
  ASSERT_NE(rt->level_bytes, nullptr);
  ASSERT_NE(rt->opcode_level_effect, nullptr);
  EXPECT_EQ(rt->fresh_level(rt->ctx), 0);
  EXPECT_GT(rt->level_bytes(rt->ctx, 0), 0u);
  EXPECT_EQ(rt->level_bytes(rt->ctx, 1), 0u); // invalid level
  EXPECT_EQ(rt->opcode_level_effect(rt->ctx, FHN_ADD_CC), FHN_LEVEL_PRESERVE);
  EXPECT_EQ(rt->opcode_level_effect(rt->ctx, FHN_HMULT), FHN_LEVEL_PRESERVE);
}
