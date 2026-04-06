#include "Backend/External.h"
#include "FHN/FhnDefaultExecutor.h"
#include "FHN/fhn_program.h"

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

TEST(FhnExternalBackend, ExecutorSupportsOps) {
  ExternalBackend backend(getTestLibPath(), nullptr, "toyfhe_");

  auto *exec = backend.getFhnExecutor();
  ASSERT_NE(exec, nullptr);
  EXPECT_TRUE(exec->supports(FHN_ENCRYPT));
  EXPECT_TRUE(exec->supports(FHN_DECRYPT));
  EXPECT_TRUE(exec->supports(FHN_ADD_CC));
  EXPECT_TRUE(exec->supports(FHN_HMULT));
}

// Resolve buffer helpers from the loaded library
using BufAllocFn = FhnBuffer *(*)(FhnBackendCtx *);
using BufFreeFn = void (*)(FhnBackendCtx *, FhnBuffer *);
using BufReadIntFn = int64_t (*)(FhnBackendCtx *, FhnBuffer *);

TEST(FhnExternalBackend, EncryptAddDecryptViaDlopen) {
  ExternalBackend backend(getTestLibPath(), nullptr, "toyfhe_");

  auto &vtable = backend.getVTable();
  auto *ctx = backend.getFhnCtx();
  auto *exec = backend.getFhnExecutor();

  // Resolve buffer helpers from the shared lib
  void *dl = dlopen(getTestLibPath().c_str(), RTLD_LAZY | RTLD_NOLOAD);
  ASSERT_NE(dl, nullptr);
  auto buf_alloc = reinterpret_cast<BufAllocFn>(dlsym(dl, "toyfhe_fhn_buffer_alloc"));
  auto buf_free = reinterpret_cast<BufFreeFn>(dlsym(dl, "toyfhe_fhn_buffer_free"));
  auto buf_read = reinterpret_cast<BufReadIntFn>(dlsym(dl, "toyfhe_fhn_buffer_read_int"));
  ASSERT_NE(buf_alloc, nullptr);
  ASSERT_NE(buf_free, nullptr);
  ASSERT_NE(buf_read, nullptr);

  // Program: encrypt(10), encrypt(20), add, decrypt
  auto *prog = fhn_program_alloc(4, 0, 1);
  prog->output_ids[0] = 4;
  prog->instructions[0] = {FHN_ENCRYPT, 1, {0, 0, 0, 0}, {10, 0, 0, 0}, {}};
  prog->instructions[1] = {FHN_ENCRYPT, 2, {0, 0, 0, 0}, {20, 0, 0, 0}, {}};
  prog->instructions[2] = {FHN_ADD_CC, 3, {1, 2, 0, 0}, {}, {}};
  prog->instructions[3] = {FHN_DECRYPT, 4, {3, 0, 0, 0}, {}, {}};

  FhnBuffer *bufs[5];
  for (int i = 0; i < 5; i++)
    bufs[i] = buf_alloc(ctx);

  int err = exec->execute(ctx, prog, bufs);
  EXPECT_EQ(err, 0);

  int64_t result = buf_read(ctx, bufs[4]);
  EXPECT_EQ(result, 30);

  for (int i = 0; i < 5; i++)
    buf_free(ctx, bufs[i]);
  fhn_program_free(prog);
  dlclose(dl);
}

TEST(FhnExternalBackend, InvalidLibraryThrows) {
  EXPECT_THROW(ExternalBackend("/nonexistent/libfoo.so"), std::runtime_error);
}
