// ---------------------------------------------------------------------------
// Integration test: Cheddar-FHE GPU backend via ExternalBackend (dlopen)
//
// Demonstrates:
// 1. Loading cheddar as an FHN backend via dlopen
// 2. Running CKKS operations (add, hmult, scalar ops) on GPU
// 3. How easy it is for FHE developers to add new kernels
//
// Uses a lightweight parameter set and shared context for fast execution.
// ---------------------------------------------------------------------------

#include "Backend/External.h"
#include "FHN/FhnDefaultExecutor.h"
#include "FHN/fhn_program.h"

#include "core/Container.h"
#include "core/Type.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <vector>

using namespace fhenomenon;
using word = uint32_t;

// Buffer helper function pointers (resolved once from the .so)
using BufAllocFn = FhnBuffer *(*)(FhnBackendCtx *);
using BufFreeFn = void (*)(FhnBackendCtx *, FhnBuffer *);
using BufReadComplexFn = void (*)(FhnBackendCtx *, FhnBuffer *, double *, double *, int);

// ---------------------------------------------------------------------------
// Shared fixture: cheddar context created ONCE for all tests.
// This is the main speedup — GPU init + key gen happens only once.
// ---------------------------------------------------------------------------
class CheddarGpuTest : public ::testing::Test {
  protected:
  // Created once, shared by all tests
  static std::unique_ptr<ExternalBackend> backend_;
  static void *dl_;
  static BufAllocFn buf_alloc_;
  static BufFreeFn buf_free_;
  static BufReadComplexFn buf_read_;

  static void SetUpTestSuite() {
    std::string lib = std::string(TEST_LIB_DIR) + "/libcheddar_fhn.so";
    std::string param = std::string(PARAM_DIR) + "/testparam_light.json";

    auto t0 = std::chrono::high_resolution_clock::now();
    backend_ = std::make_unique<ExternalBackend>(lib, param.c_str());
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "\n[  SETUP  ] Context + keys: " << ms << " ms (one-time cost)\n" << std::endl;

    dl_ = dlopen(lib.c_str(), RTLD_LAZY | RTLD_NOLOAD);
    buf_alloc_ = reinterpret_cast<BufAllocFn>(dlsym(dl_, "cheddar_fhn_buffer_alloc"));
    buf_free_ = reinterpret_cast<BufFreeFn>(dlsym(dl_, "cheddar_fhn_buffer_free"));
    buf_read_ = reinterpret_cast<BufReadComplexFn>(dlsym(dl_, "cheddar_fhn_buffer_read_complex"));
  }

  static void TearDownTestSuite() {
    backend_.reset();
    if (dl_)
      dlclose(dl_);
    dl_ = nullptr;
  }

  // Convenience accessors
  FhnBackendCtx *ctx() { return backend_->getFhnCtx(); }
  FhnDefaultExecutor *exec() { return backend_->getFhnExecutor(); }

  // Helper: CheddarBufKind::Message = 3
  struct CheddarFhnBuffer {
    int kind;
    cheddar::Ciphertext<word> ct;
    cheddar::Plaintext<word> pt;
    std::vector<cheddar::Complex> msg;
  };

  void setMessage(FhnBuffer *buf, const std::vector<cheddar::Complex> &msg) {
    auto *b = reinterpret_cast<CheddarFhnBuffer *>(buf);
    b->msg = msg;
    b->kind = 3; // CheddarBufKind::Message
  }

  // RAII buffer set
  struct Buffers {
    FhnBuffer **ptrs;
    int n;
    FhnBackendCtx *c;
    BufAllocFn alloc_fn;
    BufFreeFn free_fn;

    Buffers(int count, FhnBackendCtx *ctx, BufAllocFn af, BufFreeFn ff) : n(count), c(ctx), alloc_fn(af), free_fn(ff) {
      ptrs = new FhnBuffer *[count];
      for (int i = 0; i < count; i++)
        ptrs[i] = alloc_fn(ctx);
    }
    ~Buffers() {
      for (int i = 0; i < n; i++)
        free_fn(c, ptrs[i]);
      delete[] ptrs;
    }
    FhnBuffer *operator[](int i) { return ptrs[i]; }
    FhnBuffer **data() { return ptrs; }
  };

  Buffers allocBuffers(int n) { return Buffers(n, ctx(), buf_alloc_, buf_free_); }

  void readResult(FhnBuffer *buf, std::vector<double> &real, std::vector<double> &imag, int slots) {
    real.resize(slots);
    imag.resize(slots);
    buf_read_(ctx(), buf, real.data(), imag.data(), slots);
  }
};

// Static member definitions
std::unique_ptr<ExternalBackend> CheddarGpuTest::backend_;
void *CheddarGpuTest::dl_ = nullptr;
BufAllocFn CheddarGpuTest::buf_alloc_ = nullptr;
BufFreeFn CheddarGpuTest::buf_free_ = nullptr;
BufReadComplexFn CheddarGpuTest::buf_read_ = nullptr;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(CheddarGpuTest, BackendInfo) {
  auto *info = backend_->getInfo();
  ASSERT_NE(info, nullptr);
  EXPECT_STREQ(info->name, "cheddar-ckks");
  EXPECT_EQ(info->device_type, FHN_DEVICE_GPU);
}

TEST_F(CheddarGpuTest, SupportsAllCKKSOps) {
  // Ciphertext-ciphertext ops
  EXPECT_TRUE(exec()->supports(FHN_ADD_CC));
  EXPECT_TRUE(exec()->supports(FHN_SUB_CC));
  EXPECT_TRUE(exec()->supports(FHN_MULT_CC));
  EXPECT_TRUE(exec()->supports(FHN_HMULT));

  // Scalar ops (added via cheddar::ScalarOps)
  EXPECT_TRUE(exec()->supports(FHN_ADD_CS));
  EXPECT_TRUE(exec()->supports(FHN_MULT_CS));
  EXPECT_TRUE(exec()->supports(FHN_SUB_SC));

  // Key-switching / fused
  EXPECT_TRUE(exec()->supports(FHN_RELINEARIZE));
  EXPECT_TRUE(exec()->supports(FHN_RESCALE));
  EXPECT_TRUE(exec()->supports(FHN_HMULT));
  EXPECT_TRUE(exec()->supports(FHN_HROT));
  EXPECT_TRUE(exec()->supports(FHN_HROT_ADD));
  EXPECT_TRUE(exec()->supports(FHN_HCONJ_ADD));

  // Boolean NOT supported (CKKS scheme)
  EXPECT_FALSE(exec()->supports(FHN_AND));
  EXPECT_FALSE(exec()->supports(FHN_OR));
}

TEST_F(CheddarGpuTest, EncodeEncryptAddDecryptDecode) {
  // [1,2,3,4] + [10,20,30,40] = [11,22,33,44]
  const int N = 4;
  auto bufs = allocBuffers(10);

  setMessage(bufs[1], {{1, 0}, {2, 0}, {3, 0}, {4, 0}});
  setMessage(bufs[4], {{10, 0}, {20, 0}, {30, 0}, {40, 0}});

  auto *prog = fhn_program_alloc(7, 0, 1);
  prog->output_ids[0] = 9;
  prog->instructions[0] = {FHN_ENCODE, 2, {1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0}};
  prog->instructions[1] = {FHN_ENCRYPT, 3, {2, 0, 0, 0}, {}, {}};
  prog->instructions[2] = {FHN_ENCODE, 5, {4, 0, 0, 0}, {0, 0, 0, 0}, {0, 0}};
  prog->instructions[3] = {FHN_ENCRYPT, 6, {5, 0, 0, 0}, {}, {}};
  prog->instructions[4] = {FHN_ADD_CC, 7, {3, 6, 0, 0}, {}, {}};
  prog->instructions[5] = {FHN_DECRYPT, 8, {7, 0, 0, 0}, {}, {}};
  prog->instructions[6] = {FHN_DECODE, 9, {8, 0, 0, 0}, {}, {}};

  auto t0 = std::chrono::high_resolution_clock::now();
  EXPECT_EQ(exec()->execute(ctx(), prog, bufs.data()), 0);
  auto t1 = std::chrono::high_resolution_clock::now();

  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  std::cout << "  ADD_CC pipeline: " << us << " us" << std::endl;

  std::vector<double> real, imag;
  readResult(bufs[9], real, imag, N);
  for (int i = 0; i < N; i++)
    EXPECT_NEAR(real[i], (i + 1) + (i + 1) * 10, 1e-3) << "slot " << i;

  fhn_program_free(prog);
}

TEST_F(CheddarGpuTest, HMultGpu) {
  // [2,3,4,5] * [10,10,10,10] = [20,30,40,50]
  const int N = 4;
  auto bufs = allocBuffers(10);

  setMessage(bufs[1], {{2, 0}, {3, 0}, {4, 0}, {5, 0}});
  setMessage(bufs[4], {{10, 0}, {10, 0}, {10, 0}, {10, 0}});

  auto *prog = fhn_program_alloc(7, 0, 1);
  prog->output_ids[0] = 9;
  prog->instructions[0] = {FHN_ENCODE, 2, {1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0}};
  prog->instructions[1] = {FHN_ENCRYPT, 3, {2, 0, 0, 0}, {}, {}};
  prog->instructions[2] = {FHN_ENCODE, 5, {4, 0, 0, 0}, {0, 0, 0, 0}, {0, 0}};
  prog->instructions[3] = {FHN_ENCRYPT, 6, {5, 0, 0, 0}, {}, {}};
  prog->instructions[4] = {FHN_HMULT, 7, {3, 6, 0, 0}, {}, {}};
  prog->instructions[5] = {FHN_DECRYPT, 8, {7, 0, 0, 0}, {}, {}};
  prog->instructions[6] = {FHN_DECODE, 9, {8, 0, 0, 0}, {}, {}};

  auto t0 = std::chrono::high_resolution_clock::now();
  EXPECT_EQ(exec()->execute(ctx(), prog, bufs.data()), 0);
  auto t1 = std::chrono::high_resolution_clock::now();

  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  std::cout << "  HMULT pipeline: " << us << " us" << std::endl;

  std::vector<double> real, imag;
  readResult(bufs[9], real, imag, N);
  double expected[] = {20, 30, 40, 50};
  for (int i = 0; i < N; i++)
    EXPECT_NEAR(real[i], expected[i], 0.1) << "slot " << i;

  fhn_program_free(prog);
}

// ---------------------------------------------------------------------------
// Scalar kernel tests — demonstrates how easy it is to add new kernels.
//
// The cheddar developer adds:
//   1. A helper in ScalarOps.h  (3 lines of C++)
//   2. A kernel function        (4 lines wrapping the helper)
//   3. One line in the kernel table
//
// That's it. fhenomenon handles the rest.
// ---------------------------------------------------------------------------

TEST_F(CheddarGpuTest, AddScalar) {
  // [1,2,3,4] + 100.0 = [101,102,103,104]
  const int N = 4;
  auto bufs = allocBuffers(8);

  setMessage(bufs[1], {{1, 0}, {2, 0}, {3, 0}, {4, 0}});

  // encode -> encrypt -> add_cs(scalar=100.0) -> decrypt -> decode
  auto *prog = fhn_program_alloc(5, 0, 1);
  prog->output_ids[0] = 6;
  prog->instructions[0] = {FHN_ENCODE, 2, {1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0}};
  prog->instructions[1] = {FHN_ENCRYPT, 3, {2, 0, 0, 0}, {}, {}};
  prog->instructions[2] = {FHN_ADD_CS, 4, {3, 0, 0, 0}, {}, {100.0, 0}};
  prog->instructions[3] = {FHN_DECRYPT, 5, {4, 0, 0, 0}, {}, {}};
  prog->instructions[4] = {FHN_DECODE, 6, {5, 0, 0, 0}, {}, {}};

  auto t0 = std::chrono::high_resolution_clock::now();
  EXPECT_EQ(exec()->execute(ctx(), prog, bufs.data()), 0);
  auto t1 = std::chrono::high_resolution_clock::now();

  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  std::cout << "  ADD_CS pipeline: " << us << " us" << std::endl;

  std::vector<double> real, imag;
  readResult(bufs[6], real, imag, N);
  for (int i = 0; i < N; i++)
    EXPECT_NEAR(real[i], (i + 1) + 100.0, 1e-2) << "slot " << i;

  fhn_program_free(prog);
}

TEST_F(CheddarGpuTest, MultScalar) {
  // [10,20,30,40] * 0.5 = [5,10,15,20]
  const int N = 4;
  auto bufs = allocBuffers(8);

  setMessage(bufs[1], {{10, 0}, {20, 0}, {30, 0}, {40, 0}});

  auto *prog = fhn_program_alloc(5, 0, 1);
  prog->output_ids[0] = 6;
  prog->instructions[0] = {FHN_ENCODE, 2, {1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0}};
  prog->instructions[1] = {FHN_ENCRYPT, 3, {2, 0, 0, 0}, {}, {}};
  prog->instructions[2] = {FHN_MULT_CS, 4, {3, 0, 0, 0}, {}, {0.5, 0}};
  prog->instructions[3] = {FHN_DECRYPT, 5, {4, 0, 0, 0}, {}, {}};
  prog->instructions[4] = {FHN_DECODE, 6, {5, 0, 0, 0}, {}, {}};

  auto t0 = std::chrono::high_resolution_clock::now();
  EXPECT_EQ(exec()->execute(ctx(), prog, bufs.data()), 0);
  auto t1 = std::chrono::high_resolution_clock::now();

  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  std::cout << "  MULT_CS pipeline: " << us << " us" << std::endl;

  std::vector<double> real, imag;
  readResult(bufs[6], real, imag, N);
  double expected[] = {5, 10, 15, 20};
  for (int i = 0; i < N; i++)
    EXPECT_NEAR(real[i], expected[i], 0.1) << "slot " << i;

  fhn_program_free(prog);
}

TEST_F(CheddarGpuTest, ChainedScalarOps) {
  // (([1,2,3,4] + 10.0) * 2.0) = [22,24,26,28]
  const int N = 4;
  auto bufs = allocBuffers(9);

  setMessage(bufs[1], {{1, 0}, {2, 0}, {3, 0}, {4, 0}});

  // encode -> encrypt -> add_cs(10) -> mult_cs(2) -> decrypt -> decode
  auto *prog = fhn_program_alloc(6, 0, 1);
  prog->output_ids[0] = 7;
  prog->instructions[0] = {FHN_ENCODE, 2, {1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0}};
  prog->instructions[1] = {FHN_ENCRYPT, 3, {2, 0, 0, 0}, {}, {}};
  prog->instructions[2] = {FHN_ADD_CS, 4, {3, 0, 0, 0}, {}, {10.0, 0}};
  prog->instructions[3] = {FHN_MULT_CS, 5, {4, 0, 0, 0}, {}, {2.0, 0}};
  prog->instructions[4] = {FHN_DECRYPT, 6, {5, 0, 0, 0}, {}, {}};
  prog->instructions[5] = {FHN_DECODE, 7, {6, 0, 0, 0}, {}, {}};

  auto t0 = std::chrono::high_resolution_clock::now();
  EXPECT_EQ(exec()->execute(ctx(), prog, bufs.data()), 0);
  auto t1 = std::chrono::high_resolution_clock::now();

  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  std::cout << "  Chained ADD_CS+MULT_CS: " << us << " us" << std::endl;

  std::vector<double> real, imag;
  readResult(bufs[7], real, imag, N);
  double expected[] = {22, 24, 26, 28};
  for (int i = 0; i < N; i++)
    EXPECT_NEAR(real[i], expected[i], 0.5) << "slot " << i;

  fhn_program_free(prog);
}
