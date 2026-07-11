// ---------------------------------------------------------------------------
// Integration test: Cheddar-FHE GPU backend via ExternalBackend (dlopen)
//
// Demonstrates:
// 1. Loading cheddar as an FHN backend via dlopen
// 2. Encrypting inputs through the host-side data plane, running compute-only
//    CKKS programs (add, hmult, scalar ops) on GPU, decrypting outputs
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
#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <complex>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace fhenomenon;
using word = uint32_t;

// Data-plane / buffer helper function pointers (resolved once from the .so)
using BufAllocFn = FhnBuffer *(*)(FhnBackendCtx *);
using BufFreeFn = void (*)(FhnBackendCtx *, FhnBuffer *);
using BufEncryptMsgFn = int (*)(FhnBackendCtx *, FhnBuffer *);
using BufReadComplexFn = void (*)(FhnBackendCtx *, FhnBuffer *, double *, double *, int);
using EncryptF64Fn = int (*)(FhnBackendCtx *, FhnBuffer *, double);
using DecryptF64Fn = int (*)(FhnBackendCtx *, const FhnBuffer *, double *);

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
  static BufEncryptMsgFn buf_encrypt_;
  static BufReadComplexFn buf_read_;
  static EncryptF64Fn encrypt_f64_;
  static DecryptF64Fn decrypt_f64_;

  static void SetUpTestSuite() {
    std::string lib = std::string(TEST_LIB_DIR) + "/libcheddar_fhn.so";
    std::string param = std::string(PARAM_DIR) + "/testparam_light.json";

    // Splice the rotation-key request into the parameter config: the backend
    // only prepares rotation keys listed under "rotation_keys", and its
    // rotation kernels reject every other distance. The rotation tests use a
    // 4-slot message rotated by +1 and -1; cheddar reduces distances modulo
    // the ciphertext's slot count, so the required keys are 1 and 3
    // (rotate(-1) == rotate(+3) on 4 slots).
    std::ifstream param_file(param);
    ASSERT_TRUE(param_file.is_open()) << "cannot open " << param;
    nlohmann::json config = nlohmann::json::parse(param_file);
    config["rotation_keys"] = {1, 3};
    std::string config_str = config.dump();

    auto t0 = std::chrono::high_resolution_clock::now();
    backend_ = std::make_unique<ExternalBackend>(lib, config_str.c_str());
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "\n[  SETUP  ] Context + keys: " << ms << " ms (one-time cost)\n" << std::endl;

    dl_ = dlopen(lib.c_str(), RTLD_LAZY | RTLD_NOLOAD);
    buf_alloc_ = reinterpret_cast<BufAllocFn>(dlsym(dl_, "fhn_buffer_alloc"));
    buf_free_ = reinterpret_cast<BufFreeFn>(dlsym(dl_, "fhn_buffer_free"));
    buf_encrypt_ = reinterpret_cast<BufEncryptMsgFn>(dlsym(dl_, "cheddar_fhn_buffer_encrypt_message"));
    buf_read_ = reinterpret_cast<BufReadComplexFn>(dlsym(dl_, "cheddar_fhn_buffer_read_complex"));
    encrypt_f64_ = reinterpret_cast<EncryptF64Fn>(dlsym(dl_, "fhn_encrypt_f64"));
    decrypt_f64_ = reinterpret_cast<DecryptF64Fn>(dlsym(dl_, "fhn_decrypt_f64"));
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

  // Host-side data plane: stage a message, then encode + encrypt it in place.
  void encryptMessage(FhnBuffer *buf, const std::vector<cheddar::Complex> &msg) {
    setMessage(buf, msg);
    ASSERT_EQ(buf_encrypt_(ctx(), buf), 0);
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
BufEncryptMsgFn CheddarGpuTest::buf_encrypt_ = nullptr;
BufReadComplexFn CheddarGpuTest::buf_read_ = nullptr;
EncryptF64Fn CheddarGpuTest::encrypt_f64_ = nullptr;
DecryptF64Fn CheddarGpuTest::decrypt_f64_ = nullptr;

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

// Encryption and decryption live in the host-side data plane, not in the
// instruction stream: encrypt into buffers, run a compute-only program,
// decrypt the output buffer.
TEST_F(CheddarGpuTest, DataPlaneRoundTrip) {
  auto bufs = allocBuffers(2);

  ASSERT_EQ(encrypt_f64_(ctx(), bufs[0], 2.5), 0);
  double value = 0.0;
  ASSERT_EQ(decrypt_f64_(ctx(), bufs[0], &value), 0);
  EXPECT_NEAR(value, 2.5, 1e-3);

  // Decrypting an empty (non-ciphertext) buffer is an error, not a zero.
  EXPECT_NE(decrypt_f64_(ctx(), bufs[1], &value), 0);
}

TEST_F(CheddarGpuTest, AddVectors) {
  // [1,2,3,4] + [10,20,30,40] = [11,22,33,44]  (inputs encrypted host-side)
  const int N = 4;
  auto bufs = allocBuffers(4);

  encryptMessage(bufs[1], {{1, 0}, {2, 0}, {3, 0}, {4, 0}});
  encryptMessage(bufs[2], {{10, 0}, {20, 0}, {30, 0}, {40, 0}});

  auto *prog = fhn_program_alloc(1, 2, 1);
  prog->input_ids[0] = 1;
  prog->input_ids[1] = 2;
  prog->output_ids[0] = 3;
  prog->instructions[0] = {FHN_ADD_CC, 3, {1, 2, 0, 0}, {}, {}};

  auto t0 = std::chrono::high_resolution_clock::now();
  EXPECT_EQ(exec()->execute(ctx(), prog, bufs.data()), 0);
  auto t1 = std::chrono::high_resolution_clock::now();

  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  std::cout << "  ADD_CC pipeline: " << us << " us" << std::endl;

  std::vector<double> real, imag;
  readResult(bufs[3], real, imag, N);
  for (int i = 0; i < N; i++)
    EXPECT_NEAR(real[i], (i + 1) + (i + 1) * 10, 1e-3) << "slot " << i;

  fhn_program_free(prog);
}

TEST_F(CheddarGpuTest, HMultGpu) {
  // [2,3,4,5] * [10,10,10,10] = [20,30,40,50]
  const int N = 4;
  auto bufs = allocBuffers(4);

  encryptMessage(bufs[1], {{2, 0}, {3, 0}, {4, 0}, {5, 0}});
  encryptMessage(bufs[2], {{10, 0}, {10, 0}, {10, 0}, {10, 0}});

  auto *prog = fhn_program_alloc(1, 2, 1);
  prog->input_ids[0] = 1;
  prog->input_ids[1] = 2;
  prog->output_ids[0] = 3;
  prog->instructions[0] = {FHN_HMULT, 3, {1, 2, 0, 0}, {}, {}};

  auto t0 = std::chrono::high_resolution_clock::now();
  EXPECT_EQ(exec()->execute(ctx(), prog, bufs.data()), 0);
  auto t1 = std::chrono::high_resolution_clock::now();

  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  std::cout << "  HMULT pipeline: " << us << " us" << std::endl;

  std::vector<double> real, imag;
  readResult(bufs[3], real, imag, N);
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
  auto bufs = allocBuffers(3);

  encryptMessage(bufs[1], {{1, 0}, {2, 0}, {3, 0}, {4, 0}});

  // add_cs(scalar=100.0), inputs/outputs handled by the host data plane
  auto *prog = fhn_program_alloc(1, 1, 1);
  prog->input_ids[0] = 1;
  prog->output_ids[0] = 2;
  prog->instructions[0] = {FHN_ADD_CS, 2, {1, 0, 0, 0}, {}, {100.0, 0}};

  auto t0 = std::chrono::high_resolution_clock::now();
  EXPECT_EQ(exec()->execute(ctx(), prog, bufs.data()), 0);
  auto t1 = std::chrono::high_resolution_clock::now();

  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  std::cout << "  ADD_CS pipeline: " << us << " us" << std::endl;

  std::vector<double> real, imag;
  readResult(bufs[2], real, imag, N);
  for (int i = 0; i < N; i++)
    EXPECT_NEAR(real[i], (i + 1) + 100.0, 1e-2) << "slot " << i;

  fhn_program_free(prog);
}

TEST_F(CheddarGpuTest, MultScalar) {
  // [10,20,30,40] * 0.5 = [5,10,15,20]
  const int N = 4;
  auto bufs = allocBuffers(3);

  encryptMessage(bufs[1], {{10, 0}, {20, 0}, {30, 0}, {40, 0}});

  auto *prog = fhn_program_alloc(1, 1, 1);
  prog->input_ids[0] = 1;
  prog->output_ids[0] = 2;
  prog->instructions[0] = {FHN_MULT_CS, 2, {1, 0, 0, 0}, {}, {0.5, 0}};

  auto t0 = std::chrono::high_resolution_clock::now();
  EXPECT_EQ(exec()->execute(ctx(), prog, bufs.data()), 0);
  auto t1 = std::chrono::high_resolution_clock::now();

  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  std::cout << "  MULT_CS pipeline: " << us << " us" << std::endl;

  std::vector<double> real, imag;
  readResult(bufs[2], real, imag, N);
  double expected[] = {5, 10, 15, 20};
  for (int i = 0; i < N; i++)
    EXPECT_NEAR(real[i], expected[i], 0.1) << "slot " << i;

  fhn_program_free(prog);
}

TEST_F(CheddarGpuTest, ChainedScalarOps) {
  // (([1,2,3,4] + 10.0) * 2.0) = [22,24,26,28]
  const int N = 4;
  auto bufs = allocBuffers(4);

  encryptMessage(bufs[1], {{1, 0}, {2, 0}, {3, 0}, {4, 0}});

  // add_cs(10) -> mult_cs(2)
  auto *prog = fhn_program_alloc(2, 1, 1);
  prog->input_ids[0] = 1;
  prog->output_ids[0] = 3;
  prog->instructions[0] = {FHN_ADD_CS, 2, {1, 0, 0, 0}, {}, {10.0, 0}};
  prog->instructions[1] = {FHN_MULT_CS, 3, {2, 0, 0, 0}, {}, {2.0, 0}};

  auto t0 = std::chrono::high_resolution_clock::now();
  EXPECT_EQ(exec()->execute(ctx(), prog, bufs.data()), 0);
  auto t1 = std::chrono::high_resolution_clock::now();

  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  std::cout << "  Chained ADD_CS+MULT_CS: " << us << " us" << std::endl;

  std::vector<double> real, imag;
  readResult(bufs[3], real, imag, N);
  double expected[] = {22, 24, 26, 28};
  for (int i = 0; i < N; i++)
    EXPECT_NEAR(real[i], expected[i], 0.5) << "slot " << i;

  fhn_program_free(prog);
}

// ---------------------------------------------------------------------------
// Rotation tests — the rotation keys come from the "rotation_keys" config
// entry passed in SetUpTestSuite. cheddar reduces rotation distances modulo
// the ciphertext's slot count, so a 4-slot message rotated by +1 and -1
// needs the keys for distances 1 and 3.
// ---------------------------------------------------------------------------

TEST_F(CheddarGpuTest, RotateBothDirections) {
  // rotate([1,2,3,4], +1) = [2,3,4,1] ; rotate([1,2,3,4], -1) = [4,1,2,3]
  const int N = 4;
  auto bufs = allocBuffers(4);

  encryptMessage(bufs[1], {{1, 0}, {2, 0}, {3, 0}, {4, 0}});

  auto *prog = fhn_program_alloc(2, 1, 2);
  prog->input_ids[0] = 1;
  prog->output_ids[0] = 2;
  prog->output_ids[1] = 3;
  prog->instructions[0] = {FHN_ROTATE, 2, {1, 0, 0, 0}, {1, 0, 0, 0}, {}};
  prog->instructions[1] = {FHN_ROTATE, 3, {1, 0, 0, 0}, {-1, 0, 0, 0}, {}};

  auto t0 = std::chrono::high_resolution_clock::now();
  EXPECT_EQ(exec()->execute(ctx(), prog, bufs.data()), 0);
  auto t1 = std::chrono::high_resolution_clock::now();

  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  std::cout << "  ROTATE(+1)+ROTATE(-1) pipeline: " << us << " us" << std::endl;

  std::vector<double> real, imag;
  readResult(bufs[2], real, imag, N);
  double expected_left[] = {2, 3, 4, 1};
  for (int i = 0; i < N; i++)
    EXPECT_NEAR(real[i], expected_left[i], 1e-2) << "slot " << i;

  readResult(bufs[3], real, imag, N);
  double expected_right[] = {4, 1, 2, 3};
  for (int i = 0; i < N; i++)
    EXPECT_NEAR(real[i], expected_right[i], 1e-2) << "slot " << i;

  fhn_program_free(prog);
}

TEST_F(CheddarGpuTest, RotateUnpreparedDistanceFails) {
  // No key was prepared for distance 2: the rotation kernel must reject it
  // with a non-zero rc instead of reaching cheddar's AssertTrue/std::exit.
  auto bufs = allocBuffers(3);

  encryptMessage(bufs[1], {{1, 0}, {2, 0}, {3, 0}, {4, 0}});

  auto *prog = fhn_program_alloc(1, 1, 1);
  prog->input_ids[0] = 1;
  prog->output_ids[0] = 2;
  prog->instructions[0] = {FHN_ROTATE, 2, {1, 0, 0, 0}, {2, 0, 0, 0}, {}};
  EXPECT_NE(exec()->execute(ctx(), prog, bufs.data()), 0);

  fhn_program_free(prog);
}
