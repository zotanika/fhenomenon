#include "CKKS/Backend.h"
#include <array>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <vector>

#include <cstdlib>
#include <new>
namespace backend_allocations {
thread_local bool track = false;
thread_local size_t count = 0;
thread_local size_t fail_at = 0;
void allocation() {
  if (track && ++count == fail_at)
    throw std::bad_alloc();
}
} // namespace backend_allocations
void *operator new(size_t size) {
  backend_allocations::allocation();
  if (void *p = std::malloc(size ? size : 1))
    return p;
  throw std::bad_alloc();
}
void *operator new[](size_t size) { return ::operator new(size); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { ::operator delete(p); }
void operator delete(void *p, size_t) noexcept { ::operator delete(p); }
void operator delete[](void *p, size_t) noexcept { ::operator delete(p); }
void *operator new(size_t size, std::align_val_t alignment) {
  backend_allocations::allocation();
  void *p = nullptr;
  if (posix_memalign(&p, static_cast<size_t>(alignment), size ? size : 1) != 0)
    throw std::bad_alloc();
  return p;
}
void *operator new[](size_t size, std::align_val_t a) { return ::operator new(size, a); }
void operator delete(void *p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void *p, std::align_val_t a) noexcept { ::operator delete(p, a); }
void operator delete(void *p, size_t, std::align_val_t a) noexcept { ::operator delete(p, a); }
void operator delete[](void *p, size_t, std::align_val_t a) noexcept { ::operator delete(p, a); }

namespace {
using Context = std::unique_ptr<FhnBackendCtx, decltype(&ckks_fhn_destroy)>;
using Client = std::unique_ptr<FhnClientCtx, decltype(&ckks_fhn_client_destroy)>;
struct Buffer {
  FhnBackendCtx *ctx;
  FhnBuffer *ptr;
  explicit Buffer(FhnBackendCtx *c) : ctx(c), ptr(ckks_fhn_buffer_alloc(c)) {}
  ~Buffer() { ckks_fhn_buffer_free(ctx, ptr); }
};
FhnKernelFn kernel(FhnBackendCtx *ctx, FhnOpCode op) {
  auto *table = ckks_fhn_get_kernels(ctx);
  if (table)
    for (uint32_t i = 0; i < table->num_kernels; ++i)
      if (table->kernels[i].opcode == op)
        return table->kernels[i].fn;
  return nullptr;
}
FhnProgram program(FhnInstruction *inst, uint32_t count) {
  return FhnProgram{FHN_ABI_VERSION, count, inst, 0, nullptr, 0, nullptr};
}
TEST(CkksBackend, RealFractionalArithmeticAndLevelAccounting) {
  Context ctx(ckks_fhn_create(nullptr), ckks_fhn_destroy);
  ASSERT_TRUE(ctx);
  Client client(ckks_fhn_client_create(ctx.get()), ckks_fhn_client_destroy);
  ASSERT_TRUE(client);
  EXPECT_EQ(ckks_fhn_client_create(ctx.get()), nullptr);
  Buffer x(ctx.get()), y(ctx.get()), out(ctx.get());
  ASSERT_NE(x.ptr, nullptr);
  ASSERT_NE(y.ptr, nullptr);
  ASSERT_NE(out.ptr, nullptr);
  const std::array<double, 4> a{0.5, -1.25, 2, 0.125}, b{1.5, 2, -0.75, -4};
  ASSERT_EQ(ckks_fhn_client_encrypt_vec_f64(client.get(), x.ptr, a.data(), 4), 0);
  ASSERT_EQ(ckks_fhn_client_encrypt_vec_f64(client.get(), y.ptr, b.data(), 4), 0);
  FhnInstruction inst{};
  inst.opcode = FHN_HMULT;
  auto p = program(&inst, 1);
  ASSERT_EQ(ckks_fhn_client_prepare(client.get(), &p), 0);
  auto multiply = kernel(ctx.get(), FHN_HMULT);
  ASSERT_NE(multiply, nullptr);
  const FhnBuffer *operands[2]{x.ptr, y.ptr};
  backend_allocations::count = 0;
  backend_allocations::track = true;
  const int rc = multiply(ctx.get(), out.ptr, operands, nullptr, nullptr);
  backend_allocations::track = false;
  ASSERT_EQ(rc, 0);
  EXPECT_EQ(backend_allocations::count, 0u);
  FhnCkksBufferInfo info{};
  ASSERT_EQ(ckks_fhn_buffer_info(ctx.get(), out.ptr, &info), 0);
  EXPECT_EQ(info.level, 1);
  EXPECT_EQ(info.components, 2u);
  EXPECT_EQ(info.logical_slots, 4u);
  EXPECT_LT(info.live_payload_bytes, info.reserved_bytes);
  EXPECT_EQ(ckks_fhn_level_bytes(ctx.get(), 0), info.reserved_bytes);
  EXPECT_EQ(ckks_fhn_level_bytes(ctx.get(), 2), info.reserved_bytes);
  std::array<double, 4> values{};
  ASSERT_EQ(ckks_fhn_client_decrypt_vec_f64(client.get(), out.ptr, values.data(), 4), 0);
  for (size_t i = 0; i < 4; ++i)
    EXPECT_NEAR(values[i], a[i] * b[i], 1e-5);
  const FhnBuffer *square[2]{out.ptr, out.ptr};
  ASSERT_EQ(multiply(ctx.get(), out.ptr, square, nullptr, nullptr), 0);
  ASSERT_EQ(ckks_fhn_client_decrypt_vec_f64(client.get(), out.ptr, values.data(), 4), 0);
  for (size_t i = 0; i < 4; ++i)
    EXPECT_NEAR(values[i], std::pow(a[i] * b[i], 2), 1e-4);
  ASSERT_EQ(ckks_fhn_buffer_info(ctx.get(), out.ptr, &info), 0);
  EXPECT_EQ(info.level, 0);
  EXPECT_NE(multiply(ctx.get(), out.ptr, square, nullptr, nullptr), 0);
  ASSERT_EQ(ckks_fhn_client_decrypt_vec_f64(client.get(), out.ptr, values.data(), 4), 0);
  for (size_t i = 0; i < 4; ++i)
    EXPECT_NEAR(values[i], std::pow(a[i] * b[i], 2), 1e-4);
}
TEST(CkksBackend, ComplexRoundTripAndArithmetic) {
  Context ctx(ckks_fhn_create(nullptr), ckks_fhn_destroy);
  ASSERT_TRUE(ctx);
  Client client(ckks_fhn_client_create(ctx.get()), ckks_fhn_client_destroy);
  ASSERT_TRUE(client);
  Buffer x(ctx.get()), y(ctx.get()), real(ctx.get()), out(ctx.get());
  const std::array<FhnComplex64, 4> a{{{2.5, -3}, {-0.75, 0.5}, {0, 4}, {1, -0.125}}};
  const std::array<FhnComplex64, 4> b{{{0.5, 2}, {-2, -0.25}, {1.5, 0}, {-3, 0.5}}};
  const std::array<double, 4> r{0.5, -1.25, 2, 0.125};
  FhnClientEncryptVectorComplex64Fn encrypt = ckks_fhn_client_encrypt_vec_c64;
  FhnClientDecryptVectorComplex64Fn decrypt = ckks_fhn_client_decrypt_vec_c64;
  ASSERT_EQ(encrypt(client.get(), x.ptr, a.data(), a.size()), 0);
  ASSERT_EQ(encrypt(client.get(), y.ptr, b.data(), b.size()), 0);
  ASSERT_EQ(ckks_fhn_client_encrypt_vec_f64(client.get(), real.ptr, r.data(), r.size()), 0);
  auto complex = [](FhnComplex64 z) { return std::complex<double>(z.real, z.imag); };
  auto check = [&](const FhnBuffer *buffer, auto expected) {
    std::array<FhnComplex64, 4> values{};
    ASSERT_EQ(decrypt(client.get(), buffer, values.data(), values.size()), 0);
    for (size_t i = 0; i < values.size(); ++i) {
      EXPECT_NEAR(values[i].real, expected(i).real(), 1e-5) << "slot " << i;
      EXPECT_NEAR(values[i].imag, expected(i).imag(), 1e-5) << "slot " << i;
    }
  };
  check(x.ptr, [&](size_t i) { return complex(a[i]); });
  check(real.ptr, [&](size_t i) { return std::complex<double>(r[i]); });
  std::array<double, 4> projected{};
  ASSERT_EQ(ckks_fhn_client_decrypt_vec_f64(client.get(), x.ptr, projected.data(), projected.size()), 0);
  for (size_t i = 0; i < a.size(); ++i)
    EXPECT_NEAR(projected[i], a[i].real, 1e-5);

  FhnInstruction inst[3]{};
  inst[0].opcode = FHN_CONJUGATE;
  inst[1].opcode = FHN_ROTATE;
  inst[1].params[0] = 1;
  inst[2].opcode = FHN_HMULT;
  auto p = program(inst, 3);
  ASSERT_EQ(ckks_fhn_client_prepare(client.get(), &p), 0);
  const FhnBuffer *args[]{x.ptr, y.ptr};
  auto conjugate = kernel(ctx.get(), FHN_CONJUGATE);
  auto rotate = kernel(ctx.get(), FHN_ROTATE);
  auto multiply = kernel(ctx.get(), FHN_HMULT);
  ASSERT_NE(conjugate, nullptr);
  ASSERT_NE(rotate, nullptr);
  ASSERT_NE(multiply, nullptr);
  ASSERT_EQ(conjugate(ctx.get(), out.ptr, args, inst[0].params, nullptr), 0);
  check(out.ptr, [&](size_t i) { return std::conj(complex(a[i])); });
  ASSERT_EQ(rotate(ctx.get(), out.ptr, args, inst[1].params, nullptr), 0);
  check(out.ptr, [&](size_t i) { return complex(a[(i + 1) % a.size()]); });
  ASSERT_EQ(multiply(ctx.get(), out.ptr, args, inst[2].params, nullptr), 0);
  check(out.ptr, [&](size_t i) { return complex(a[i]) * complex(b[i]); });
  args[1] = real.ptr;
  ASSERT_EQ(multiply(ctx.get(), out.ptr, args, inst[2].params, nullptr), 0);
  check(out.ptr, [&](size_t i) { return complex(a[i]) * r[i]; });
}

TEST(CkksBackend, ComplexValidationLeavesOutputsUnchanged) {
  Context ctx(ckks_fhn_create(nullptr), ckks_fhn_destroy);
  ASSERT_TRUE(ctx);
  Client client(ckks_fhn_client_create(ctx.get()), ckks_fhn_client_destroy);
  ASSERT_TRUE(client);
  Context other(ckks_fhn_create(nullptr), ckks_fhn_destroy);
  ASSERT_TRUE(other);
  Buffer input(ctx.get()), empty(ctx.get()), foreign(other.get());
  const std::array<FhnComplex64, 2> slots{{{2.5, -3}, {-0.75, 0.5}}};
  auto encrypt = ckks_fhn_client_encrypt_vec_c64;
  auto decrypt = ckks_fhn_client_decrypt_vec_c64;
  ASSERT_EQ(encrypt(client.get(), input.ptr, slots.data(), slots.size()), 0);
  EXPECT_NE(encrypt(nullptr, input.ptr, slots.data(), 2), 0);
  EXPECT_NE(encrypt(client.get(), nullptr, slots.data(), 2), 0);
  EXPECT_NE(encrypt(client.get(), foreign.ptr, slots.data(), 2), 0);
  EXPECT_NE(encrypt(client.get(), input.ptr, nullptr, 2), 0);
  EXPECT_NE(encrypt(client.get(), input.ptr, slots.data(), 0), 0);
  const std::array<FhnComplex64, 3> uneven{};
  EXPECT_NE(encrypt(client.get(), input.ptr, uneven.data(), uneven.size()), 0);
  // The default degree is 8192, with 4096 physical slots.
  const std::vector<FhnComplex64> oversized(8192);
  EXPECT_NE(encrypt(client.get(), input.ptr, oversized.data(), oversized.size()), 0);
  for (const FhnComplex64 invalid :
       {FhnComplex64{0, std::numeric_limits<double>::infinity()},
        FhnComplex64{std::numeric_limits<double>::quiet_NaN(), 0}, FhnComplex64{0, 0x1p63}, FhnComplex64{-0x1p63, 0}})
    EXPECT_NE(encrypt(client.get(), input.ptr, &invalid, 1), 0);
  std::array<FhnComplex64, 3> actual{{{7, -7}, {9, -9}, {11, -11}}};
  EXPECT_NE(decrypt(nullptr, input.ptr, actual.data(), 2), 0);
  EXPECT_NE(decrypt(client.get(), nullptr, actual.data(), 2), 0);
  EXPECT_NE(decrypt(client.get(), foreign.ptr, actual.data(), 2), 0);
  EXPECT_NE(decrypt(client.get(), empty.ptr, actual.data(), 2), 0);
  EXPECT_NE(decrypt(client.get(), input.ptr, nullptr, 2), 0);
  EXPECT_NE(decrypt(client.get(), input.ptr, actual.data(), 1), 0);
  EXPECT_NE(decrypt(client.get(), input.ptr, actual.data(), 3), 0);
  EXPECT_EQ(actual[0].real, 7);
  EXPECT_EQ(actual[0].imag, -7);
  EXPECT_EQ(actual[1].real, 9);
  EXPECT_EQ(actual[1].imag, -9);
  EXPECT_EQ(actual[2].real, 11);
  EXPECT_EQ(actual[2].imag, -11);
  ASSERT_EQ(decrypt(client.get(), input.ptr, actual.data(), slots.size()), 0);
  for (size_t i = 0; i < slots.size(); ++i) {
    EXPECT_NEAR(actual[i].real, slots[i].real, 1e-5);
    EXPECT_NEAR(actual[i].imag, slots[i].imag, 1e-5);
  }
}
TEST(CkksBackend, ComplexFullSlotCapacity) {
  Context ctx(ckks_fhn_create(nullptr), ckks_fhn_destroy);
  ASSERT_TRUE(ctx);
  Client client(ckks_fhn_client_create(ctx.get()), ckks_fhn_client_destroy);
  ASSERT_TRUE(client);
  Buffer input(ctx.get());
  std::vector<FhnComplex64> slots(4096);
  for (size_t i = 0; i < slots.size(); ++i)
    slots[i] = {static_cast<double>(i % 9) - 4, static_cast<double>(i % 13) - 6};
  std::vector<FhnComplex64> actual(slots.size());
  ASSERT_EQ(ckks_fhn_client_encrypt_vec_c64(client.get(), input.ptr, slots.data(), slots.size()), 0);
  ASSERT_EQ(ckks_fhn_client_decrypt_vec_c64(client.get(), input.ptr, actual.data(), actual.size()), 0);
  for (size_t i = 0; i < slots.size(); ++i) {
    EXPECT_NEAR(actual[i].real, slots[i].real, 1e-5) << "slot " << i;
    EXPECT_NEAR(actual[i].imag, slots[i].imag, 1e-5) << "slot " << i;
  }
}
TEST(CkksBackend, KeysExpireWithoutSecretInComputeContext) {
  Context ctx(ckks_fhn_create(nullptr), ckks_fhn_destroy);
  ASSERT_TRUE(ctx);
  Client client(ckks_fhn_client_create(ctx.get()), ckks_fhn_client_destroy);
  ASSERT_TRUE(client);
  Buffer x(ctx.get()), out(ctx.get());
  const double value = 0.75;
  ASSERT_EQ(ckks_fhn_client_encrypt_vec_f64(client.get(), x.ptr, &value, 1), 0);
  auto multiply = kernel(ctx.get(), FHN_HMULT);
  auto add = kernel(ctx.get(), FHN_ADD_CC);
  ASSERT_NE(multiply, nullptr);
  ASSERT_NE(add, nullptr);
  const FhnBuffer *args[2]{x.ptr, x.ptr};
  EXPECT_NE(multiply(ctx.get(), out.ptr, args, nullptr, nullptr), 0);
  FhnInstruction inst{};
  inst.opcode = FHN_HMULT;
  auto p = program(&inst, 1);
  ASSERT_EQ(ckks_fhn_client_prepare(client.get(), &p), 0);
  ASSERT_EQ(multiply(ctx.get(), out.ptr, args, nullptr, nullptr), 0);
  client.reset();
  EXPECT_NE(multiply(ctx.get(), out.ptr, args, nullptr, nullptr), 0);
  EXPECT_EQ(add(ctx.get(), out.ptr, args, nullptr, nullptr), 0);
}
TEST(CkksBackend, RotationsFusedAliasesAndValidation) {
  Context ctx(ckks_fhn_create(nullptr), ckks_fhn_destroy);
  ASSERT_TRUE(ctx);
  Client client(ckks_fhn_client_create(ctx.get()), ckks_fhn_client_destroy);
  ASSERT_TRUE(client);
  Buffer x(ctx.get()), y(ctx.get());
  const std::array<double, 4> a{0.5, -1.25, 2, 0.125};
  ASSERT_EQ(ckks_fhn_client_encrypt_vec_f64(client.get(), x.ptr, a.data(), 4), 0);
  ASSERT_EQ(ckks_fhn_client_encrypt_vec_f64(client.get(), y.ptr, a.data(), 4), 0);
  FhnInstruction inst{};
  inst.opcode = FHN_HROT_ADD;
  inst.params[0] = -1;
  auto p = program(&inst, 1);
  ASSERT_EQ(ckks_fhn_client_prepare(client.get(), &p), 0);
  const FhnBuffer *args[2]{x.ptr, y.ptr};
  auto rotate_add = kernel(ctx.get(), FHN_HROT_ADD);
  ASSERT_NE(rotate_add, nullptr);
  backend_allocations::count = 0;
  backend_allocations::track = true;
  const int rc = rotate_add(ctx.get(), y.ptr, args, inst.params, nullptr);
  backend_allocations::track = false;
  ASSERT_EQ(rc, 0);
  EXPECT_EQ(backend_allocations::count, 0u);
  std::array<double, 4> actual{};
  ASSERT_EQ(ckks_fhn_client_decrypt_vec_f64(client.get(), y.ptr, actual.data(), 4), 0);
  for (size_t i = 0; i < 4; ++i)
    EXPECT_NEAR(actual[i], a[(i + 3) % 4] + a[i], 1e-5);
  std::array<double, 4> bad = a;
  bad[1] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_NE(ckks_fhn_client_encrypt_vec_f64(client.get(), y.ptr, bad.data(), 4), 0);
  EXPECT_NE(ckks_fhn_client_encrypt_vec_f64(client.get(), y.ptr, a.data(), 3), 0);
  EXPECT_NE(ckks_fhn_client_decrypt_vec_f64(client.get(), y.ptr, actual.data(), 2), 0);
  ASSERT_EQ(ckks_fhn_client_decrypt_vec_f64(client.get(), y.ptr, actual.data(), 4), 0);
  for (size_t i = 0; i < 4; ++i)
    EXPECT_NEAR(actual[i], a[(i + 3) % 4] + a[i], 1e-5);
  Context other(ckks_fhn_create(nullptr), ckks_fhn_destroy);
  ASSERT_TRUE(other);
  Buffer foreign(other.get());
  const FhnBuffer *mixed[2]{x.ptr, foreign.ptr};
  EXPECT_NE(rotate_add(ctx.get(), y.ptr, mixed, inst.params, nullptr), 0);
}
TEST(CkksBackend, RejectsUnsupportedAndInsecureConfigurations) {
  EXPECT_EQ(ckks_fhn_create("{\"log_degree\":10}"), nullptr);
  EXPECT_EQ(ckks_fhn_create("{\"secret\":\"sparse\"}"), nullptr);
  EXPECT_EQ(ckks_fhn_create("{\"unknown\":1}"), nullptr);
  EXPECT_EQ(ckks_fhn_create("{\"log_degree\":-1}"), nullptr);
  EXPECT_EQ(ckks_fhn_create("[]"), nullptr);
  EXPECT_EQ(ckks_fhn_create("not-json"), nullptr);
}
TEST(CkksBackend, PreparationCanRetryEveryAllocationFailure) {
  bool exhausted = false;
  for (size_t failure = 1; failure <= 64; ++failure) {
    Context ctx(ckks_fhn_create(nullptr), ckks_fhn_destroy);
    ASSERT_TRUE(ctx);
    Client client(ckks_fhn_client_create(ctx.get()), ckks_fhn_client_destroy);
    ASSERT_TRUE(client);
    Buffer input(ctx.get()), output(ctx.get());
    const double values[]{0.25, -1.5, 2.125, 0.75};
    ASSERT_EQ(ckks_fhn_client_encrypt_vec_f64(client.get(), input.ptr, values, 4), 0);
    FhnInstruction instruction{};
    instruction.opcode = FHN_ROTATE;
    instruction.params[0] = 1;
    auto p = program(&instruction, 1);
    backend_allocations::count = 0;
    backend_allocations::fail_at = failure;
    backend_allocations::track = true;
    const int first = ckks_fhn_client_prepare(client.get(), &p);
    backend_allocations::track = false;
    const bool injected = backend_allocations::count >= failure;
    backend_allocations::fail_at = 0;
    EXPECT_EQ(first, injected ? -1 : 0) << "allocation " << failure;
    ASSERT_EQ(ckks_fhn_client_prepare(client.get(), &p), 0) << "allocation " << failure;
    auto rotate = kernel(ctx.get(), FHN_ROTATE);
    ASSERT_NE(rotate, nullptr);
    const FhnBuffer *args[]{input.ptr};
    const int result = rotate(ctx.get(), output.ptr, args, instruction.params, nullptr);
    EXPECT_EQ(result, 0) << "allocation " << failure;
    if (result == 0) {
      double actual[4]{};
      ASSERT_EQ(ckks_fhn_client_decrypt_vec_f64(client.get(), output.ptr, actual, 4), 0);
      for (size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(actual[i], values[(i + 1) % 4], 1e-5);
    }
    if (!injected) {
      exhausted = true;
      break;
    }
  }
  EXPECT_TRUE(exhausted);
}
} // namespace
