#include "Backend/Builtin.h"
#include "Utils/log.h"

#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>

// --- TFHE-rs FFI Function Declarations ---
#ifdef FHENOMENON_USE_TFHE
extern "C" {
TfheContext *tfhe_context_create();
void tfhe_context_destroy(TfheContext *ctx);
void tfhe_ciphertext_destroy(CiphertextHandle *handle);

CiphertextHandle *tfhe_encrypt_int32(TfheContext *ctx, int32_t value);
int32_t tfhe_decrypt_int32(TfheContext *ctx, CiphertextHandle *handle);
CiphertextHandle *tfhe_add_int32(TfheContext *ctx, CiphertextHandle *a, CiphertextHandle *b);
CiphertextHandle *tfhe_mul_int32(TfheContext *ctx, CiphertextHandle *a, CiphertextHandle *b);

CiphertextHandle *tfhe_bitand_int32(TfheContext *ctx, CiphertextHandle *a, CiphertextHandle *b);
CiphertextHandle *tfhe_bitor_int32(TfheContext *ctx, CiphertextHandle *a, CiphertextHandle *b);
CiphertextHandle *tfhe_bitxor_int32(TfheContext *ctx, CiphertextHandle *a, CiphertextHandle *b);
CiphertextHandle *tfhe_eq_int32(TfheContext *ctx, CiphertextHandle *a, CiphertextHandle *b);
CiphertextHandle *tfhe_lt_int32(TfheContext *ctx, CiphertextHandle *a, CiphertextHandle *b);
CiphertextHandle *tfhe_le_int32(TfheContext *ctx, CiphertextHandle *a, CiphertextHandle *b);
}
#endif
// -----------------------------------------

namespace fhenomenon {

#ifdef FHENOMENON_USE_TFHE
// Custom deleter for CiphertextHandle shared_ptr
struct CiphertextHandleDeleter {
  void operator()(CiphertextHandle *handle) const {
    if (handle) {
      tfhe_ciphertext_destroy(handle);
    }
  }
};
using SharedCiphertextHandle = std::shared_ptr<CiphertextHandle>;

// Helper to check FFI result and throw if null
inline SharedCiphertextHandle checkAndWrap(CiphertextHandle *raw, const char *opName) {
  if (!raw) {
    throw std::runtime_error(std::string("TFHE FFI operation failed: ") + opName);
  }
  return SharedCiphertextHandle(raw, CiphertextHandleDeleter());
}
#endif

BuiltinBackend::BuiltinBackend() {
  initialize();

  // Initialize FHN executor. The FHN backend context owns the ToyFHE engine
  // and its keys; everything reaches them through the data plane and kernels.
  // ctx_core_ shares ownership with buffer deleters, so the context outlives
  // any buffer still held by an entity when the backend is destroyed.
  ctx_core_ = std::shared_ptr<FhnBackendCtx>(toyfhe_fhn_create(nullptr), &toyfhe_fhn_destroy);
  fhn_ctx_ = ctx_core_.get();
  fhn_table_ = toyfhe_fhn_get_kernels(fhn_ctx_);
  fhn_executor_ = std::make_unique<FhnDefaultExecutor>(fhn_table_);
#ifndef FHENOMENON_USE_TFHE
  runtime_ = {fhn_ctx_, fhn_executor_.get(), toyfhe_fhn_buffer_alloc, toyfhe_fhn_buffer_free, ctx_core_};
#endif
}

BuiltinBackend::~BuiltinBackend() {
  fhn_executor_.reset();
  fhn_ctx_ = nullptr;
  ctx_core_.reset(); // context destroyed once the last buffer releases it
#ifdef FHENOMENON_USE_TFHE
  if (context_) {
    tfhe_context_destroy(context_);
    context_ = nullptr;
  }
#endif
}

std::shared_ptr<FhnBuffer> BuiltinBackend::makeBuffer() const {
  FhnBuffer *raw = toyfhe_fhn_buffer_alloc(fhn_ctx_);
  if (!raw) {
    throw std::runtime_error("BuiltinBackend: toyfhe_fhn_buffer_alloc failed");
  }
  // The deleter shares ownership of the context, so a buffer held by an
  // entity that outlives the backend still frees against a live context.
  auto core = ctx_core_;
  return std::shared_ptr<FhnBuffer>(raw, [core](FhnBuffer *buf) { toyfhe_fhn_buffer_free(core.get(), buf); });
}

std::shared_ptr<FhnBuffer> BuiltinBackend::bufferOf(const FhenonBase &entity, const char *opName) const {
  const auto *ct = std::any_cast<FhnCiphertext>(&entity.ciphertext_);
  if (!ct || ct->owner != this) {
    throw std::runtime_error(std::string("BuiltinBackend: ") + opName + ": operand was not encrypted by this backend");
  }
  return ct->buffer;
}

std::shared_ptr<FhnBuffer> BuiltinBackend::runSingleOp(FhnOpCode op, FhnBuffer *a, FhnBuffer *b, double fparam) const {
  const uint32_t num_inputs = b ? 2u : 1u;
  std::unique_ptr<FhnProgram, decltype(&fhn_program_free)> prog(fhn_program_alloc(1, num_inputs, 1), &fhn_program_free);
  if (!prog) {
    throw std::runtime_error("BuiltinBackend: fhn_program_alloc failed");
  }

  // Buffer ids: a = 1, optional b = 2, result = next.
  const uint32_t result_id = num_inputs + 1;
  prog->instructions[0].opcode = op;
  prog->instructions[0].result_id = result_id;
  prog->instructions[0].operands[0] = 1;
  prog->input_ids[0] = 1;
  if (b) {
    prog->instructions[0].operands[1] = 2;
    prog->input_ids[1] = 2;
  }
  prog->instructions[0].fparams[0] = fparam;
  prog->output_ids[0] = result_id;

  auto result = makeBuffer();
  FhnBuffer *buffers[4] = {nullptr, a, nullptr, nullptr};
  buffers[2] = b ? b : result.get();
  if (b) {
    buffers[3] = result.get();
  }

  const int rc = fhn_executor_->execute(fhn_ctx_, prog.get(), buffers);
  if (rc != 0) {
    throw std::runtime_error("BuiltinBackend: FHN executor failed with rc=" + std::to_string(rc));
  }
  return result;
}

void BuiltinBackend::ensureReady() const {
#ifdef FHENOMENON_USE_TFHE
  std::call_once(initFlag_, [this]() { const_cast<BuiltinBackend *>(this)->initialize(); });
#endif
  // Non-TFHE: nothing to do — the FHN context owns the engine and keys.
}

void BuiltinBackend::initialize() {
#ifdef FHENOMENON_USE_TFHE
  if (!context_) {
    context_ = tfhe_context_create();
    if (!context_) {
      throw std::runtime_error("Failed to create TFHE context");
    }
    LOG_MESSAGE("BuiltinBackend: Initialized TFHE context.");
  }
#endif
  // Non-TFHE: no-op — the FHN context owns the engine and keys.
}

void BuiltinBackend::generateKeys() {
  // Non-TFHE: no-op — the FHN context generates its keys on creation.
}

void BuiltinBackend::loadKeys([[maybe_unused]] const std::string &publicKeyPath,
                              [[maybe_unused]] const std::string &secretKeyPath) {
  LOG_MESSAGE("BuiltinBackend: loadKeys is a no-op.");
}

void BuiltinBackend::saveKeys([[maybe_unused]] const std::string &publicKeyPath,
                              [[maybe_unused]] const std::string &secretKeyPath) {
  LOG_MESSAGE("BuiltinBackend: saveKeys is a no-op.");
}

void BuiltinBackend::transform(FhenonBase &entity, [[maybe_unused]] const Parameter &params) const {
  ensureReady();
  auto type = entity.type();

#ifdef FHENOMENON_USE_TFHE
  if (type == typeid(int)) {
    auto &derivedEntity = dynamic_cast<Fhenon<int> &>(entity);
    int32_t value = derivedEntity.getValue();

    auto handle = checkAndWrap(tfhe_encrypt_int32(context_, value), "encrypt_int32");

    entity.ciphertext_ = handle;
    entity.isEncrypted_ = true;
    LOG_MESSAGE("BuiltinBackend: Encrypted int value " << value << " (TFHE)");
  } else {
    throw std::runtime_error("BuiltinBackend: Unsupported type for encryption (only int supported with TFHE)");
  }
#else
  // Host-side data plane: encryption never goes through the kernel table.
  if (type == typeid(int)) {
    auto &derivedEntity = dynamic_cast<Fhenon<int> &>(entity);
    auto buffer = makeBuffer();
    if (toyfhe_fhn_encrypt_i64(fhn_ctx_, buffer.get(), derivedEntity.getValue()) != 0) {
      throw std::runtime_error("BuiltinBackend: toyfhe_fhn_encrypt_i64 failed");
    }
    entity.ciphertext_ = FhnCiphertext{buffer, this};
    entity.isEncrypted_ = true;
    LOG_MESSAGE("BuiltinBackend: Encrypted int value " << derivedEntity.getValue());
  } else if (type == typeid(double)) {
    auto &derivedEntity = dynamic_cast<Fhenon<double> &>(entity);
    auto buffer = makeBuffer();
    if (toyfhe_fhn_encrypt_f64(fhn_ctx_, buffer.get(), derivedEntity.getValue()) != 0) {
      throw std::runtime_error("BuiltinBackend: toyfhe_fhn_encrypt_f64 failed");
    }
    entity.ciphertext_ = FhnCiphertext{buffer, this};
    entity.isEncrypted_ = true;
    LOG_MESSAGE("BuiltinBackend: Encrypted double value " << derivedEntity.getValue());
  } else if (type == typeid(float)) {
    auto &derivedEntity = dynamic_cast<Fhenon<float> &>(entity);
    auto buffer = makeBuffer();
    if (toyfhe_fhn_encrypt_f64(fhn_ctx_, buffer.get(), static_cast<double>(derivedEntity.getValue())) != 0) {
      throw std::runtime_error("BuiltinBackend: toyfhe_fhn_encrypt_f64 failed");
    }
    entity.ciphertext_ = FhnCiphertext{buffer, this};
    entity.isEncrypted_ = true;
    LOG_MESSAGE("BuiltinBackend: Encrypted float value " << derivedEntity.getValue());
  } else {
    throw std::runtime_error("BuiltinBackend: Unsupported type for encryption");
  }
#endif
}

namespace {
template <typename T>
#ifdef FHENOMENON_USE_TFHE
std::shared_ptr<FhenonBase> makeResultFhenon(const Fhenon<T> &reference, SharedCiphertextHandle ciphertext) {
  auto resultFhenon = std::make_shared<Fhenon<T>>(0);
  resultFhenon->ciphertext_ = ciphertext;
  resultFhenon->isEncrypted_ = true;
  resultFhenon->setProfile(reference.getProfile());
  return resultFhenon;
}
#else
std::shared_ptr<FhenonBase> makeResultFhenon(const Fhenon<T> &reference, FhnCiphertext ciphertext) {
  auto resultFhenon = std::make_shared<Fhenon<T>>(0);
  resultFhenon->ciphertext_ = std::move(ciphertext);
  resultFhenon->isEncrypted_ = true;
  resultFhenon->setProfile(reference.getProfile());
  return resultFhenon;
}

// Dispatch on the reference entity's runtime type so the result Fhenon<T>
// matches it; the ciphertext itself is type-agnostic on the FHN data plane.
std::shared_ptr<FhenonBase> makeTypedResult(const FhenonBase &reference, FhnCiphertext ciphertext, const char *opName) {
  auto type = reference.type();
  if (type == typeid(int)) {
    return makeResultFhenon(dynamic_cast<const Fhenon<int> &>(reference), std::move(ciphertext));
  }
  if (type == typeid(double)) {
    return makeResultFhenon(dynamic_cast<const Fhenon<double> &>(reference), std::move(ciphertext));
  }
  if (type == typeid(float)) {
    return makeResultFhenon(dynamic_cast<const Fhenon<float> &>(reference), std::move(ciphertext));
  }
  throw std::runtime_error(std::string("BuiltinBackend: Unsupported type for ") + opName);
}
#endif
} // namespace

std::shared_ptr<FhenonBase> BuiltinBackend::add(const FhenonBase &a, const FhenonBase &b) const {
  ensureReady();

  if (!a.isEncrypted_ || !b.isEncrypted_ || !a.ciphertext_.has_value() || !b.ciphertext_.has_value()) {
    throw std::runtime_error("BuiltinBackend: Cannot add unencrypted Fhenon values");
  }

#ifdef FHENOMENON_USE_TFHE
  if (a.type() == typeid(int) && b.type() == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Fhenon<int> &>(a);
    auto ctA = std::any_cast<SharedCiphertextHandle>(a.ciphertext_);
    auto ctB = std::any_cast<SharedCiphertextHandle>(b.ciphertext_);

    auto result = checkAndWrap(tfhe_add_int32(context_, ctA.get(), ctB.get()), "add_int32");

    LOG_MESSAGE("BuiltinBackend: Performed addition (TFHE)");
    return makeResultFhenon(derivedA, result);
  }
  throw std::runtime_error("BuiltinBackend: Unsupported types for add (TFHE only supports int)");
#else
  auto ctA = bufferOf(a, "add");
  auto ctB = bufferOf(b, "add");
  auto result = runSingleOp(FHN_ADD_CC, ctA.get(), ctB.get(), 0.0);
  LOG_MESSAGE("BuiltinBackend: Performed FHN addition (ADD_CC)");
  return makeTypedResult(a, FhnCiphertext{std::move(result), this}, "add");
#endif
}

std::shared_ptr<FhenonBase> BuiltinBackend::multiply(const FhenonBase &a, const FhenonBase &b) const {
  ensureReady();
  if (!a.isEncrypted_ || !b.isEncrypted_ || !a.ciphertext_.has_value() || !b.ciphertext_.has_value()) {
    throw std::runtime_error("BuiltinBackend: Cannot multiply unencrypted Fhenon values");
  }

#ifdef FHENOMENON_USE_TFHE
  if (a.type() == typeid(int) && b.type() == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Fhenon<int> &>(a);
    auto ctA = std::any_cast<SharedCiphertextHandle>(a.ciphertext_);
    auto ctB = std::any_cast<SharedCiphertextHandle>(b.ciphertext_);

    auto result = checkAndWrap(tfhe_mul_int32(context_, ctA.get(), ctB.get()), "mul_int32");

    LOG_MESSAGE("BuiltinBackend: Performed multiplication (TFHE)");
    return makeResultFhenon(derivedA, result);
  }
  throw std::runtime_error("BuiltinBackend: Unsupported types for multiply (TFHE only supports int)");
#else
  if (a.type() != b.type()) {
    throw std::runtime_error("BuiltinBackend: Cannot multiply Fhenon values of different types");
  }

  auto ctA = bufferOf(a, "multiply");
  auto ctB = bufferOf(b, "multiply");
  auto result = runSingleOp(FHN_HMULT, ctA.get(), ctB.get(), 0.0);
  LOG_MESSAGE("BuiltinBackend: Performed FHN multiplication (HMULT)");
  return makeTypedResult(a, FhnCiphertext{std::move(result), this}, "multiply");
#endif
}

std::shared_ptr<FhenonBase> BuiltinBackend::addPlain([[maybe_unused]] const FhenonBase &a,
                                                     [[maybe_unused]] double scalar) {
  ensureReady();
#ifdef FHENOMENON_USE_TFHE
  throw std::runtime_error("BuiltinBackend: addPlain not implemented for TFHE");
#else
  if (!a.isEncrypted_ || !a.ciphertext_.has_value()) {
    throw std::runtime_error("BuiltinBackend: Cannot add plain to unencrypted Fhenon value");
  }
  auto ctA = bufferOf(a, "addPlain");
  auto result = runSingleOp(FHN_ADD_CS, ctA.get(), nullptr, scalar);
  LOG_MESSAGE("BuiltinBackend: FHN addPlain (ADD_CS) with scalar " << scalar);
  return makeTypedResult(a, FhnCiphertext{std::move(result), this}, "addPlain");
#endif
}

std::shared_ptr<FhenonBase> BuiltinBackend::multiplyPlain([[maybe_unused]] const FhenonBase &a,
                                                          [[maybe_unused]] double scalar) {
  ensureReady();
#ifdef FHENOMENON_USE_TFHE
  throw std::runtime_error("BuiltinBackend: multiplyPlain not implemented for TFHE");
#else
  if (!a.isEncrypted_ || !a.ciphertext_.has_value()) {
    throw std::runtime_error("BuiltinBackend: Cannot multiply plain with unencrypted Fhenon value");
  }
  auto ctA = bufferOf(a, "multiplyPlain");
  auto result = runSingleOp(FHN_MULT_CS, ctA.get(), nullptr, scalar);
  LOG_MESSAGE("BuiltinBackend: FHN multiplyPlain (MULT_CS) with scalar " << scalar);
  return makeTypedResult(a, FhnCiphertext{std::move(result), this}, "multiplyPlain");
#endif
}

std::any BuiltinBackend::decrypt(const FhenonBase &entity) const {
  ensureReady();

#ifdef FHENOMENON_USE_TFHE
  if (!entity.isEncrypted_ || !entity.ciphertext_.has_value()) {
    // Fallback for unencrypted values
    auto type = entity.type();
    if (type == typeid(int)) {
      return dynamic_cast<const Fhenon<int> &>(entity).getValue();
    }
    return {};
  }

  auto type = entity.type();
  if (type == typeid(int)) {
    try {
      auto handle = std::any_cast<SharedCiphertextHandle>(entity.ciphertext_);
      if (!handle || !handle.get()) {
        throw std::runtime_error("BuiltinBackend: Ciphertext handle is null");
      }
      int32_t value = tfhe_decrypt_int32(context_, handle.get());
      LOG_MESSAGE("BuiltinBackend: Decrypted int value " << value << " (TFHE)");
      return value;
    } catch (const std::bad_any_cast &e) {
      throw std::runtime_error(std::string("BuiltinBackend: Failed to cast ciphertext handle: ") + e.what());
    } catch (const std::exception &e) {
      throw std::runtime_error(std::string("BuiltinBackend: Decryption failed: ") + e.what());
    }
  }
  throw std::runtime_error("BuiltinBackend: Unsupported type for decryption (TFHE only supports int)");
#else
  if (!entity.isEncrypted_ || !entity.ciphertext_.has_value()) {
    auto type = entity.type();
    if (type == typeid(int)) {
      const auto &derivedEntity = dynamic_cast<const Fhenon<int> &>(entity);
      return derivedEntity.getValue();
    } else if (type == typeid(double)) {
      const auto &derivedEntity = dynamic_cast<const Fhenon<double> &>(entity);
      return derivedEntity.getValue();
    } else if (type == typeid(float)) {
      const auto &derivedEntity = dynamic_cast<const Fhenon<float> &>(entity);
      return derivedEntity.getValue();
    }
    return {};
  }

  auto type = entity.type();
  auto ct = bufferOf(entity, "decrypt");
  if (type == typeid(int)) {
    int64_t value = 0;
    if (toyfhe_fhn_decrypt_i64(fhn_ctx_, ct.get(), &value) != 0) {
      throw std::runtime_error("BuiltinBackend: toyfhe_fhn_decrypt_i64 failed");
    }
    LOG_MESSAGE("BuiltinBackend: Decrypted int value " << value);
    return static_cast<int>(value);
  } else if (type == typeid(double)) {
    double value = 0.0;
    if (toyfhe_fhn_decrypt_f64(fhn_ctx_, ct.get(), &value) != 0) {
      throw std::runtime_error("BuiltinBackend: toyfhe_fhn_decrypt_f64 failed");
    }
    LOG_MESSAGE("BuiltinBackend: Decrypted double value " << value);
    return value;
  } else if (type == typeid(float)) {
    double value = 0.0;
    if (toyfhe_fhn_decrypt_f64(fhn_ctx_, ct.get(), &value) != 0) {
      throw std::runtime_error("BuiltinBackend: toyfhe_fhn_decrypt_f64 failed");
    }
    LOG_MESSAGE("BuiltinBackend: Decrypted float value " << static_cast<float>(value));
    return static_cast<float>(value);
  }
  return {};
#endif
}

#ifdef FHENOMENON_USE_TFHE
std::shared_ptr<FhenonBase> BuiltinBackend::executeBinaryTfheOp(const FhenonBase &a, const FhenonBase &b,
                                                                TfheBinaryOp op, const char *opName) const {
  ensureReady();
  if (a.type() == typeid(int) && b.type() == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Fhenon<int> &>(a);
    auto ctA = std::any_cast<SharedCiphertextHandle>(a.ciphertext_);
    auto ctB = std::any_cast<SharedCiphertextHandle>(b.ciphertext_);
    auto result = checkAndWrap(op(context_, ctA.get(), ctB.get()), opName);
    return makeResultFhenon(derivedA, result);
  }
  throw std::runtime_error(std::string(opName) + " only supports int type with TFHE backend");
}
#endif

std::shared_ptr<FhenonBase> BuiltinBackend::bitAnd([[maybe_unused]] const FhenonBase &a,
                                                   [[maybe_unused]] const FhenonBase &b) const {
#ifdef FHENOMENON_USE_TFHE
  return executeBinaryTfheOp(a, b, tfhe_bitand_int32, "bitwise_and");
#else
  throw std::runtime_error("Bitwise AND not supported by BuiltinBackend");
#endif
}

std::shared_ptr<FhenonBase> BuiltinBackend::bitOr([[maybe_unused]] const FhenonBase &a,
                                                  [[maybe_unused]] const FhenonBase &b) const {
#ifdef FHENOMENON_USE_TFHE
  return executeBinaryTfheOp(a, b, tfhe_bitor_int32, "bitwise_or");
#else
  throw std::runtime_error("Bitwise OR not supported by BuiltinBackend");
#endif
}

std::shared_ptr<FhenonBase> BuiltinBackend::bitXor([[maybe_unused]] const FhenonBase &a,
                                                   [[maybe_unused]] const FhenonBase &b) const {
#ifdef FHENOMENON_USE_TFHE
  return executeBinaryTfheOp(a, b, tfhe_bitxor_int32, "bitwise_xor");
#else
  throw std::runtime_error("Bitwise XOR not supported by BuiltinBackend");
#endif
}

std::shared_ptr<FhenonBase> BuiltinBackend::compareEq([[maybe_unused]] const FhenonBase &a,
                                                      [[maybe_unused]] const FhenonBase &b) const {
#ifdef FHENOMENON_USE_TFHE
  return executeBinaryTfheOp(a, b, tfhe_eq_int32, "compare_eq");
#else
  throw std::runtime_error("Equality comparison not supported by BuiltinBackend");
#endif
}

std::shared_ptr<FhenonBase> BuiltinBackend::compareLt([[maybe_unused]] const FhenonBase &a,
                                                      [[maybe_unused]] const FhenonBase &b) const {
#ifdef FHENOMENON_USE_TFHE
  return executeBinaryTfheOp(a, b, tfhe_lt_int32, "compare_lt");
#else
  throw std::runtime_error("Less than comparison not supported by BuiltinBackend");
#endif
}

std::shared_ptr<FhenonBase> BuiltinBackend::compareLe([[maybe_unused]] const FhenonBase &a,
                                                      [[maybe_unused]] const FhenonBase &b) const {
#ifdef FHENOMENON_USE_TFHE
  return executeBinaryTfheOp(a, b, tfhe_le_int32, "compare_le");
#else
  throw std::runtime_error("Less equal comparison not supported by BuiltinBackend");
#endif
}

} // namespace fhenomenon
