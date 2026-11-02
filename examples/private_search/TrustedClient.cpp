#include "SearchWorkflow.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <dlfcn.h>
#include <limits>
#include <stdexcept>

namespace private_search {
namespace {
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
void validateEmbedding(const std::vector<double> &values) {
  if (values.size() != dimensions)
    throw std::invalid_argument("embedding must have 384 dimensions");
  double squared_norm = 0;
  for (double value : values) {
    if (!std::isfinite(value))
      throw std::invalid_argument("embedding must be finite");
    squared_norm += value * value;
  }
  if (!std::isfinite(squared_norm) || std::abs(squared_norm - 1) > 1e-4)
    throw std::invalid_argument("embedding must be L2 normalized");
}
void validateShape(const SearchShape &shape) {
  if (shape.width != width || !shape.document_count || !shape.documents_per_ciphertext ||
      shape.documents_per_ciphertext > tile_capacity)
    throw std::invalid_argument("invalid example score layout");
}
} // namespace

struct TrustedClient::Impl {
  // Members are destroyed in reverse order: the client key and all function
  // pointers remain covered by a live context/library lease until release.
  std::unique_ptr<void, decltype(&dlclose)> library{nullptr, dlclose};
  NativeServices services;
  std::unique_ptr<FhnClientCtx, FhnClientDestroyFn> client{nullptr, nullptr};
  std::unique_ptr<NativeRuntime> input_native;
  FhnClientEncryptVectorDoubleFn encrypt = nullptr;
  FhnClientDecryptVectorDoubleFn decrypt = nullptr;
  double key_setup_ms = 0;
  template <class T> T symbol(const char *name) {
    auto result = reinterpret_cast<T>(dlsym(library.get(), name));
    require(result != nullptr, "missing native client service");
    return result;
  }
  Impl(fhenomenon::runtime::Runtime &runtime, const std::string &path) {
    const auto start = std::chrono::steady_clock::now();
    library.reset(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL));
    require(bool(library), "cannot load native CKKS library");
    services = fhenomenon::runtime::loadNativeServices(path);
    services.compile = nullptr;
    services.run = nullptr;
    services.executable_stats = nullptr;
    services.executable_free = nullptr;
    auto create = symbol<FhnClientCreateFn>("ckks_fhn_client_create");
    client = {create(services.context.get()), symbol<FhnClientDestroyFn>("ckks_fhn_client_destroy")};
    require(bool(client), "client creation failed");
    encrypt = symbol<FhnClientEncryptVectorDoubleFn>("ckks_fhn_client_encrypt_vec_f64");
    decrypt = symbol<FhnClientDecryptVectorDoubleFn>("ckks_fhn_client_decrypt_vec_f64");

    // This list requests evaluation keys. It is NOT an executable FHE circuit.
    // The requirement belongs to the selected native kernel family, not main().
    const auto requirements = fhenomenon::contrib::nativeSearchKeyRequirements(width);
    std::vector<FhnInstruction> requests;
    if (requirements.multiplication) {
      FhnInstruction multiply{};
      multiply.opcode = FHN_HMULT;
      requests.push_back(multiply);
    }
    for (auto step : requirements.rotation_steps) {
      FhnInstruction rotate{};
      rotate.opcode = FHN_ROTATE;
      rotate.params[0] = step;
      requests.push_back(rotate);
    }
    FhnProgram evaluation_key_request{
      FHN_ABI_VERSION, static_cast<uint32_t>(requests.size()), requests.data(), 0, nullptr, 0, nullptr};
    auto prepare = symbol<FhnClientPrepareFn>("ckks_fhn_client_prepare");
    require(prepare(client.get(), &evaluation_key_request) == 0, "evaluation key preparation failed");
    input_native = std::make_unique<NativeRuntime>(runtime, services);
    key_setup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  }
};

TrustedClient::TrustedClient(fhenomenon::runtime::Runtime &runtime, const std::string &path)
  : impl_(std::make_unique<Impl>(runtime, path)) {}
TrustedClient::~TrustedClient() = default;
NativeServices TrustedClient::computeServices() const { return impl_->services; }
double TrustedClient::keySetupMs() const { return impl_->key_setup_ms; }

Value TrustedClient::encryptVector(NativeRuntime &native, const std::vector<double> &values) {
  if (values.size() != slots || !std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); }))
    throw std::invalid_argument("packed example vector must contain 4096 finite values");
  require(native.context() == impl_->services.context.get(), "client and ciphertext context differ");
  auto release = [&](FhnBuffer *buffer) { impl_->services.buffer_free(impl_->services.context.get(), buffer); };
  std::unique_ptr<FhnBuffer, decltype(release)> buffer(impl_->services.buffer_alloc(impl_->services.context.get()),
                                                       release);
  require(bool(buffer), "ciphertext allocation failed");
  require(impl_->encrypt(impl_->client.get(), buffer.get(), values.data(), values.size()) == 0,
          "client encryption failed");
  return native.importCopy(buffer.get());
}
std::vector<double> TrustedClient::decryptVector(NativeRuntime &native, const Value &value) {
  require(native.context() == impl_->services.context.get(), "client and ciphertext context differ");
  std::vector<double> result(slots);
  require(impl_->decrypt(impl_->client.get(), native.buffer(value), result.data(), result.size()) == 0,
          "client decryption failed");
  return result;
}

EncryptedDocuments TrustedClient::encryptDocuments(const std::vector<Document> &documents, std::uint32_t capacity) {
  if (documents.empty() || documents.size() > std::numeric_limits<std::uint32_t>::max() || !capacity ||
      capacity > tile_capacity)
    throw std::invalid_argument("invalid document count or packing capacity");
  for (const auto &document : documents)
    validateEmbedding(document.embedding);
  EncryptedDocuments result{{}, SearchShape{width, static_cast<std::uint32_t>(documents.size()), capacity}};
  for (size_t first = 0; first < documents.size(); first += capacity) {
    std::vector<double> packed(slots);
    const auto lanes = capacity == 1 ? tile_capacity : capacity;
    for (unsigned lane = 0; lane < lanes; ++lane) {
      const auto document = first + (capacity == 1 ? 0 : lane);
      if (document < documents.size()) {
        const auto &embedding = documents[document].embedding;
        std::copy(embedding.begin(), embedding.end(), packed.begin() + lane * width);
      }
    }
    result.blocks.push_back(encryptVector(*impl_->input_native, packed));
  }
  return result;
}
EncryptedQuery TrustedClient::encryptQuery(const Query &query) {
  validateEmbedding(query.embedding);
  std::vector<double> packed(slots);
  for (unsigned lane = 0; lane < tile_capacity; ++lane)
    std::copy(query.embedding.begin(), query.embedding.end(), packed.begin() + lane * width);
  return {encryptVector(*impl_->input_native, packed)};
}
std::vector<double> TrustedClient::decryptScores(const EncryptedScores &encrypted) {
  validateShape(encrypted.shape);
  const auto capacity = encrypted.shape.documents_per_ciphertext;
  const auto count = encrypted.shape.document_count;
  if (encrypted.blocks.size() != (std::uint64_t(count) + capacity - 1) / capacity)
    throw std::invalid_argument("score ciphertext count does not match the declared document count");
  for (const auto &block : encrypted.blocks) {
    FhnCkksBufferInfo info{};
    if (impl_->services.buffer_info(impl_->services.context.get(), impl_->input_native->buffer(block), &info) != 0 ||
        info.logical_slots != slots)
      throw std::invalid_argument("score ciphertext has an incompatible slot layout");
  }
  std::vector<double> scores;
  scores.reserve(count);
  for (const auto &block : encrypted.blocks) {
    const auto decoded = decryptVector(*impl_->input_native, block);
    // Only block-start lanes are document scores. Ignore the zero-filled tail
    // and mixed partial sums in all other lanes.
    for (unsigned lane = 0; lane < capacity && scores.size() < count; ++lane) {
      const double score = decoded[lane * width];
      require(std::isfinite(score), "decrypted document score is not finite");
      scores.push_back(score);
    }
  }
  return scores;
}
} // namespace private_search
