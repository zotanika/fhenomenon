#include "CKKS/Keys.h"
#include "CKKS/Arena.h"
#include "CKKS/EvalTables.h"
#include "CKKS/ModArith.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <new>
#include <sys/random.h>

namespace fhenomenon::ckks {
namespace {
void wipe(void *memory, std::size_t bytes) noexcept {
  volatile unsigned char *p = static_cast<volatile unsigned char *>(memory);
  while (bytes-- != 0)
    *p++ = 0;
}
struct WipeWords {
  uint64_t *data;
  std::size_t words;
  ~WipeWords() {
    if (data != nullptr)
      wipe(data, words * sizeof(uint64_t));
  }
};
class RandomReader {
  public:
  explicit RandomReader(RandomSource &source) : source_(source) {}
  ~RandomReader() { wipe(buffer_.data(), buffer_.size()); }
  bool word(uint64_t &value) {
    if (cursor_ == buffer_.size()) {
      if (!source_.fill(buffer_.data(), buffer_.size()))
        return false;
      cursor_ = 0;
    }
    std::memcpy(&value, buffer_.data() + cursor_, sizeof(value));
    cursor_ += sizeof(value);
    return true;
  }
  bool uniform(uint64_t modulus, uint64_t &value) {
    // Reject the incomplete residue interval at the bottom of [0,2^64).
    const uint64_t threshold = (uint64_t{0} - modulus) % modulus;
    do {
      if (!word(value))
        return false;
    } while (value < threshold);
    value %= modulus;
    return true;
  }
  bool ternary(int64_t &value) {
    uint64_t sample = 0;
    if (!uniform(3, sample))
      return false;
    value = static_cast<int64_t>(sample) - 1;
    return true;
  }
  bool error(int64_t &value) {
    uint64_t sample = 0;
    if (!word(sample))
      return false;
    constexpr uint64_t mask = (uint64_t{1} << 21) - 1;
    value = __builtin_popcountll(sample & mask) - __builtin_popcountll((sample >> 21) & mask);
    return true;
  }

  private:
  RandomSource &source_;
  std::array<uint8_t, 256> buffer_{};
  std::size_t cursor_ = buffer_.size();
};
uint64_t residue(int64_t value, uint64_t modulus) {
  if (value >= 0)
    return static_cast<uint64_t>(value) % modulus;
  const uint64_t absolute = static_cast<uint64_t>(-(value + 1)) + 1;
  const uint64_t reduced = absolute % modulus;
  return reduced == 0 ? 0 : modulus - reduced;
}
uint64_t addMod(uint64_t a, uint64_t b, uint64_t q) {
  const uint64_t sum = a + b;
  return sum >= q ? sum - q : sum;
}
uint64_t subtractMod(uint64_t a, uint64_t b, uint64_t q) { return a >= b ? a - b : q - (b - a); }
bool supported(const Params &params) {
  if (params.secret() != SecretDistribution::UniformTernary || !params.terminalPrimes().empty())
    return false;
  for (uint32_t digit = 0; digit < params.dnum(); ++digit)
    if (params.digitRange(digit).first >= params.digitRange(digit).second)
      return false;
  return true;
}
PolyLayout fullLayout(const EvaluationTables &tables, bool auxiliary) {
  const auto &params = tables.params();
  return {
    {tables.paramsId(), 0, params.levelCount(), 0, auxiliary ? static_cast<uint32_t>(params.auxPrimes().size()) : 0},
    params.logDegree(),
    NttForm::Evaluation};
}
bool validPoly(ConstRnsPolyView p, const EvaluationTables &tables, bool auxiliary = false) {
  const auto &b = p.layout.basis;
  const auto &params = tables.params();
  return p.wellFormed() && p.layout.log_degree == params.logDegree() && b.params_id == tables.paramsId() &&
         p.layout.form == NttForm::Evaluation && b.main_begin == 0 && b.main_end > 0 &&
         b.main_end <= params.levelCount() && b.aux_begin == 0 &&
         b.aux_end == (auxiliary ? params.auxPrimes().size() : 0);
}
bool validSecret(const SecretKey &secret, const EvaluationTables &tables) {
  const auto view = secret.view();
  return validPoly(view, tables, true) && view.layout.basis.main_end == tables.params().levelCount();
}
bool reduced(ConstRnsPolyView p, const Params &params) {
  const std::size_t n = params.degree();
  for (uint32_t limb = 0; limb < p.layout.basis.main_end; ++limb)
    for (std::size_t j = 0; j < n; ++j)
      if (p.limb(limb)[j] >= params.mainPrimes()[limb])
        return false;
  return true;
}
const NttTables &nttFor(uint32_t limb, const EvaluationTables &tables) {
  return limb < tables.params().levelCount() ? tables.mainNtt(limb)
                                             : tables.auxNtt(limb - tables.params().levelCount());
}
uint64_t modulusFor(uint32_t limb, const Params &params) {
  return limb < params.levelCount() ? params.mainPrimes()[limb] : params.auxPrimes()[limb - params.levelCount()];
}
bool sampleSmall(int64_t *out, std::size_t n, RandomReader &random, bool error) {
  for (std::size_t i = 0; i < n; ++i)
    if (!(error ? random.error(out[i]) : random.ternary(out[i])))
      return false;
  return true;
}
void smallNtt(uint64_t *out, const int64_t *small, std::size_t n, const NttTables &table) {
  for (std::size_t i = 0; i < n; ++i)
    out[i] = residue(small[i], table.modulus());
  ntt::forward(out, table);
}
} // namespace

bool SystemRandom::fill(uint8_t *output, std::size_t bytes) {
  if (output == nullptr && bytes != 0)
    return false;
  while (bytes != 0) {
#if defined(__linux__)
    const std::size_t chunk = std::min<std::size_t>(bytes, 256);
    const auto received = ::getrandom(output, chunk, 0);
    if (received < 0) {
      if (errno == EINTR)
        continue;
      return false;
    }
    if (received == 0)
      return false;
    const auto consumed = static_cast<std::size_t>(received);
#else
    const std::size_t consumed = std::min<std::size_t>(bytes, 256);
    if (::getentropy(output, consumed) != 0) {
      if (errno == EINTR)
        continue;
      return false;
    }
#endif
    output += consumed;
    bytes -= consumed;
  }
  return true;
}

SecretKey::SecretKey(std::size_t words, PolyLayout layout) : slab_(words), layout_(layout) {}
SecretKey::~SecretKey() { wipe(slab_.data(), slab_.capacityWords() * sizeof(uint64_t)); }
ConstRnsPolyView SecretKey::view() const { return {slab_.data(), slab_.capacityWords(), layout_}; }
PublicKey::PublicKey(PolyLayout layout) : slab_(2 * layout.wordCount()), layout_{layout, 1.0, 2} {}
ConstCiphertextView PublicKey::view() const {
  return {slab_.data(), slab_.capacityWords(), layout_.poly.wordCount(), layout_};
}
EvaluationKeyView EvaluationKey::view() const {
  auto result = metadata_;
  result.digits = digits_.data();
  result.digit_count = static_cast<uint32_t>(digits_.size());
  return result;
}
std::size_t keyScratchBytes(const Params &params) noexcept {
  // Conservative bound: eight full-QP polynomials plus eight coefficient
  // vectors and alignment. Durable Secret/Public/EvaluationKey owners allocate
  // only in the explicitly named generate* functions, outside computation.
  return params.degree() * sizeof(uint64_t) * (8 * (params.levelCount() + params.auxPrimes().size()) + 8) + 4096;
}

std::unique_ptr<SecretKey> generateSecretKey(const EvaluationTables &tables, RandomSource &source, Arena &arena) {
  const auto &params = tables.params();
  if (!supported(params) || arena.remaining() < keyScratchBytes(params))
    return {};
  Arena::Scope scope(arena);
  const std::size_t n = params.degree();
  auto *memory = arena.allocate<uint64_t>(n);
  if (memory == nullptr)
    return {};
  WipeWords cleanup{memory, n};
  auto *small = reinterpret_cast<int64_t *>(memory);
  RandomReader random(source);
  if (!sampleSmall(small, n, random, false))
    return {};
  const auto layout = fullLayout(tables, true);
  auto result = std::unique_ptr<SecretKey>(new SecretKey(layout.wordCount(), layout));
  for (uint32_t limb = 0; limb < layout.basis.limbCount(); ++limb)
    smallNtt(result->slab_.data() + limb * n, small, n, nttFor(limb, tables));
  return result;
}

std::unique_ptr<PublicKey> generatePublicKey(const SecretKey &secret, const EvaluationTables &tables,
                                             RandomSource &source, Arena &arena) {
  const auto &params = tables.params();
  if (!supported(params) || !validSecret(secret, tables) || arena.remaining() < keyScratchBytes(params))
    return {};
  Arena::Scope scope(arena);
  const std::size_t n = params.degree();
  auto *memory = arena.allocate<uint64_t>(2 * n);
  if (memory == nullptr)
    return {};
  WipeWords cleanup{memory, 2 * n};
  auto *error = reinterpret_cast<int64_t *>(memory);
  uint64_t *error_ntt = memory + n;
  RandomReader random(source);
  if (!sampleSmall(error, n, random, true))
    return {};
  auto result = std::unique_ptr<PublicKey>(new PublicKey(fullLayout(tables, false)));
  const auto layout = result->layout_;
  CiphertextView output{result->slab_.data(), result->slab_.capacityWords(), layout.poly.wordCount(), layout};
  for (uint32_t limb = 0; limb < params.levelCount(); ++limb) {
    const uint64_t q = params.mainPrimes()[limb];
    smallNtt(error_ntt, error, n, tables.mainNtt(limb));
    auto *b = output.poly(0).limb(limb);
    auto *a = output.poly(1).limb(limb);
    const auto *s = secret.view().limb(limb);
    for (std::size_t j = 0; j < n; ++j) {
      if (!random.uniform(q, a[j]))
        return {};
      b[j] = subtractMod(error_ntt[j], modarith::mulMod(a[j], s[j], q), q);
    }
  }
  return result;
}

std::shared_ptr<const EvaluationKey> EvaluationKey::generate(const SecretKey &secret, SwitchKeyKind kind,
                                                             uint64_t galois_element, const EvaluationTables &tables,
                                                             RandomSource &source, Arena &arena) {
  const auto &params = tables.params();
  const std::size_t n = params.degree();
  if (!supported(params) || !validSecret(secret, tables) || arena.remaining() < keyScratchBytes(params) ||
      galois_element == 0 || galois_element >= 2 * n || (galois_element & 1U) == 0)
    return {};
  Arena::Scope scope(arena);
  // Three coefficient vectors: the shared error, its transform, and sigma(s).
  auto *memory = arena.allocate<uint64_t>(4 * n);
  if (memory == nullptr)
    return {};
  WipeWords cleanup{memory, 4 * n};
  auto *error = reinterpret_cast<int64_t *>(memory);
  uint64_t *error_ntt = memory + n;
  uint64_t *old = memory + 2 * n;
  uint64_t *coefficient = memory + 3 * n;
  RandomReader random(source);
  auto result = std::shared_ptr<EvaluationKey>(new EvaluationKey());
  const auto layout = fullLayout(tables, true);
  result->metadata_ = {nullptr,
                       0,
                       tables.paramsId(),
                       params.logDegree(),
                       params.levelCount(),
                       static_cast<uint32_t>(params.auxPrimes().size()),
                       kind,
                       galois_element};
  result->slabs_.reserve(params.dnum());
  result->digits_.reserve(params.dnum());
  for (uint32_t digit = 0; digit < params.dnum(); ++digit) {
    if (!sampleSmall(error, n, random, true))
      return {};
    result->slabs_.emplace_back(2 * layout.wordCount());
    auto &slab = result->slabs_.back();
    CiphertextView output{slab.data(), slab.capacityWords(), layout.wordCount(), {layout, 1.0, 2}};
    const auto range = params.digitRange(digit);
    for (uint32_t limb = 0; limb < layout.basis.limbCount(); ++limb) {
      const uint64_t q = modulusFor(limb, params);
      const auto &ntt_table = nttFor(limb, tables);
      const uint64_t *s = secret.view().limb(limb);
      smallNtt(error_ntt, error, n, ntt_table);
      const bool carries_message = limb >= range.first && limb < range.second;
      if (carries_message) {
        if (kind == SwitchKeyKind::Relinearization) {
          for (std::size_t j = 0; j < n; ++j)
            old[j] = modarith::mulMod(s[j], s[j], q);
        } else {
          std::copy_n(s, n, coefficient);
          ntt::inverse(coefficient, ntt_table);
          for (std::size_t j = 0; j < n; ++j) {
            const uint64_t target = (j * galois_element) % (2 * n);
            old[target % n] = target >= n && coefficient[j] != 0 ? q - coefficient[j] : coefficient[j];
          }
          ntt::forward(old, ntt_table);
        }
      }
      auto *b = output.poly(0).limb(limb);
      auto *a = output.poly(1).limb(limb);
      const uint64_t p = carries_message ? tables.auxProductMod(limb) : 0;
      for (std::size_t j = 0; j < n; ++j) {
        if (!random.uniform(q, a[j]))
          return {};
        b[j] = subtractMod(error_ntt[j], modarith::mulMod(a[j], s[j], q), q);
        if (carries_message)
          b[j] = addMod(b[j], modarith::mulMod(p, old[j], q), q);
      }
    }
    result->digits_.push_back(asConst(output));
  }
  return result;
}
std::shared_ptr<const EvaluationKey> generateRelinearizationKey(const SecretKey &secret, const EvaluationTables &tables,
                                                                RandomSource &source, Arena &arena) {
  return EvaluationKey::generate(secret, SwitchKeyKind::Relinearization, 1, tables, source, arena);
}
std::shared_ptr<const EvaluationKey> generateGaloisKey(const SecretKey &secret, uint64_t galois_element,
                                                       const EvaluationTables &tables, RandomSource &source,
                                                       Arena &arena) {
  return EvaluationKey::generate(secret, SwitchKeyKind::Galois, galois_element, tables, source, arena);
}

bool encrypt(CiphertextView &out, ConstRnsPolyView plaintext, double scale, const PublicKey &public_key,
             RandomSource &source, const EvaluationTables &tables, Arena &arena) {
  const auto &params = tables.params();
  const auto pk = public_key.view();
  CiphertextView destination{out.data, out.capacity_words, out.poly_stride_words, {plaintext.layout, scale, 2}};
  if (!validPoly(plaintext, tables) || !std::isfinite(scale) || scale <= 0 || !destination.wellFormed() ||
      !pk.wellFormed() || !validPoly(pk.poly(0), tables) || pk.layout.poly.basis.main_end != params.levelCount() ||
      !reduced(plaintext, params) || arena.remaining() < keyScratchBytes(params))
    return false;
  Arena::Scope scope(arena);
  const std::size_t n = params.degree();
  const std::size_t words = plaintext.layout.wordCount();
  auto *memory = arena.allocate<uint64_t>(2 * words + 6 * n);
  if (memory == nullptr)
    return false;
  WipeWords cleanup{memory, 2 * words + 6 * n};
  auto *u = reinterpret_cast<int64_t *>(memory + 2 * words);
  auto *e0 = u + n;
  auto *e1 = e0 + n;
  uint64_t *u_ntt = memory + 2 * words + 3 * n;
  uint64_t *e0_ntt = u_ntt + n;
  uint64_t *e1_ntt = e0_ntt + n;
  RandomReader random(source);
  if (!sampleSmall(u, n, random, false) || !sampleSmall(e0, n, random, true) || !sampleSmall(e1, n, random, true))
    return false;
  for (uint32_t limb = 0; limb < plaintext.layout.basis.main_end; ++limb) {
    const auto &table = tables.mainNtt(limb);
    const uint64_t q = table.modulus();
    smallNtt(u_ntt, u, n, table);
    smallNtt(e0_ntt, e0, n, table);
    smallNtt(e1_ntt, e1, n, table);
    const auto *b = pk.poly(0).limb(limb);
    const auto *a = pk.poly(1).limb(limb);
    const auto *m = plaintext.limb(limb);
    for (std::size_t j = 0; j < n; ++j) {
      memory[limb * n + j] = addMod(addMod(modarith::mulMod(b[j], u_ntt[j], q), e0_ntt[j], q), m[j], q);
      memory[words + limb * n + j] = addMod(modarith::mulMod(a[j], u_ntt[j], q), e1_ntt[j], q);
    }
  }
  std::copy_n(memory, words, destination.poly(0).data);
  std::copy_n(memory + words, words, destination.poly(1).data);
  out.layout = destination.layout;
  return true;
}

bool decrypt(RnsPolyView &out, ConstCiphertextView ciphertext, const SecretKey &secret, const EvaluationTables &tables,
             Arena &arena) {
  const auto &params = tables.params();
  RnsPolyView destination{out.data, out.capacity_words, ciphertext.layout.poly};
  if (!ciphertext.wellFormed() || !validPoly(ciphertext.poly(0), tables) || !destination.wellFormed() ||
      !validSecret(secret, tables) || arena.remaining() < keyScratchBytes(params))
    return false;
  for (uint32_t i = 0; i < ciphertext.layout.num_polys; ++i)
    if (!reduced(ciphertext.poly(i), params))
      return false;
  Arena::Scope scope(arena);
  const std::size_t n = params.degree();
  const std::size_t words = ciphertext.layout.poly.wordCount();
  auto *temporary = arena.allocate<uint64_t>(words);
  if (temporary == nullptr)
    return false;
  WipeWords cleanup{temporary, words};
  for (uint32_t limb = 0; limb < ciphertext.layout.poly.basis.main_end; ++limb) {
    const uint64_t q = params.mainPrimes()[limb];
    const auto *s = secret.view().limb(limb);
    for (std::size_t j = 0; j < n; ++j) {
      uint64_t value = ciphertext.poly(ciphertext.layout.num_polys - 1).limb(limb)[j];
      for (uint32_t p = ciphertext.layout.num_polys - 1; p != 0; --p)
        value = addMod(modarith::mulMod(value, s[j], q), ciphertext.poly(p - 1).limb(limb)[j], q);
      temporary[limb * n + j] = value;
    }
  }
  std::copy_n(temporary, words, destination.data);
  out.layout = destination.layout;
  return true;
}
} // namespace fhenomenon::ckks
