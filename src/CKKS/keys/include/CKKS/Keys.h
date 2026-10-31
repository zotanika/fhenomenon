#pragma once

#include "CKKS/Storage.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace fhenomenon::ckks {
class Arena;
class Params;
class EvaluationTables;

class RandomSource {
  public:
  virtual ~RandomSource() = default;
  virtual bool fill(uint8_t *output, std::size_t bytes) = 0;
};

// OS entropy only; an entropy error is propagated, never replaced by a PRNG.
class SystemRandom final : public RandomSource {
  public:
  bool fill(uint8_t *output, std::size_t bytes) override;
};

class SecretKey {
  public:
  ~SecretKey();
  SecretKey(const SecretKey &) = delete;
  SecretKey &operator=(const SecretKey &) = delete;
  // Host-only access; L4 receives EvaluationKeyView and cannot name this type.
  ConstRnsPolyView view() const;

  private:
  SecretKey(std::size_t words, PolyLayout layout);
  PolySlab slab_;
  PolyLayout layout_;
  friend std::unique_ptr<SecretKey> generateSecretKey(const EvaluationTables &, RandomSource &, Arena &);
};

class PublicKey {
  public:
  PublicKey(const PublicKey &) = delete;
  PublicKey &operator=(const PublicKey &) = delete;
  ConstCiphertextView view() const;

  private:
  explicit PublicKey(PolyLayout layout);
  PolySlab slab_;
  CiphertextLayout layout_;
  friend std::unique_ptr<PublicKey> generatePublicKey(const SecretKey &, const EvaluationTables &, RandomSource &,
                                                      Arena &);
};

class EvaluationKey {
  public:
  EvaluationKey(const EvaluationKey &) = delete;
  EvaluationKey &operator=(const EvaluationKey &) = delete;
  EvaluationKeyView view() const;

  private:
  EvaluationKey() = default;
  static std::shared_ptr<const EvaluationKey> generate(const SecretKey &, SwitchKeyKind, uint64_t,
                                                       const EvaluationTables &, RandomSource &, Arena &);
  std::vector<PolySlab> slabs_;
  std::vector<ConstCiphertextView> digits_;
  EvaluationKeyView metadata_{};
  friend std::shared_ptr<const EvaluationKey> generateRelinearizationKey(const SecretKey &, const EvaluationTables &,
                                                                         RandomSource &, Arena &);
  friend std::shared_ptr<const EvaluationKey> generateGaloisKey(const SecretKey &, uint64_t, const EvaluationTables &,
                                                                RandomSource &, Arena &);
};

std::unique_ptr<SecretKey> generateSecretKey(const EvaluationTables &, RandomSource &, Arena &);
std::unique_ptr<PublicKey> generatePublicKey(const SecretKey &, const EvaluationTables &, RandomSource &, Arena &);
std::shared_ptr<const EvaluationKey> generateRelinearizationKey(const SecretKey &, const EvaluationTables &,
                                                                RandomSource &, Arena &);
std::shared_ptr<const EvaluationKey> generateGaloisKey(const SecretKey &, uint64_t galois_element,
                                                       const EvaluationTables &, RandomSource &, Arena &);

// Encryption requires public material only. Output is committed after all
// validation, randomness and scratch requests have succeeded.
bool encrypt(CiphertextView &, ConstRnsPolyView plaintext, double scale, const PublicKey &, RandomSource &,
             const EvaluationTables &, Arena &);
bool decrypt(RnsPolyView &, ConstCiphertextView, const SecretKey &, const EvaluationTables &, Arena &);

// Conservative temporary-workspace bound for any key generation or host
// encrypt/decrypt call, excluding durable key storage and immutable tables.
std::size_t keyScratchBytes(const Params &) noexcept;
} // namespace fhenomenon::ckks
