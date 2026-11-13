#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fhenomenon::tensor_ckks {

struct Cipher;
using Value = std::shared_ptr<Cipher>;

struct Config {
  unsigned log_degree = 17;
  unsigned levels = 42;
  unsigned evaluation_top = 8;
  unsigned log_scale = 48;
  unsigned bottom_bits = 62;
  unsigned bootstrap_prime_bits = 62;
  unsigned digit_width = 3;
  unsigned logical_slots = 256;
  bool test_only = false;
};

// Owns public evaluation material, immutable tables, explicit workspace and
// ciphertext allocations. Secret keys exist only in Client, never here.
class Engine {
  public:
  explicit Engine(Config config);
  ~Engine();
  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;
  const Config &config() const;
  unsigned physicalSlots() const;
  unsigned period(const Value &) const;
  unsigned count(const Value &) const;
  unsigned level(const Value &) const;
  double scale(const Value &) const;
  double valueScale(unsigned level) const;
  // Public RNS prime used by the scope compiler for scale transfer rules.
  std::uint64_t evaluationModulus(unsigned level) const;
  std::uint64_t bottomModulus() const;
  std::size_t bytes(const Value &) const;
  std::string representation(const Value &) const;
  Value withRepresentation(const Value &, const std::string &);
  Value broadcastTo(const Value &, unsigned period, unsigned count);
  Value add(const Value &, const Value &);
  Value sub(const Value &, const Value &);
  Value mul(const Value &, const Value &);
  Value neg(const Value &);
  Value rotate(const Value &, std::int64_t step);
  Value conjugate(const Value &);
  Value plain(const Value &, const std::vector<std::complex<double>> &, bool multiply);
  Value levelDown(const Value &, unsigned target);
  // Arithmetic basis extension only. Not a semantic refresh by itself.
  Value liftBottom(const Value &, unsigned top);
  // Explicit interpretation change used with compensating public transform
  // factors, not an arithmetic precision-restoration operation.
  Value cloneWithScale(const Value &, double new_scale);
  Value multiplyInteger(const Value &, std::int64_t);
  Value multiplyI(const Value &);
  void enableCuda();
  bool isCuda() const;
  void synchronize();
  std::size_t workspaceBytes() const;
  std::size_t keyBytes() const;
  std::size_t plaintextCacheBytes() const;
  std::uint64_t plaintextCacheHits() const;
  std::uint64_t arithmeticCalls() const;

  private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
  friend class Client;
};

class Client {
  public:
  explicit Client(Engine &, const std::vector<std::int32_t> &rotations = {});
  ~Client();
  Client(const Client &) = delete;
  Client &operator=(const Client &) = delete;
  Value encrypt(const std::vector<std::complex<double>> &);
  // Always decrypts an owned copy at one Q prime. No evaluator decryption.
  std::vector<std::complex<double>> decrypt(const Value &, double public_bound = 1.0);
  std::uint64_t encryptCalls() const;
  std::uint64_t decryptCalls() const;
  unsigned lastEgressModuli() const;

  private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace fhenomenon::tensor_ckks
