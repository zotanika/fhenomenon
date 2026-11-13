#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace openvla::interop {

// Message semantics and polynomial storage are independent. A full polynomial
// NTT does not convert coefficient messages to CKKS SIMD slot messages.
enum class MessageEncoding { Slots, Coefficients, SubringSlotsInCoefficients };
enum class PolynomialDomain { Coefficients, Ntt, PartialNtt };
enum class MessageOrder { Natural, BitReversed };

struct Representation {
  std::string backend_id, context_id, key_id;
  std::uint32_t log_degree, log_slots;
  std::vector<std::uint64_t> active_primes;
  std::int64_t level;
  double scale;
  MessageEncoding message_encoding;
  PolynomialDomain polynomial_domain;
  MessageOrder message_order;
  std::uint32_t partial_log_degree;
  void validate() const {
    if (backend_id.empty() || context_id.empty() || key_id.empty())
      throw std::invalid_argument("ciphertext backend/context/key identity is required");
    if (log_degree < 1 || log_degree > 62 || log_slots >= log_degree)
      throw std::invalid_argument("complex message does not fit the declared ring");
    if (level < 0 || active_primes.empty() || !std::isfinite(scale) || scale <= 0)
      throw std::invalid_argument("invalid ciphertext level/modulus/scale");
    for (std::size_t i = 0; i < active_primes.size(); ++i) {
      if (active_primes[i] < 3 || active_primes[i] % 2 == 0)
        throw std::invalid_argument("invalid active modulus");
      for (std::size_t j = 0; j < i; ++j)
        if (active_primes[i] == active_primes[j])
          throw std::invalid_argument("duplicate active modulus");
    }
    if (message_encoding != MessageEncoding::Slots && message_encoding != MessageEncoding::Coefficients &&
        message_encoding != MessageEncoding::SubringSlotsInCoefficients)
      throw std::invalid_argument("unknown message encoding");
    if (message_order != MessageOrder::Natural && message_order != MessageOrder::BitReversed)
      throw std::invalid_argument("unknown message order");
    if (message_encoding == MessageEncoding::Slots && message_order != MessageOrder::Natural)
      throw std::invalid_argument("slot arithmetic requires explicit natural slot order");
    if (polynomial_domain == PolynomialDomain::PartialNtt) {
      if (message_encoding != MessageEncoding::SubringSlotsInCoefficients || partial_log_degree < 1 ||
          partial_log_degree >= log_degree)
        throw std::invalid_argument("partial NTT requires a matching sub-ring layout");
    } else {
      if (polynomial_domain != PolynomialDomain::Coefficients && polynomial_domain != PolynomialDomain::Ntt)
        throw std::invalid_argument("unknown polynomial domain");
      if (partial_log_degree != 0)
        throw std::invalid_argument("sub-ring degree present without partial NTT");
    }
  }
};

struct DecryptionPolicy {
  std::string backend_id, context_id, key_id;
  std::int64_t minimum_level;
  std::vector<std::uint64_t> minimum_modulus;
};

enum class RefreshKind { BasisExtension, SemanticRefresh };
struct ModRaiseContract {
  std::int64_t target_level, maximum_level;
  double maximum_absolute_error;
};
// The precision bound must come from the evaluator's numerical error model;
// metadata cannot prove precision. Independent encrypted numerical tests also
// validate the profile. A basis lift alone never satisfies this contract.
inline void require_modraise_result(const Representation &source, const Representation &result,
                                    const ModRaiseContract &contract, RefreshKind kind,
                                    double output_absolute_error_bound) {
  source.validate();
  result.validate();
  if (kind != RefreshKind::SemanticRefresh)
    throw std::invalid_argument("ModRaise requires semantic refresh, not a polynomial basis extension");
  if (contract.target_level < 0 || contract.maximum_level < contract.target_level ||
      result.level != contract.target_level)
    throw std::invalid_argument("ModRaise must produce the exact supported target level");
  if (!std::isfinite(contract.maximum_absolute_error) || contract.maximum_absolute_error < 0 ||
      !std::isfinite(output_absolute_error_bound) || output_absolute_error_bound < 0 ||
      output_absolute_error_bound > contract.maximum_absolute_error)
    throw std::invalid_argument("ModRaise exceeds the declared numerical error budget");
  if (source.backend_id != result.backend_id || source.context_id != result.context_id ||
      source.key_id != result.key_id || source.log_degree != result.log_degree ||
      source.log_slots != result.log_slots || source.message_encoding != result.message_encoding ||
      source.message_order != result.message_order || source.polynomial_domain != result.polynomial_domain ||
      source.partial_log_degree != result.partial_log_degree)
    throw std::invalid_argument("ModRaise changed ciphertext identity or logical representation");
}

inline void require_slot_arithmetic(const Representation &value) {
  value.validate();
  if (value.message_encoding != MessageEncoding::Slots)
    throw std::invalid_argument("Hadamard arithmetic requires slot messages, not polynomial coefficients");
}
inline void require_compatible(const Representation &a, const Representation &b) {
  a.validate();
  b.validate();
  if (a.backend_id != b.backend_id || a.context_id != b.context_id || a.key_id != b.key_id)
    throw std::invalid_argument("ciphertexts have different backend/context/key identities");
  if (a.log_degree != b.log_degree || a.log_slots != b.log_slots || a.active_primes != b.active_primes ||
      a.level != b.level || a.message_encoding != b.message_encoding || a.polynomial_domain != b.polynomial_domain ||
      a.message_order != b.message_order || a.partial_log_degree != b.partial_log_degree)
    throw std::invalid_argument("ciphertexts have incompatible representations");
  if (std::abs(a.scale - b.scale) > 1e-12 * std::max(a.scale, b.scale))
    throw std::invalid_argument("ciphertext scales require real arithmetic alignment");
}
inline void require_minimum_level_egress(const Representation &value, const DecryptionPolicy &policy) {
  value.validate();
  if (policy.minimum_level < 0 || policy.minimum_modulus.empty())
    throw std::invalid_argument("minimum decryption level must be explicitly configured");
  if (value.backend_id != policy.backend_id || value.context_id != policy.context_id || value.key_id != policy.key_id ||
      value.level != policy.minimum_level || value.active_primes != policy.minimum_modulus)
    throw std::invalid_argument("decrypt requires the client key and exact configured minimum modulus");
}

} // namespace openvla::interop
