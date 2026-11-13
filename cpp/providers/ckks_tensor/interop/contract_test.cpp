#include "contract.hpp"
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace openvla::interop;

static void check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
static void rejects(const std::function<void()> &operation, const char *message) {
  bool rejected = false;
  try {
    operation();
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected, message);
}
static Representation slots() {
  return {"native",
          "context-1",
          "key-1",
          13,
          3,
          {65537, 114689, 147457},
          2,
          1048576.,
          MessageEncoding::Slots,
          PolynomialDomain::Ntt,
          MessageOrder::Natural,
          0};
}

int main() {
  try {
    auto a = slots();
    require_slot_arithmetic(a);
    require_compatible(a, a);
    auto b = a;
    b.message_encoding = MessageEncoding::Coefficients;
    b.message_order = MessageOrder::BitReversed;
    // Catches treating polynomial NTT storage as logical slot encoding.
    rejects([&] { require_slot_arithmetic(b); }, "coefficient NTT accepted as SIMD slots");
    rejects([&] { require_compatible(a, b); }, "different message representations accepted");
    b = a;
    b.context_id = "context-2";
    rejects([&] { require_compatible(a, b); }, "cross-context ciphertext accepted");
    b = a;
    b.key_id = "key-2";
    rejects([&] { require_compatible(a, b); }, "different secret key accepted");
    b = a;
    b.active_primes.back() = 163841;
    rejects([&] { require_compatible(a, b); }, "equal-length different modulus accepted");
    b = a;
    b.scale *= 2;
    rejects([&] { require_compatible(a, b); }, "different scale accepted");
    b = a;
    b.polynomial_domain = PolynomialDomain::PartialNtt;
    b.partial_log_degree = 4;
    rejects([&] { b.validate(); }, "partial NTT accepted without sub-ring layout");
    b.message_encoding = MessageEncoding::SubringSlotsInCoefficients;
    b.validate();
    rejects([&] { require_slot_arithmetic(b); }, "sub-ring partial slot layout accepted as ordinary slots");
    b = a;
    b.scale = std::numeric_limits<double>::infinity();
    rejects([&] { b.validate(); }, "nonfinite scale accepted");
    b = a;
    b.log_slots = b.log_degree;
    rejects([&] { b.validate(); }, "oversized complex message accepted");
    DecryptionPolicy policy{"native", "context-1", "key-1", 0, {65537}};
    rejects([&] { require_minimum_level_egress(a, policy); }, "high-level decryption accepted");
    b = a;
    b.level = 0;
    b.active_primes = {65537};
    require_minimum_level_egress(b, policy);
    b.active_primes = {114689};
    rejects([&] { require_minimum_level_egress(b, policy); }, "wrong bottom modulus accepted");
    check(a.level == 2 && a.active_primes.size() == 3, "egress validation mutated source");
    ModRaiseContract raise{4, 5, 1e-4};
    b = a;
    b.level = 4;
    b.active_primes = {65537, 114689, 147457, 163841, 188417};
    require_modraise_result(a, b, raise, RefreshKind::SemanticRefresh, 5e-5);
    rejects([&] { require_modraise_result(a, b, raise, RefreshKind::BasisExtension, 5e-5); },
            "plain modulus basis extension accepted as semantic refresh");
    rejects([&] { require_modraise_result(a, b, raise, RefreshKind::SemanticRefresh, 2e-4); },
            "refresh exceeded the requested numerical error bound");
    b.level = 3;
    rejects([&] { require_modraise_result(a, b, raise, RefreshKind::SemanticRefresh, 5e-5); },
            "refresh returned a different target level");
    b.level = 4;
    raise.target_level = 6;
    rejects([&] { require_modraise_result(a, b, raise, RefreshKind::SemanticRefresh, 5e-5); },
            "target above backend maximum accepted");
    raise.target_level = 4;
    raise.maximum_absolute_error = -1;
    rejects([&] { require_modraise_result(a, b, raise, RefreshKind::SemanticRefresh, 5e-5); },
            "invalid numerical error budget accepted");
    std::cout << "representation contracts passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
