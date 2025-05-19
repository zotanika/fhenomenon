#include "Backend/Builtin.h"

#include <optional>

using namespace fhenomenon;

namespace {

// int -> ASCII -> encrypted int
int encryptASCII(int value) {
  int encrypted{}, shift{};

  while (value > 0) {
    // get the lowest 7 bits for ASCII range
    int asciiChar = (value & 0x7F) + 1;

    encrypted |= (asciiChar << shift);
    shift += 8;  // next byte
    value >>= 7; // shift right by 7 bits
  }

  return encrypted;
}

int decryptASCII(int encrypted) {
  int decrypted{}, shift{};

  while (encrypted > 0) {
    // extract and convert back to the original value
    int asciiChar = (encrypted & 0xFF) - 1;

    decrypted |= (asciiChar << shift);
    shift += 7;      // move to the next 7-bit block
    encrypted >>= 8; // shift right by 8 bits
  }

  return decrypted;
}

} // namespace

void BuiltinBackendMockup::transform(CompuonBase &entity, [[maybe_unused]] const Parameter &params) {

  auto type = entity.type();

  if (type == typeid(int)) {
    Compuon<int> &derivedEntity = dynamic_cast<Compuon<int> &>(entity);

    derivedEntity.setValue(encryptASCII(derivedEntity.getValue()));
  }
  // (TODO) add other types
}

std::shared_ptr<CompuonBase> BuiltinBackendMockup::add(const CompuonBase &a, const CompuonBase &b) {
  auto type = a.type();

  if (type == typeid(int)) {
    const Compuon<int> &derivedA = dynamic_cast<const Compuon<int> &>(a);
    const Compuon<int> &derivedB = dynamic_cast<const Compuon<int> &>(b);

    int addition = decryptASCII(derivedA.getValue()) + decryptASCII(derivedB.getValue());

    return std::make_shared<Compuon<int>>(encryptASCII(addition));
  }
  // (TODO) add other types

  return nullptr;
}

std::shared_ptr<CompuonBase> BuiltinBackendMockup::multiply(const CompuonBase &a, const CompuonBase &b) {
  // auto type = a.type();
  std::optional<const std::type_index> type =
    (a.type() == b.type() ? std::optional<const std::type_index>(a.type()) : std::nullopt);

  if (type.value() == typeid(int)) {
    const Compuon<int> &derivedA = dynamic_cast<const Compuon<int> &>(a);
    const Compuon<int> &derivedB = dynamic_cast<const Compuon<int> &>(b);

    int multiplication = decryptASCII(derivedA.getValue()) * decryptASCII(derivedB.getValue());
    return std::make_shared<Compuon<int>>(encryptASCII(multiplication));
  }
  // (TODO) add other types

  return nullptr;
}

std::any BuiltinBackendMockup::decrypt(const CompuonBase &entity) const {
  auto type = entity.type();

  if (type == typeid(int)) {
    const Compuon<int> &derivedEntity = dynamic_cast<const Compuon<int> &>(entity);

    return decryptASCII(derivedEntity.getValue());
  }
  // (TODO) add other types

  return nullptr;
}
