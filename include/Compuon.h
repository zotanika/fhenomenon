#pragma once

#include "Common.h"
#include "Crypto/ToyFHE.h"
#include "Utils/log.h"

#include <memory>
#include <typeindex>

namespace fhenomenon {

class Profile;
class Session;

class CompuonBase {
  public:
  virtual ~CompuonBase() = default;
  virtual std::type_index type() const = 0;

  // Store encrypted values using the toy FHE ciphertext representation
  std::shared_ptr<toyfhe::Ciphertext> ciphertext_;
  bool isEncrypted_ = false;
};

template <typename T> class Compuon final : public CompuonBase, public std::enable_shared_from_this<Compuon<T>> {
  private:
  T val_;
  std::shared_ptr<Profile> profile_;
  bool isScalar_ = false;

  public:
  // constructor
  // (TODO) explicit Compuon(const T v) : val_(v) {}
  Compuon(const T v) : val_(v) { LOG_MESSAGE("Constructor with value: " << val_); }
  // copy constrcutor
  Compuon(const Compuon<T> &other);

  // move constructor
  Compuon(Compuon<T> &&other) noexcept
    : std::enable_shared_from_this<Compuon<T>>(std::move(other)), val_(std::move(other.val_)),
      profile_(std::move(other.profile_)) {
    LOG_MESSAGE("Move constructor with value: " << val_ << " " << this << "other:" << &other);
    // reset 'other'
    other.val_ = T();
    other.profile_ = nullptr;
  }

  Compuon<T> &operator=(const T &scalar);
  Compuon<T> &operator=(const Compuon &other);
  Compuon<T> &operator=(Compuon &&other) noexcept;
  Compuon<T> operator+(const T &scalar) const;
  Compuon<T> operator+(const Compuon<T> &other) const;
  Compuon<T> operator*(const T &scalar) const;
  Compuon<T> operator*(const Compuon<T> &other) const;

  std::type_index type() const override { return typeid(T); }

  void belong(std::shared_ptr<Profile> newProfile);
  std::shared_ptr<Profile> getProfile() const { return profile_; }
  void setProfile(std::shared_ptr<Profile> newProfile) { profile_ = newProfile; }

  bool isScalar() const { return isScalar_; }
  void setScalar() { isScalar_ = true; }

  T decrypt() const;

  T getValue() const { return val_; }

  void setValue(T value) { val_ = value; }
};

} // namespace fhenomenon
