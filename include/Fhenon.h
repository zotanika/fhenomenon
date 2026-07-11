#pragma once

#include "Common.h"
#include "Crypto/ToyFHE.h"
#include "Utils/log.h"

#include <any>
#include <memory>
#include <typeindex>

namespace fhenomenon {

class Profile;
class Session;

class FhenonBase {
  public:
  virtual ~FhenonBase() = default;
  virtual std::type_index type() const = 0;

  // Store encrypted values using backend-specific representation
  std::any ciphertext_;
  bool isEncrypted_ = false;
};

template <typename T> class Fhenon final : public FhenonBase, public std::enable_shared_from_this<Fhenon<T>> {
  private:
  T val_;
  std::shared_ptr<Profile> profile_;
  bool isScalar_ = false;

  public:
  // constructor
  // (TODO) explicit Fhenon(const T v) : val_(v) {}
  Fhenon(const T v) : val_(v) { LOG_MESSAGE("Constructor with value: " << val_); }
  // copy constrcutor
  Fhenon(const Fhenon<T> &other);

  // move constructor
  Fhenon(Fhenon<T> &&other) noexcept;

  // Deregisters from an active recording session; a tracked variable dying
  // before run() executes poisons the recording (see Session::poisonRecording).
  ~Fhenon() override;

  Fhenon<T> &operator=(const T &scalar);
  Fhenon<T> &operator=(const Fhenon &other);
  Fhenon<T> &operator=(Fhenon &&other) noexcept;
  Fhenon<T> operator+(const T &scalar) const;
  Fhenon<T> operator+(const Fhenon<T> &other) const;
  Fhenon<T> operator*(const T &scalar) const;
  Fhenon<T> operator*(const Fhenon<T> &other) const;

  Fhenon<T> operator&(const Fhenon<T> &other) const;
  Fhenon<T> operator|(const Fhenon<T> &other) const;
  Fhenon<T> operator^(const Fhenon<T> &other) const;
  Fhenon<T> operator==(const Fhenon<T> &other) const;
  Fhenon<T> operator<(const Fhenon<T> &other) const;
  Fhenon<T> operator<=(const Fhenon<T> &other) const;

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
