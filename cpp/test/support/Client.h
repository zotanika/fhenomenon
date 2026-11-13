#pragma once
#include "FHN/NativeRuntime.h"
#include <memory>
#include <vector>

namespace test_support {
// Trusted in-process validation role. Never pass this object to an evaluator.
// The Runtime must outlive this object, and this client owns evaluation keys.
class Client {
  public:
  Client(fhenomenon::runtime::Runtime &, const std::string &library, std::uint32_t slots);
  ~Client();
  fhenomenon::runtime::NativeServices services() const;
  fhenomenon::runtime::Value encrypt(const std::vector<double> &);
  std::vector<double> decrypt(const fhenomenon::runtime::Value &) const;

  private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace test_support
