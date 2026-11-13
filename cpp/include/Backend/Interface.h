#pragma once

#include "Fhenon.h"

#include <any>

namespace fhenomenon {

class BackendInterface {
  public:
  virtual ~BackendInterface() = default;

  virtual void transform(FhenonBase &entity, const Parameter &params) = 0;
  virtual std::shared_ptr<FhenonBase> add(const FhenonBase &a, const FhenonBase &b) = 0;
  virtual std::shared_ptr<FhenonBase> multiply(const FhenonBase &a, const FhenonBase &b) = 0;
  virtual std::any decrypt(const FhenonBase &entity) const = 0;
};

} // namespace fhenomenon
