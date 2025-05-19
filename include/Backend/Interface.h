#pragma once

#include "Compuon.h"

#include <any>

namespace fhenomenon {

class BackendInterface {
  public:
  virtual ~BackendInterface() = default;

  virtual void transform(CompuonBase &entity, const Parameter &params) = 0;
  virtual std::shared_ptr<CompuonBase> add(const CompuonBase &a, const CompuonBase &b) = 0;
  virtual std::shared_ptr<CompuonBase> multiply(const CompuonBase &a, const CompuonBase &b) = 0;
  virtual std::any decrypt(const CompuonBase &entity) const = 0;
};

} // namespace fhenomenon
