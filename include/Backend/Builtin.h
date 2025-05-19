#pragma once

#include "Backend/Backend.h"
#include "Compuon.h"

namespace fhenomenon {

extern template class Compuon<int>;

class BuiltinBackendMockup final : public Backend {
  public:
  BackendType getBackendType() const override { return BackendType::BuiltinBackend; }
  void transform(CompuonBase &entity, [[maybe_unused]] const Parameter &params) override;
  std::shared_ptr<CompuonBase> add(const CompuonBase &a, const CompuonBase &b) override;
  std::shared_ptr<CompuonBase> multiply(const CompuonBase &a, const CompuonBase &b) override;
  std::any decrypt(const CompuonBase &entity) const override;
};

} // namespace fhenomenon
