#pragma once

#include "CKKS/BConvTables.h"
#include "CKKS/CrtTables.h"
#include "CKKS/Layout.h"
#include "CKKS/NttTables.h"
#include "CKKS/Params.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace fhenomenon::ckks {

// Explicit setup prepares every active-Q prefix and digit conversion. This
// trades setup memory for an immutable, allocation-free evaluation path. NTT
// storage is O((|Q|+|P|)N); CRT/conversion constants are polynomial in |Q|+|P|.
class EvaluationTables {
  public:
  struct DigitTables {
    uint32_t begin = 0;
    uint32_t end = 0;
    // Indices into current Q followed by P, excluding [begin,end).
    std::vector<uint32_t> destinations;
    std::optional<BConvTables> conversion;
  };

  static std::unique_ptr<EvaluationTables> create(const Params &params, std::string *error = nullptr);
  const Params &params() const { return params_; }
  ParamsId paramsId() const { return params_id_; }
  const NttTables &mainNtt(uint32_t index) const { return main_ntt_.at(index); }
  const NttTables &auxNtt(uint32_t index) const { return aux_ntt_.at(index); }
  const CrtTables &mainCrt(uint32_t level) const { return main_crt_.at(level); }
  const CrtTables &auxCrt() const { return *aux_crt_; }
  // Exact centered remainder basis P followed by the Q prime removed at level.
  const CrtTables &switchRescaleCrt(uint32_t level) const { return switch_rescale_crt_.at(level); }
  const DigitTables &digit(uint32_t level, uint32_t index) const { return digits_.at(level).at(index); }
  uint64_t auxProductMod(uint32_t index) const { return aux_product_mod_.at(index); }
  uint64_t auxProductInverse(uint32_t index) const { return aux_product_inverse_.at(index); }
  uint64_t dropInverse(uint32_t level, uint32_t index) const { return drop_inverse_.at(level).at(index); }

  private:
  explicit EvaluationTables(const Params &params) : params_(params), params_id_(ParamsId::of(params)) {}
  Params params_;
  ParamsId params_id_;
  std::vector<NttTables> main_ntt_;
  std::vector<NttTables> aux_ntt_;
  std::vector<CrtTables> main_crt_;
  std::optional<CrtTables> aux_crt_;
  std::vector<CrtTables> switch_rescale_crt_;
  std::vector<std::vector<DigitTables>> digits_;
  std::vector<uint64_t> aux_product_mod_;
  std::vector<uint64_t> aux_product_inverse_;
  std::vector<std::vector<uint64_t>> drop_inverse_;
};

} // namespace fhenomenon::ckks
