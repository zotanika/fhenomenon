#include "CKKS/EvalTables.h"

#include "CKKS/ModArith.h"

#include <algorithm>
#include <utility>

namespace fhenomenon::ckks {

std::unique_ptr<EvaluationTables> EvaluationTables::create(const Params &params, std::string *error) {
  const auto aux_count = static_cast<uint32_t>(params.auxPrimes().size());
  if (uint64_t{params.levelCount()} + aux_count > RnsBasis::kMaxLimbs) {
    if (error != nullptr)
      *error = "Q and P exceed the storage limb limit";
    return nullptr;
  }
  for (uint32_t d = 0; d < params.dnum(); ++d) {
    const auto range = params.digitRange(d);
    if (range.first == range.second) {
      if (error != nullptr)
        *error = "empty full-level key-switching digit";
      return nullptr;
    }
  }
  std::unique_ptr<EvaluationTables> result(new EvaluationTables(params));
  for (auto q : params.mainPrimes()) {
    auto ntt = NttTables::create(q, params.logDegree(), error);
    if (!ntt)
      return nullptr;
    result->main_ntt_.push_back(std::move(*ntt));
    uint64_t product = 1;
    for (auto p : params.auxPrimes())
      product = modarith::mulMod(product, p % q, q);
    result->aux_product_mod_.push_back(product);
    result->aux_product_inverse_.push_back(modarith::invMod(product, q));
  }
  for (auto p : params.auxPrimes()) {
    auto ntt = NttTables::create(p, params.logDegree(), error);
    if (!ntt)
      return nullptr;
    result->aux_ntt_.push_back(std::move(*ntt));
  }
  result->aux_crt_ = CrtTables::create(params.auxPrimes(), error);
  if (!result->aux_crt_)
    return nullptr;
  result->digits_.resize(params.levelCount());
  result->drop_inverse_.resize(params.levelCount());
  for (uint32_t level = 0; level < params.levelCount(); ++level) {
    const uint32_t count = level + 1;
    const auto &main = params.mainPrimes();
    std::vector<uint64_t> prefix(main.begin(), main.begin() + count);
    auto crt = CrtTables::create(prefix, error);
    if (!crt)
      return nullptr;
    result->main_crt_.push_back(std::move(*crt));
    auto removed_basis = params.auxPrimes();
    removed_basis.push_back(main[level]);
    auto removed_crt = CrtTables::create(std::move(removed_basis), error);
    if (!removed_crt)
      return nullptr;
    result->switch_rescale_crt_.push_back(std::move(*removed_crt));
    for (uint32_t i = 0; i < level; ++i)
      result->drop_inverse_[level].push_back(modarith::invMod(main[level] % main[i], main[i]));
    auto &digits = result->digits_[level];
    digits.resize(params.dnum());
    for (uint32_t d = 0; d < params.dnum(); ++d) {
      const auto range = params.digitRange(d);
      auto &digit = digits[d];
      digit.begin = std::min(range.first, count);
      digit.end = std::min(range.second, count);
      if (digit.begin == digit.end)
        continue;
      std::vector<uint64_t> from(main.begin() + digit.begin, main.begin() + digit.end);
      std::vector<uint64_t> to;
      for (uint32_t i = 0; i < count + aux_count; ++i) {
        if (i >= digit.begin && i < digit.end)
          continue;
        digit.destinations.push_back(i);
        to.push_back(i < count ? main[i] : params.auxPrimes()[i - count]);
      }
      digit.conversion = BConvTables::create(from, to, error);
      if (!digit.conversion)
        return nullptr;
    }
  }
  if (error != nullptr)
    error->clear();
  return result;
}

} // namespace fhenomenon::ckks
