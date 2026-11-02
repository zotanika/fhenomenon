#include "FHN/contrib/SearchKernels.h"
#include "FHN/contrib/MatvecKernels.h"
#include "ScoreBlock.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace fhenomenon::contrib {
namespace {
using namespace runtime;
constexpr const char *indexFormat = "contrib.search.index";
constexpr const char *preparationFormat = "contrib.search.query-preparation";
struct Provider {
  NativeServices services;
  std::shared_ptr<NativeRuntime> native;
};
// Ciphertext Values live only in the public dependency list. The mapping lets
// multiple corpus blocks share a ciphertext without duplicating its ownership.
struct Index {
  std::uint64_t identity = 0;
  std::shared_ptr<const Provider> provider;
  SearchShape shape;
  FhnCkksBufferInfo info;
  std::vector<std::size_t> blocks;
};
struct Preparation {
  std::uint64_t identity = 0;
  std::shared_ptr<const Provider> provider;
  std::uint64_t query_identity;
  std::uint32_t width;
};
bool powerOfTwo(std::uint32_t n) { return n && !(n & (n - 1)); }
bool metadata(const Provider &provider, const Value &value, FhnCkksBufferInfo &info) {
  try {
    return provider.services.buffer_info(provider.services.context.get(), provider.native->buffer(value), &info) == 0 &&
           info.components == 2 && info.level >= 1 && powerOfTwo(info.logical_slots) && std::isfinite(info.scale) &&
           info.scale > 0;
  } catch (const Error &) {
    return false;
  }
}
bool compatible(const FhnCkksBufferInfo &a, const FhnCkksBufferInfo &b) {
  return a.logical_slots == b.logical_slots && a.level == b.level &&
         std::abs(a.scale - b.scale) <= 1e-12 * std::max(a.scale, b.scale);
}
bool shapeValid(const SearchShape &shape, std::uint32_t logicalSlots) {
  return powerOfTwo(shape.width) && shape.document_count && shape.documents_per_ciphertext &&
         std::uint64_t(shape.width) * shape.documents_per_ciphertext <= logicalSlots;
}
std::size_t blockCount(const SearchShape &shape) {
  return (std::uint64_t(shape.document_count) + shape.documents_per_ciphertext - 1) / shape.documents_per_ciphertext;
}
std::shared_ptr<const Index> index(const Provider &provider, const Value &value) {
  if (!value || value.kind() != ResourceKind::Artifact || value.format() != indexFormat)
    return {};
  const auto *owned = std::any_cast<std::shared_ptr<const Index>>(&value.payload());
  if (!owned || !*owned || (*owned)->provider.get() != &provider || (*owned)->identity != value.identity())
    return {};
  return *owned;
}
bool preparationValid(const Provider &provider, const Value &value, const Value &query, std::uint32_t width) {
  if (!value || value.kind() != ResourceKind::Artifact || value.format() != preparationFormat)
    return false;
  const auto *owned = std::any_cast<std::shared_ptr<const Preparation>>(&value.payload());
  return owned && *owned && (*owned)->identity == value.identity() && (*owned)->provider.get() == &provider &&
         (*owned)->query_identity == query.identity() && (*owned)->width == width && value.dependencies().size() == 1;
}
bool indexApplicable(const Provider &provider, const Request &request, std::string &reason) {
  const auto *shape = std::any_cast<SearchShape>(&request.attributes);
  FhnCkksBufferInfo first{};
  if (!shape || request.inputs.empty() || !metadata(provider, request.inputs[0], first) ||
      !shapeValid(*shape, first.logical_slots) || request.inputs.size() != blockCount(*shape)) {
    reason = "invalid search index shape, ciphertext count or native metadata";
    return false;
  }
  for (const auto &block : request.inputs) {
    FhnCkksBufferInfo info{};
    if (!metadata(provider, block, info) || !compatible(first, info)) {
      reason = "search index blocks require one context, level, scale and logical slot count";
      return false;
    }
  }
  return true;
}
bool prepareApplicable(const Provider &provider, const Request &request, std::string &reason) {
  const auto *width = std::any_cast<std::uint32_t>(&request.attributes);
  FhnCkksBufferInfo info{};
  if (!width || !powerOfTwo(*width) || request.inputs.size() != 1 || !metadata(provider, request.inputs[0], info) ||
      *width > info.logical_slots) {
    reason = "invalid search query preparation source or width";
    return false;
  }
  return true;
}
bool scoreApplicable(const Provider &provider, const Request &request, std::string &reason) {
  if ((request.inputs.size() != 2 && request.inputs.size() != 3) || request.attributes.has_value()) {
    reason = "collection scoring requires query, index and optional search preparation";
    return false;
  }
  const auto owned = index(provider, request.inputs[1]);
  FhnCkksBufferInfo query{};
  if (!owned || !metadata(provider, request.inputs[0], query) || !compatible(owned->info, query) ||
      (request.inputs.size() == 3 &&
       !preparationValid(provider, request.inputs[2], request.inputs[0], owned->shape.width))) {
    reason = "search index, query or typed preparation does not match this provider";
    return false;
  }
  return true;
}
bool blockApplicable(const Provider &provider, const Request &request, bool prepared, bool row, std::string &reason) {
  const auto *shape = std::any_cast<SearchShape>(&request.attributes);
  FhnCkksBufferInfo query{}, block{};
  if (request.inputs.size() != (prepared ? 3u : 2u) || !shape || !metadata(provider, request.inputs[0], query) ||
      !metadata(provider, request.inputs[1], block) || !shapeValid(*shape, query.logical_slots) ||
      shape->document_count > shape->documents_per_ciphertext || !compatible(query, block) ||
      (shape->document_count == 1) != row ||
      (prepared && !preparationValid(provider, request.inputs[2], request.inputs[0], shape->width))) {
    reason = "block layout, native operands or typed query preparation do not match recipe";
    return false;
  }
  return true;
}
Value executeIndex(Runtime &runtime, const std::shared_ptr<const Provider> &provider, const Request &request) {
  const auto shape = std::any_cast<SearchShape>(request.attributes);
  // The metadata and index mapping are charged as retained host payload, while
  // shared ciphertext dependencies retain their original single charges.
  if (request.inputs.size() > (std::numeric_limits<std::size_t>::max() - sizeof(Index)) / sizeof(std::size_t))
    throw Error(ErrorCode::InvalidRequest, "Search index metadata size overflows");
  Values dependencies;
  dependencies.reserve(request.inputs.size());
  std::vector<std::size_t> mapping;
  mapping.reserve(request.inputs.size());
  for (const auto &block : request.inputs) {
    auto found = std::find_if(dependencies.begin(), dependencies.end(),
                              [&](const Value &v) { return v.identity() == block.identity(); });
    mapping.push_back(static_cast<std::size_t>(found - dependencies.begin()));
    if (found == dependencies.end())
      dependencies.push_back(block);
  }
  if (mapping.capacity() > (std::numeric_limits<std::size_t>::max() - sizeof(Index)) / sizeof(std::size_t))
    throw Error(ErrorCode::InvalidRequest, "Search index metadata capacity overflows");
  const auto bytes = sizeof(Index) + mapping.capacity() * sizeof(std::size_t);
  FhnCkksBufferInfo info{};
  if (!metadata(*provider, request.inputs[0], info))
    throw Error(ErrorCode::InvalidRequest, "Search index metadata changed");
  std::shared_ptr<Index> owned;
  auto result = runtime.makeValue(
    indexFormat, ResourceKind::Artifact, bytes,
    [&]() -> std::any {
      owned = std::make_shared<Index>(Index{0, provider, shape, info, std::move(mapping)});
      return std::shared_ptr<const Index>(owned);
    },
    std::move(dependencies));
  // Seal before publication: copying a public std::any payload cannot transplant
  // trusted metadata onto an artifact with forged ciphertext dependencies.
  owned->identity = result.identity();
  return result;
}
Value executePrepare(Runtime &runtime, const std::shared_ptr<const Provider> &provider, const Request &request) {
  const auto width = std::any_cast<std::uint32_t>(request.attributes);
  auto underlying = runtime.submit({"contrib.matvec.prepare", request.inputs, MatvecPrepareRequest{width}}).at(0);
  std::shared_ptr<Preparation> owned;
  auto result = runtime.makeValue(preparationFormat, ResourceKind::Artifact, sizeof(Preparation),
                                  [&]() -> std::any {
                                    owned = std::make_shared<Preparation>(
                                      Preparation{0, provider, request.inputs[0].identity(), width});
                                    return std::shared_ptr<const Preparation>(owned);
                                  },
                                  {std::move(underlying)});
  owned->identity = result.identity();
  return result;
}
Values executeScore(Runtime &runtime, const Request &request) {
  const auto owned = *std::any_cast<std::shared_ptr<const Index>>(&request.inputs[1].payload());
  Values results;
  results.reserve(owned->blocks.size());
  std::uint32_t remaining = owned->shape.document_count;
  for (auto dependency : owned->blocks) {
    auto shape = owned->shape;
    shape.document_count = std::min(remaining, shape.documents_per_ciphertext);
    Values inputs{request.inputs[0], request.inputs[1].dependencies().at(dependency)};
    if (request.inputs.size() == 3)
      inputs.push_back(request.inputs[2]);
    auto block = runtime.submit({"contrib.search.score-block", std::move(inputs), shape});
    if (block.size() != 1)
      throw Error(ErrorCode::ExecutionFailed, "Search block recipe returned an incorrect result count");
    results.push_back(std::move(block[0]));
    remaining -= shape.document_count;
  }
  return results;
}
Values executeBlock(Runtime &runtime, const Provider &provider, const Request &request, bool prepared) {
  if (prepared)
    return search_detail::scoreBlockPrepared(runtime, request.inputs[0], request.inputs[1],
                                             request.inputs[2].dependencies()[0]);
  const auto width = std::any_cast<SearchShape>(request.attributes).width;
  return search_detail::scoreBlockOrdinary(*provider.native, request.inputs[0], request.inputs[1], width);
}
} // namespace
SearchKeyRequirements nativeSearchKeyRequirements(std::uint32_t width) {
  if (!powerOfTwo(width))
    throw runtime::Error(runtime::ErrorCode::InvalidRequest, "Search reduction width must be a positive power of two");
  SearchKeyRequirements result{true, {}};
  for (std::uint64_t step = 1; step < width; step *= 2)
    result.rotation_steps.push_back(static_cast<std::int64_t>(step));
  return result;
}

void registerSearchKernels(runtime::Runtime &runtime, runtime::NativeServices services,
                           const std::string &nativeLibraryPath) {
  registerMatvecKernels(runtime, services, nativeLibraryPath);
  auto provider = std::make_shared<Provider>();
  provider->services = std::move(services);
  provider->native = std::make_shared<NativeRuntime>(runtime, provider->services);
  runtime.addRecipe({"contrib.search.index", "contrib.search.index.native", 0,
                     [provider](const Request &r, std::string &why) { return indexApplicable(*provider, r, why); },
                     [provider](Runtime &rt, const Request &r) { return Values{executeIndex(rt, provider, r)}; }});
  runtime.addRecipe({"contrib.search.prepare-query", "contrib.search.prepare-query.native", 0,
                     [provider](const Request &r, std::string &why) { return prepareApplicable(*provider, r, why); },
                     [provider](Runtime &rt, const Request &r) { return Values{executePrepare(rt, provider, r)}; }});
  runtime.addRecipe({"contrib.search.score", "contrib.search.score.composed", 0,
                     [provider](const Request &r, std::string &why) { return scoreApplicable(*provider, r, why); },
                     [](Runtime &rt, const Request &r) { return executeScore(rt, r); }});
  for (bool prepared : {true, false}) {
    for (bool row : {true, false}) {
      const auto name =
        std::string("contrib.search.score-block.") + (row ? "row." : "tile.") + (prepared ? "prepared" : "ordinary");
      runtime.addRecipe(
        {"contrib.search.score-block", name, prepared ? 20 : 0,
         [provider, prepared, row](const Request &r, std::string &why) {
           return blockApplicable(*provider, r, prepared, row, why);
         },
         [provider, prepared](Runtime &rt, const Request &r) { return executeBlock(rt, *provider, r, prepared); }});
    }
  }
}
} // namespace fhenomenon::contrib
