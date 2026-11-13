#pragma once
#include "CKKS/Arena.h"
#include "CKKS/EvalTables.h"
#include "CKKS/Keys.h"
#include "CKKS/Storage.h"
#include "FHN/fhn_backend_api.h"
#include <memory>
#include <unordered_map>

// L6-only object definitions. The compute object has no client/secret pointer.
// All public mutation paths publish canonical residues into opaque buffers.
struct FhnBackendCtx {
  std::unique_ptr<fhenomenon::ckks::EvaluationTables> tables;
  fhenomenon::ckks::Arena scratch;
  std::size_t stride;
  std::weak_ptr<const fhenomenon::ckks::EvaluationKey> relin;
  std::unordered_map<uint64_t, std::weak_ptr<const fhenomenon::ckks::EvaluationKey>> rotations;
  bool client_created = false;
  explicit FhnBackendCtx(std::unique_ptr<fhenomenon::ckks::EvaluationTables>);
};
struct FhnBuffer {
  FhnBackendCtx *owner;
  fhenomenon::ckks::PolySlab slab;
  fhenomenon::ckks::CiphertextLayout layout{};
  uint32_t logical_slots = 0;
  bool certified = false;
  explicit FhnBuffer(FhnBackendCtx *ctx) : owner(ctx), slab(3 * ctx->stride) {}
  fhenomenon::ckks::CiphertextView view() { return {slab.data(), slab.capacityWords(), owner->stride, layout}; }
  fhenomenon::ckks::ConstCiphertextView view() const {
    return {slab.data(), slab.capacityWords(), owner->stride, layout};
  }
};
struct FhnClientCtx {
  FhnBackendCtx *backend;
  fhenomenon::ckks::Arena scratch;
  fhenomenon::ckks::SystemRandom random;
  std::unique_ptr<fhenomenon::ckks::SecretKey> secret;
  std::unique_ptr<fhenomenon::ckks::PublicKey> public_key;
  std::shared_ptr<const fhenomenon::ckks::EvaluationKey> relin;
  std::unordered_map<uint64_t, std::shared_ptr<const fhenomenon::ckks::EvaluationKey>> rotations;
  explicit FhnClientCtx(FhnBackendCtx *);
};
