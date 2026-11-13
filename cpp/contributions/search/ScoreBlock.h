#pragma once
#include "FHN/NativeRuntime.h"
#include <cstdint>
#include <utility>

namespace fhenomenon::contrib::search_detail {
// The encrypted computation entry point. SearchKernels.cpp checks the request
// contract before calling these functions; neither function sees plaintext or a
// client. For width W, output[j*W] approximates dot(query, document_j).
// Other output lanes are mixed sliding sums, not document scores. See README.md.

inline runtime::Values scoreBlockOrdinary(runtime::NativeRuntime &native, const runtime::Value &encryptedQuery,
                                          const runtime::Value &encryptedDocuments, std::uint32_t width) {
  // HMULT includes native relinearization and rescale: one level is consumed.
  auto partial = native.op(FHN_HMULT, {encryptedQuery, encryptedDocuments});
  for (auto distance = width / 2; distance; distance /= 2)
    partial = native.op(FHN_ADD_CC, {partial, native.rotate(partial, distance)});
  return {std::move(partial)};
}

inline runtime::Values scoreBlockPrepared(runtime::Runtime &runtime, const runtime::Value &encryptedQuery,
                                          const runtime::Value &encryptedDocuments,
                                          const runtime::Value &matvecPreparation) {
  // Same score/level/scale contract. This larger kernel reuses the exact query's
  // prepared native source and key tables across document blocks and commands.
  return runtime.submit({"contrib.matvec.dot", {encryptedQuery, encryptedDocuments, matvecPreparation}, {}});
}
} // namespace fhenomenon::contrib::search_detail
