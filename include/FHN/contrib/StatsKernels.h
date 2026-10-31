#pragma once

#include "FHN/NativeRuntime.h"
#include <string>

namespace fhenomenon::contrib {

// Registers optional, compute-only recipes for contrib.stats.energy,
// contrib.stats.variance, contrib.packed.dot and contrib.packed.reduce.
// Request attributes must hold std::uint32_t width: a nonzero power of two
// no greater than the ciphertext logical slot count. The first three recipes
// consume two canonical ciphertexts with equal metadata; reduce consumes one.
// Energy computes (width*x-sum)^2/width; variance reduces that expression.
// The host supplies the corresponding replicated sum when using the result
// as a variance proxy. Centered statistics require input level >= 2; dot
// requires level >= 1. Products consume one level. Centered statistics report
// their actual output scale, width times the ordinary product scale; callers
// must not force a scale match with a differently normalized product.
// Partial reductions retain logical slots and accept any valid input level.
//
// nativeLibraryPath must name the same native plugin that supplies services.
// Missing optional exports cause recipe decline, allowing explicit fallbacks.
// Registered callbacks retain services and the library without retaining a
// client. Results interoperate with NativeRuntime. Staging and publication
// reserve two full ciphertext buffers while the native result is copied.
void registerStatsKernels(runtime::Runtime &, runtime::NativeServices, const std::string &nativeLibraryPath);

} // namespace fhenomenon::contrib
