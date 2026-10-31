#pragma once
#include "FHN/fhn_circuit_api.h"

// A compiled circuit owns its copied graph, typed regions and reusable storage.
// Compile before preparing keys; prepare is trusted client work. Inputs may
// share handles and outputs may alias inputs. Distinct output values require
// distinct destinations. All outputs are published only after successful run.
// Calls on one context are serialized. Destroy executables before their context.
extern "C" {
FhnExecutable *ckks_fhn_compile_circuit(FhnBackendCtx *, const FhnCircuit *, const FhnBuffer *const *, uint32_t,
                                        const FhnCompileOptions *, char *, uint32_t);
int ckks_fhn_client_prepare_executable(FhnClientCtx *, FhnExecutable *);
int ckks_fhn_run_executable(FhnExecutable *, const FhnBuffer *const *, uint32_t, FhnBuffer *const *, uint32_t);
int ckks_fhn_executable_stats(const FhnExecutable *, FhnExecutableStats *);
void ckks_fhn_executable_free(FhnExecutable *);
}
