#pragma once

#include <string_view>

namespace private_search {
// Standalone help is available before any fixture access or native key setup.
inline bool helpRequested(int argc, char **argv) {
  return argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h");
}

inline constexpr char application_usage[] = R"help(Usage: fhn-private-search NATIVE_LIBRARY FIXTURE_JSON [--query ID]
       fhn-private-search --help | -h

Encrypt document and query embeddings with native CKKS, run a scoped search,
then let the owning client decrypt and display its top five matches.

Arguments:
  NATIVE_LIBRARY   Path to the native CKKS plugin (.dylib on macOS, .so on Linux).
  FIXTURE_JSON     Fixture with precomputed document and query embeddings.

Options:
  --query ID       Select a fixture query by ID, not free-form text.
                   Default: the first query in the fixture (bundled: dispatch).
  --help, -h       Show this help without loading the fixture or native plugin.

This application always uses scoped execution with prepared eight-document
packing and prints the selected scope spans. For timing, layout comparisons,
all-query verification or scope-kernel toggles, use fhn-bench-private-search.

Example (from the repository root on macOS):
  ./build-search/bin/fhn-private-search ./build-search/lib/libckks_fhn.dylib examples/private_search/data/fixture.json --query memory
)help";

inline constexpr char benchmark_usage[] = R"help(Usage: fhn-bench-private-search NATIVE_LIBRARY FIXTURE_JSON [OPTIONS]
       fhn-bench-private-search --help | -h

Measure encrypted search and check every warm-up and measured result against
plaintext scores. By default: first query, tile-prepared mode, immediate
execution, one warm-up and three measured evaluations.

Arguments:
  NATIVE_LIBRARY       Path to the native CKKS plugin (.dylib / .so).
  FIXTURE_JSON         Fixture with precomputed document and query embeddings.

Options:
  --query ID           Select a fixture query by ID, not free-form text.
                      Default: first query. Cannot combine with --verify-all.
  --verify-all         Run every fixture query. Correctness checks run even
                      without this flag; it changes query coverage only.
  --compare            Run row-ordinary, row-prepared, tile-ordinary and
                      tile-prepared. Default: tile-prepared only.
                      Row packs one document per ciphertext; tile packs eight.
                      Prepared modes reuse explicit query state.
  --reps N             Measured evaluations per query and mode (1..1000;
                      default: 3), after one separately reported warm-up.
  --limit-documents N  Use the first N documents (1..100000; default: all).
                      If N exceeds the corpus size, use all documents.
                      With --scoped, 17 documents exercise collection + tail.
  --json OUTPUT        Write timings, errors, rankings and kernel diagnostics
                      to JSON. Default: console only. Overwrites OUTPUT;
                      its parent must exist and it cannot be FIXTURE_JSON.
  --scoped             Record block commands, then select scope kernels.
                      Default: immediate collection request. Adds capture and
                      selection to evaluation timing; preparation stays separate.
  --no-scope-kernels   Requires --scoped. Keep capture and liveness but use
                      ordinary request dispatch for every recorded command.
                      Prepared recipes and reusable state remain enabled.
  --help, -h           Show this help without loading the fixture or native plugin.

--compare chooses layouts/preparation modes; --scoped chooses dispatch.
Neither flag enables a compiler. Evaluation timing excludes key/index/query
setup, preparation, warm-up, decryption and ranking. --scoped also includes
capture/selection. The current collection scope kernel composes existing blocks;
selecting it does not imply new cryptographic fusion or a speedup.

Example (from the repository root on macOS):
  ./build-search/bin/fhn-bench-private-search ./build-search/lib/libckks_fhn.dylib examples/private_search/data/fixture.json --query memory --limit-documents 17 --scoped --reps 1
Add --no-scope-kernels to compare the same recording through request fallback.
See examples/private_search/README.md for complete comparison commands.
)help";
} // namespace private_search
