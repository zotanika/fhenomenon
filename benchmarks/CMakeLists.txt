# Flagship benchmark: encrypted matvec, fused vs decomposed dispatch.
# Binaries land in build/bin via the project-wide EXECUTABLE_OUTPUT_PATH.
add_executable(fhn-bench-matvec fhn_matvec_bench.cpp)
target_link_libraries(fhn-bench-matvec PRIVATE ${PROJECT_LIB_NAME})

# --- Corpus library (shapes, oracle, backend loader) ---
add_library(fhn_corpus_lib STATIC
  corpus/corpus_circuit.cpp
  corpus/corpus_oracle.cpp
  corpus/corpus_backend.cpp
  corpus/corpus_shapes.cpp)
target_include_directories(fhn_corpus_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/corpus)
target_link_libraries(fhn_corpus_lib PUBLIC ${PROJECT_LIB_NAME} dl)

add_executable(fhn-corpus corpus/fhn_corpus_main.cpp)
target_link_libraries(fhn-corpus PRIVATE fhn_corpus_lib)
add_dependencies(fhn-corpus toyfhe_fhn)

# Reusable native circuit evaluation; independent oracle and optional paired
# comparison against a preserved legacy plugin, excluding setup/crypto I/O.
add_executable(fhn-bench-circuit fhn_circuit_bench.cpp)
target_link_libraries(fhn-bench-circuit PRIVATE fhn_corpus_lib)
add_dependencies(fhn-bench-circuit ckks_fhn)

add_executable(fhn-bench-runtime fhn_runtime_bench.cpp)
target_link_libraries(fhn-bench-runtime PRIVATE fhn_corpus_lib fhn_stats_contribution fhn_matvec_contribution)
add_dependencies(fhn-bench-runtime ckks_fhn)

# Isolated diagnostic: coarse matvec phases and immutable-key arithmetic probes.
# Backend internals are intentionally limited to this profiling executable.
add_executable(fhn-matvec-profile fhn_matvec_profile.cpp)
target_link_libraries(fhn-matvec-profile PRIVATE fhn_ckks_backend nlohmann_json::nlohmann_json)
