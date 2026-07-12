# Flagship benchmark: encrypted matvec, fused vs decomposed dispatch.
# Binaries land in build/bin via the project-wide EXECUTABLE_OUTPUT_PATH.
add_executable(fhn-bench-matvec fhn_matvec_bench.cpp)
target_link_libraries(fhn-bench-matvec PRIVATE ${PROJECT_LIB_NAME})

# --- Corpus library (shapes, oracle, backend loader) ---
add_library(fhn_corpus_lib STATIC
  corpus/corpus_oracle.cpp)
target_include_directories(fhn_corpus_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/corpus)
target_link_libraries(fhn_corpus_lib PUBLIC ${PROJECT_LIB_NAME})
