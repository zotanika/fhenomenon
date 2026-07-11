# Flagship benchmark: encrypted matvec, fused vs decomposed dispatch.
# Binaries land in build/bin via the project-wide EXECUTABLE_OUTPUT_PATH.
add_executable(fhn-bench-matvec fhn_matvec_bench.cpp)
target_link_libraries(fhn-bench-matvec PRIVATE ${PROJECT_LIB_NAME})
