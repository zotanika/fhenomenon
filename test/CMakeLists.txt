# Add a gtest target to ctest. Make the target report a xml file.
function(add_gtest_target_to_ctest target)
  add_test(NAME "${target}" COMMAND "${target}"
                                    --gtest_output=xml:${target}.xml)
endfunction()

cpmaddpackage(
  NAME
  googletest
  GITHUB_REPOSITORY
  google/googletest
  GIT_TAG
  v1.17.0
  OPTIONS
  "INSTALL_GTEST OFF"
  "BUILD_GMOCK OFF")

add_executable(FooTest Foo.cpp FooTest.cpp)
target_link_libraries(FooTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(FooTest)

add_executable(FhnProgramTest FhnProgramTest.cpp)
target_link_libraries(FhnProgramTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(FhnProgramTest)

add_executable(RuntimeTest RuntimeTest.cpp)
target_link_libraries(RuntimeTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(RuntimeTest)

add_executable(ScopeTest ScopeTest.cpp)
target_link_libraries(ScopeTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(ScopeTest)

add_executable(NativeScopeTest NativeScopeTest.cpp)
target_link_libraries(NativeScopeTest PRIVATE fhn_private_search_example gtest_main)
target_compile_definitions(NativeScopeTest PRIVATE FHN_NATIVE_TEST_LIBRARY="$<TARGET_FILE:ckks_fhn>")
add_dependencies(NativeScopeTest ckks_fhn)
add_gtest_target_to_ctest(NativeScopeTest)

add_executable(SearchScopeTest SearchScopeTest.cpp)
target_link_libraries(SearchScopeTest PRIVATE fhn_private_search_example gtest_main)
target_compile_definitions(SearchScopeTest PRIVATE FHN_NATIVE_TEST_LIBRARY="$<TARGET_FILE:ckks_fhn>")
add_dependencies(SearchScopeTest ckks_fhn)
add_gtest_target_to_ctest(SearchScopeTest)

add_executable(NativeRuntimeTest NativeRuntimeTest.cpp)
target_link_libraries(NativeRuntimeTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
target_compile_definitions(NativeRuntimeTest PRIVATE
  FHN_NATIVE_TEST_LIBRARY="$<TARGET_FILE:ckks_fhn>")
add_dependencies(NativeRuntimeTest ckks_fhn)
add_gtest_target_to_ctest(NativeRuntimeTest)

add_executable(StatsContributionTest StatsContributionTest.cpp)
target_link_libraries(StatsContributionTest PRIVATE fhn_stats_contribution gtest_main)
target_compile_definitions(StatsContributionTest PRIVATE
  FHN_NATIVE_TEST_LIBRARY="$<TARGET_FILE:ckks_fhn>"
  FHN_NON_COARSE_TEST_LIBRARY="$<TARGET_FILE:toyfhe_fhn>")
add_dependencies(StatsContributionTest ckks_fhn toyfhe_fhn)
add_gtest_target_to_ctest(StatsContributionTest)

add_executable(MatvecContributionTest MatvecContributionTest.cpp)
target_link_libraries(MatvecContributionTest PRIVATE fhn_matvec_contribution gtest_main)
target_compile_definitions(MatvecContributionTest PRIVATE
  FHN_NATIVE_TEST_LIBRARY="$<TARGET_FILE:ckks_fhn>"
  FHN_NON_MATVEC_TEST_LIBRARY="$<TARGET_FILE:toyfhe_fhn>")
add_dependencies(MatvecContributionTest ckks_fhn toyfhe_fhn)
add_gtest_target_to_ctest(MatvecContributionTest)

add_executable(SearchContributionTest SearchContributionTest.cpp)
target_link_libraries(SearchContributionTest PRIVATE fhn_search_contribution gtest_main)
target_compile_definitions(SearchContributionTest PRIVATE
  FHN_NATIVE_TEST_LIBRARY="$<TARGET_FILE:ckks_fhn>")
add_dependencies(SearchContributionTest ckks_fhn)
add_gtest_target_to_ctest(SearchContributionTest)

add_executable(PrivateSearchFixtureTest PrivateSearchFixtureTest.cpp)
target_link_libraries(PrivateSearchFixtureTest PRIVATE nlohmann_json::nlohmann_json gtest_main)
add_gtest_target_to_ctest(PrivateSearchFixtureTest)

add_test(NAME PrivateSearchExample COMMAND fhn-private-search
  "$<TARGET_FILE:ckks_fhn>" "${PROJECT_SOURCE_DIR}/examples/private_search/data/fixture.json" --query memory)
set_tests_properties(PrivateSearchExample PROPERTIES TIMEOUT 60)

add_executable(PrivateSearchWorkflowTest PrivateSearchWorkflowTest.cpp)
target_link_libraries(PrivateSearchWorkflowTest PRIVATE fhn_private_search_example gtest_main)
target_compile_definitions(PrivateSearchWorkflowTest PRIVATE FHN_NATIVE_TEST_LIBRARY="$<TARGET_FILE:ckks_fhn>")
add_dependencies(PrivateSearchWorkflowTest ckks_fhn)
add_gtest_target_to_ctest(PrivateSearchWorkflowTest)

add_test(NAME PrivateSearchBenchmark COMMAND fhn-bench-private-search
  "$<TARGET_FILE:ckks_fhn>" "${PROJECT_SOURCE_DIR}/examples/private_search/data/fixture.json"
  --verify-all --compare --reps 1 --limit-documents 17)
set_tests_properties(PrivateSearchBenchmark PROPERTIES TIMEOUT 600)

add_test(NAME PrivateSearchScopedBenchmark COMMAND fhn-bench-private-search
  "$<TARGET_FILE:ckks_fhn>" "${PROJECT_SOURCE_DIR}/examples/private_search/data/fixture.json"
  --query memory --compare --reps 1 --limit-documents 17 --scoped)
set_tests_properties(PrivateSearchScopedBenchmark PROPERTIES TIMEOUT 600)

add_executable(FhnBackendApiTest FhnBackendApiTest.cpp)
target_link_libraries(FhnBackendApiTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(FhnBackendApiTest)

add_executable(FhnExecutorTest FhnExecutorTest.cpp)
target_link_libraries(FhnExecutorTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(FhnExecutorTest)

add_executable(FhnToyFheTest FhnToyFheTest.cpp)
target_link_libraries(FhnToyFheTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(FhnToyFheTest)

add_executable(ToyFheEngineTest ToyFheEngineTest.cpp)
target_link_libraries(ToyFheEngineTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(ToyFheEngineTest)

add_executable(SessionTest SessionTest.cpp)
target_link_libraries(SessionTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(SessionTest)

# Deliberately never creates a Session: pins the no-session (eager) paths.
add_executable(NoSessionTest NoSessionTest.cpp)
target_link_libraries(NoSessionTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(NoSessionTest)

add_executable(ParameterTest ParameterTest.cpp)
target_link_libraries(ParameterTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(ParameterTest)

add_executable(LowerToFhnProgramTest LowerToFhnProgramTest.cpp)
target_link_libraries(LowerToFhnProgramTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(LowerToFhnProgramTest)

add_executable(FhnIntegrationTest FhnIntegrationTest.cpp)
target_link_libraries(FhnIntegrationTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(FhnIntegrationTest)

add_executable(FhnMovementPlanTest FhnMovementPlanTest.cpp)
target_link_libraries(FhnMovementPlanTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(FhnMovementPlanTest)

add_executable(CorpusUnitTest CorpusUnitTest.cpp)
target_link_libraries(CorpusUnitTest PRIVATE fhn_corpus_lib gtest_main)
add_gtest_target_to_ctest(CorpusUnitTest)

# --- Flagship benchmark smoke test ---
# fhn-bench-matvec (see benchmarks/) verifies M·v correctness on both the
# fused and the decomposed dispatch paths before timing anything and exits
# nonzero on mismatch, so the binary itself doubles as a CI test.
add_test(NAME FhnMatvecBenchTest COMMAND fhn-bench-matvec --n 8 --reps 1)

# The corpus binary self-checks (plan sanity per shape, oracle-verified
# execution of the depth-zero subset on ToyFHE) and exits nonzero on any
# failure. Packed fixed-point stats/logreg exceed ToyFHE's magnitude range;
# the separate verification test requires their mismatches to fail clearly.
add_test(NAME FhnCorpusTest
         COMMAND fhn-corpus --backend $<TARGET_FILE:toyfhe_fhn> --prefix toyfhe_ --max-depth 0 --budget-bytes min)

add_test(NAME FhnCorpusVerificationTest
  COMMAND ${CMAKE_COMMAND}
    -DCORPUS=$<TARGET_FILE:fhn-corpus>
    -DTOY_BACKEND=$<TARGET_FILE:toyfhe_fhn>
    -DSCALAR_BACKEND=$<TARGET_FILE:partial_fhn>
    -DHROT_BACKEND=$<TARGET_FILE:hrot_fhn>
    -P ${CMAKE_CURRENT_SOURCE_DIR}/CorpusVerificationTest.cmake)

# Verify the public package from a clean, relocated install prefix.
add_test(NAME FhnInstallConsumerTest
  COMMAND ${CMAKE_COMMAND}
    -DPROJECT_BUILD_DIR=${CMAKE_BINARY_DIR}
    -DCONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/install-consumer
    -DCONFIG=$<CONFIG>
    -DCONSUMER_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    "-DCONSUMER_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
    "-DCONSUMER_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/InstallConsumerTest.cmake)

add_executable(FhnExternalBackendTest FhnExternalBackendTest.cpp)
target_link_libraries(FhnExternalBackendTest PRIVATE ${PROJECT_LIB_NAME} gtest_main dl)
target_compile_definitions(FhnExternalBackendTest PRIVATE
  TEST_LIB_DIR="${CMAKE_BINARY_DIR}/lib")
add_dependencies(FhnExternalBackendTest toyfhe_fhn)
add_gtest_target_to_ctest(FhnExternalBackendTest)

target_compile_definitions(CorpusUnitTest PRIVATE
  TEST_LIB_DIR="${CMAKE_BINARY_DIR}/lib")
add_dependencies(CorpusUnitTest toyfhe_fhn)

# Deliberately exports a PARTIAL level-model trio: loaders must ignore it.
add_library(partial_fhn SHARED partial_fhn_backend.cpp)
target_include_directories(partial_fhn PRIVATE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
set_target_properties(partial_fhn PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
add_dependencies(FhnExternalBackendTest partial_fhn)
add_dependencies(CorpusUnitTest partial_fhn)

# Valid backend capability variant: HROT + ADD can implement HROT_ADD.
add_library(hrot_fhn SHARED hrot_fhn_backend.cpp)
target_link_libraries(hrot_fhn PRIVATE ${PROJECT_LIB_NAME})
set_target_properties(hrot_fhn PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
add_dependencies(CorpusUnitTest hrot_fhn)

# --- Cheddar FHN backend (GPU, optional) ---
set(CHEDDAR_FHE_DIR "${CMAKE_SOURCE_DIR}/refs/cheddar-fhe")
set(CHEDDAR_LIB_PATH "${CHEDDAR_FHE_DIR}/build/libcheddar.so")
if(EXISTS "${CHEDDAR_FHE_DIR}/include/core/Context.h" AND EXISTS "${CHEDDAR_LIB_PATH}")
  message(STATUS "Cheddar-FHE found, building cheddar_fhn backend and GPU tests")

  find_package(CUDAToolkit REQUIRED)

  # Cheddar's headers depend on RMM and Thrust from its build tree
  set(CHEDDAR_BUILD_DIR "${CHEDDAR_FHE_DIR}/build")
  set(CHEDDAR_RMM_INCLUDE "${CHEDDAR_BUILD_DIR}/_deps/rmm-src/include")
  set(CHEDDAR_THRUST_INCLUDE "${CHEDDAR_BUILD_DIR}/_deps/thrust-src")
  set(CHEDDAR_CUB_INCLUDE "${CHEDDAR_BUILD_DIR}/_deps/thrust-src/dependencies/cub")
  # spdlog is required by RMM's logger.hpp
  set(CHEDDAR_SPDLOG_INCLUDE "/tmp/spdlog/include")

  add_library(cheddar_fhn SHARED
    ${CMAKE_SOURCE_DIR}/src/FHN/cheddar/CheddarFhnBackend.cpp)
  target_include_directories(cheddar_fhn PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    ${CHEDDAR_FHE_DIR}/include
    ${CHEDDAR_RMM_INCLUDE}
    ${CHEDDAR_THRUST_INCLUDE}
    ${CHEDDAR_CUB_INCLUDE}
    ${CHEDDAR_SPDLOG_INCLUDE})
  target_link_libraries(cheddar_fhn PRIVATE
    ${CHEDDAR_LIB_PATH}
    nlohmann_json::nlohmann_json)
  if(CUDAToolkit_FOUND)
    target_link_libraries(cheddar_fhn PRIVATE CUDA::cudart)
  endif()
  set_target_properties(cheddar_fhn PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

  add_executable(FhnCheddarGpuTest FhnCheddarGpuTest.cpp)
  target_include_directories(FhnCheddarGpuTest PRIVATE
    ${CHEDDAR_FHE_DIR}/include
    ${CHEDDAR_RMM_INCLUDE}
    ${CHEDDAR_THRUST_INCLUDE}
    ${CHEDDAR_CUB_INCLUDE}
    ${CHEDDAR_SPDLOG_INCLUDE})
  target_link_libraries(FhnCheddarGpuTest PRIVATE
    ${PROJECT_LIB_NAME} gtest_main dl
    ${CHEDDAR_LIB_PATH})
  if(CUDAToolkit_FOUND)
    target_link_libraries(FhnCheddarGpuTest PRIVATE CUDA::cudart)
  endif()
  target_compile_definitions(FhnCheddarGpuTest PRIVATE
    TEST_LIB_DIR="${CMAKE_BINARY_DIR}/lib"
    PARAM_DIR="${CHEDDAR_FHE_DIR}/parameters")
  add_dependencies(FhnCheddarGpuTest cheddar_fhn)
  add_gtest_target_to_ctest(FhnCheddarGpuTest)
else()
  message(STATUS "Cheddar-FHE not found or not built, skipping GPU tests")
endif()

# Built-in CKKS backend: one test executable per layer, each linking only its
# own layer target.
add_subdirectory(CKKS)

# Genuine CPU CKKS integration, requiring actual execution of every chosen
# shape and both fused and primitive paths (no Toy arithmetic in this test).
add_test(NAME CkksCorpusMatvec
  COMMAND fhn-corpus --backend $<TARGET_FILE:ckks_fhn> --prefix ckks_
    --shape matvec-fused --max-depth 2 --require-execution --compare-fallback
    --atol 0.0001 --rtol 0.000001 --budget-bytes min)
set_tests_properties(CkksCorpusMatvec PROPERTIES TIMEOUT 180)
target_compile_definitions(CorpusUnitTest PRIVATE CKKS_TEST_LIBRARY="$<TARGET_FILE:ckks_fhn>")
add_dependencies(CorpusUnitTest ckks_fhn)

add_executable(CorpusCircuitTest CorpusCircuitTest.cpp)
target_link_libraries(CorpusCircuitTest PRIVATE fhn_corpus_lib gtest_main)
add_gtest_target_to_ctest(CorpusCircuitTest)
target_compile_definitions(CorpusCircuitTest PRIVATE
  CKKS_TEST_LIBRARY="$<TARGET_FILE:ckks_fhn>"
  TOY_TEST_LIBRARY="$<TARGET_FILE:toyfhe_fhn>")
add_dependencies(CorpusCircuitTest ckks_fhn toyfhe_fhn)

add_test(NAME CorpusCircuitVerificationTest
  COMMAND ${CMAKE_COMMAND}
    -DCIRCUIT_BENCH=$<TARGET_FILE:fhn-bench-circuit>
    -DCKKS_BACKEND=$<TARGET_FILE:ckks_fhn>
    -P ${CMAKE_CURRENT_SOURCE_DIR}/CorpusCircuitVerificationTest.cmake)
set_tests_properties(CorpusCircuitVerificationTest PROPERTIES TIMEOUT 600)

add_test(NAME LightweightRuntimeCorpusTest
  COMMAND fhn-bench-runtime $<TARGET_FILE:ckks_fhn> 1)
set_tests_properties(LightweightRuntimeCorpusTest PROPERTIES TIMEOUT 600)

add_test(NAME MatvecReuseCorpusTest
  COMMAND fhn-bench-runtime $<TARGET_FILE:ckks_fhn> 1 matvec-reuse)
set_tests_properties(MatvecReuseCorpusTest PROPERTIES TIMEOUT 600)
