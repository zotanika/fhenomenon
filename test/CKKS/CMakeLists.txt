# Tests for the built-in CKKS backend.
#
# Each test links exactly ONE layer target — deliberately not
# ${PROJECT_LIB_NAME}, and not another layer. That is the check invariant I5
# actually buys: if a layer's test starts needing something from above it to
# build, the link fails here instead of a reviewer having to notice. Adding
# ${PROJECT_LIB_NAME} to any of these lines silently disables that check.
add_executable(CkksParamsTest ParamsTest.cpp)
target_link_libraries(CkksParamsTest PRIVATE fhn_ckks_params gtest_main)
add_gtest_target_to_ctest(CkksParamsTest)

add_executable(CkksNttTest NttTest.cpp)
target_link_libraries(CkksNttTest PRIVATE fhn_ckks_tables gtest_main)
add_gtest_target_to_ctest(CkksNttTest)

add_executable(CkksBConvTest BConvTest.cpp)
target_link_libraries(CkksBConvTest PRIVATE fhn_ckks_tables gtest_main)
add_gtest_target_to_ctest(CkksBConvTest)

add_executable(CkksLayoutTest LayoutTest.cpp)
target_link_libraries(CkksLayoutTest PRIVATE fhn_ckks_storage gtest_main)
add_gtest_target_to_ctest(CkksLayoutTest)

add_executable(CkksArenaTest ArenaTest.cpp)
target_link_libraries(CkksArenaTest PRIVATE fhn_ckks_arena gtest_main)
add_gtest_target_to_ctest(CkksArenaTest)

add_executable(CkksStorageTest StorageTest.cpp)
target_link_libraries(CkksStorageTest PRIVATE fhn_ckks_storage gtest_main)
add_gtest_target_to_ctest(CkksStorageTest)

add_executable(CkksCrtTest CrtTest.cpp)
target_link_libraries(CkksCrtTest PRIVATE fhn_ckks_tables gtest_main)
add_gtest_target_to_ctest(CkksCrtTest)

add_executable(CkksEvalTest EvalTest.cpp)
target_link_libraries(CkksEvalTest PRIVATE fhn_ckks_eval gtest_main)
add_gtest_target_to_ctest(CkksEvalTest)

add_executable(CkksKeysTest KeysTest.cpp)
target_link_libraries(CkksKeysTest PRIVATE fhn_ckks_keys gtest_main)
add_gtest_target_to_ctest(CkksKeysTest)

add_executable(CkksBackendTest BackendTest.cpp)
target_link_libraries(CkksBackendTest PRIVATE fhn_ckks_backend gtest_main)
add_gtest_target_to_ctest(CkksBackendTest)

add_executable(CkksRegionTest RegionTest.cpp)
target_link_libraries(CkksRegionTest PRIVATE fhn_ckks_eval gtest_main)
add_gtest_target_to_ctest(CkksRegionTest)

add_executable(CkksCircuitTest CircuitTest.cpp)
target_link_libraries(CkksCircuitTest PRIVATE fhn_ckks_backend gtest_main)
add_gtest_target_to_ctest(CkksCircuitTest)

add_executable(CkksPreparedRotationTest PreparedRotationTest.cpp)
target_link_libraries(CkksPreparedRotationTest PRIVATE fhn_ckks_backend gtest_main)
add_gtest_target_to_ctest(CkksPreparedRotationTest)

add_executable(CkksCoarseKernelsTest CoarseKernelsTest.cpp)
target_link_libraries(CkksCoarseKernelsTest PRIVATE fhn_ckks_backend gtest_main)
add_gtest_target_to_ctest(CkksCoarseKernelsTest)

add_executable(CkksPreparedArithmeticTest PreparedArithmeticTest.cpp)
target_link_libraries(CkksPreparedArithmeticTest PRIVATE fhn_ckks_eval gtest_main)
add_gtest_target_to_ctest(CkksPreparedArithmeticTest)

add_executable(CkksPreparedMatvecTest PreparedMatvecTest.cpp)
target_link_libraries(CkksPreparedMatvecTest PRIVATE fhn_ckks_backend gtest_main)
add_gtest_target_to_ctest(CkksPreparedMatvecTest)
