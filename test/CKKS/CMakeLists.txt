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

add_executable(CkksArenaTest ArenaTest.cpp)
target_link_libraries(CkksArenaTest PRIVATE fhn_ckks_arena gtest_main)
add_gtest_target_to_ctest(CkksArenaTest)
