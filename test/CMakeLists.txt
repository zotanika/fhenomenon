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

add_executable(FhnBackendApiTest FhnBackendApiTest.cpp)
target_link_libraries(FhnBackendApiTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(FhnBackendApiTest)

add_executable(FhnExecutorTest FhnExecutorTest.cpp)
target_link_libraries(FhnExecutorTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(FhnExecutorTest)

add_executable(FhnToyFheTest FhnToyFheTest.cpp)
target_link_libraries(FhnToyFheTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(FhnToyFheTest)

add_executable(LowerToFhnProgramTest LowerToFhnProgramTest.cpp)
target_link_libraries(LowerToFhnProgramTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(LowerToFhnProgramTest)

add_executable(FhnIntegrationTest FhnIntegrationTest.cpp)
target_link_libraries(FhnIntegrationTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(FhnIntegrationTest)
