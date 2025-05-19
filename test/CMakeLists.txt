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
