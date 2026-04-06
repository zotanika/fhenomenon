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

# --- Shared library for ExternalBackend testing ---
# Build ToyFheKernels as a shared lib so ExternalBackend can dlopen it
add_library(toyfhe_fhn SHARED ${CMAKE_SOURCE_DIR}/src/FHN/ToyFheKernels.cpp)
target_include_directories(toyfhe_fhn PRIVATE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
target_link_libraries(toyfhe_fhn PRIVATE ${PROJECT_LIB_NAME})
set_target_properties(toyfhe_fhn PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_executable(FhnExternalBackendTest FhnExternalBackendTest.cpp)
target_link_libraries(FhnExternalBackendTest PRIVATE ${PROJECT_LIB_NAME} gtest_main dl)
target_compile_definitions(FhnExternalBackendTest PRIVATE
  TEST_LIB_DIR="${CMAKE_BINARY_DIR}/lib")
add_dependencies(FhnExternalBackendTest toyfhe_fhn)
add_gtest_target_to_ctest(FhnExternalBackendTest)

# --- Cheddar FHN backend (GPU, optional) ---
set(CHEDDAR_FHE_DIR "${CMAKE_SOURCE_DIR}/refs/cheddar-fhe")
set(CHEDDAR_LIB_PATH "${CHEDDAR_FHE_DIR}/build/libcheddar.so")
if(EXISTS "${CHEDDAR_FHE_DIR}/include/core/Context.h" AND EXISTS "${CHEDDAR_LIB_PATH}")
  message(STATUS "Cheddar-FHE found, building cheddar_fhn backend and GPU tests")

  find_package(CUDAToolkit QUIET)

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
