cmake_minimum_required(VERSION 3.19)
project(
  Fhenomenon
  VERSION 0.0.1
  DESCRIPTION "Fhenomenon C++ runtime and native providers"
  LANGUAGES CXX)

get_filename_component(FHENOMENON_REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(PROJECT_LIB_NAME ${PROJECT_NAME})

# Ensure macOS SDK is detected so C++ standard headers (e.g., <cstddef>) are
# found
if(APPLE)
  # Discover the active SDK path if not already set
  if(NOT CMAKE_OSX_SYSROOT)
    execute_process(
      COMMAND xcrun --show-sdk-path
      OUTPUT_VARIABLE CMAKE_OSX_SYSROOT
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  endif()

  # Help Clang/LLVM find libc++ headers and link correctly when using non-Apple
  # clang
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-isysroot ${CMAKE_OSX_SYSROOT} -stdlib=libc++)
    add_link_options(-isysroot ${CMAKE_OSX_SYSROOT} -stdlib=libc++)
  endif()
endif()

include(FhnBuiltinBackend)
include(BuildType)
include(LTO)
include(CTest)
include(CheckLanguage)
include(CCache)
include(GNUInstallDirs)

option(BUILD_DOCUMENTATION "Build doxygen documentation (requires doxygen)" ON)
option(BUILD_TEST "Build tests" ON)

# doxygen
if(BUILD_DOCUMENTATION)
  add_subdirectory("${FHENOMENON_REPOSITORY_ROOT}/docs" "${CMAKE_CURRENT_BINARY_DIR}/docs")
endif()

# set(SRCS src/Foo.cpp CACHE FILEPATH "Sources" FORCE)

add_subdirectory(external)

macro(add_fhenomenon_sources)
  file(GLOB_RECURSE SUBDIR_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
  set(FHENOMENON_SOURCES
      ${FHENOMENON_SOURCES} ${SUBDIR_SOURCES}
      PARENT_SCOPE)
endmacro()

set(FHENOMENON_SOURCES)
add_subdirectory(src)

# The built-in CKKS backend is built as one static library per layer, not
# folded into the monolithic library. add_fhenomenon_sources() globs src/
# recursively, so the separation only survives if it is filtered back out
# here. That separation IS the enforcement of invariant I5 — see
# cmake/FhnCkksLayer.cmake.
list(FILTER FHENOMENON_SOURCES EXCLUDE REGEX "/CKKS/")

message(STATUS "SOURCES ${FHENOMENON_SOURCES}")

add_library(${PROJECT_LIB_NAME} STATIC ${FHENOMENON_SOURCES})
set_target_properties(${PROJECT_LIB_NAME} PROPERTIES
  POSITION_INDEPENDENT_CODE ON ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

target_include_directories(
  ${PROJECT_LIB_NAME}
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_compile_features(${PROJECT_LIB_NAME} PUBLIC cxx_std_17)
target_link_libraries(${PROJECT_LIB_NAME} PUBLIC fmt::fmt nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})

add_lto_if_possible(${PROJECT_LIB_NAME})

include(CompilerWarnings)
# target_set_warnings(${PROJECT_LIB_NAME})
target_set_warnings_as_errors(${PROJECT_LIB_NAME})

add_library(${PROJECT_LIB_NAME}::${PROJECT_LIB_NAME} ALIAS ${PROJECT_LIB_NAME})

# Define exported target
install(
  TARGETS ${PROJECT_LIB_NAME}
  EXPORT ${PROJECT_LIB_NAME}Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export the target
install(
  EXPORT ${PROJECT_LIB_NAME}Targets
  FILE ${PROJECT_LIB_NAME}Targets.cmake
  NAMESPACE ${PROJECT_LIB_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_LIB_NAME})

# Versioning
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Version.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_LIB_NAME}/Version.hpp" @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_LIB_NAME}/Version.hpp"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_LIB_NAME})

# Install include directory
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.tpp")

# Make Config.cmake and ConfigVersion.cmake
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${PROJECT_LIB_NAME}ConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_LIB_NAME}Config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_LIB_NAME}
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR)

# Install them
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_LIB_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_LIB_NAME}ConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_LIB_NAME})

# The reference plugin is useful to corpus users even when tests are disabled.
add_library(toyfhe_fhn SHARED src/FHN/ToyFheKernels.cpp)
target_link_libraries(toyfhe_fhn PRIVATE ${PROJECT_LIB_NAME})
set_target_properties(toyfhe_fhn PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# A separately linked contribution using only the public runtime interfaces.
add_library(fhn_stats_contribution STATIC contributions/stats/StatsKernels.cpp)
target_link_libraries(fhn_stats_contribution PUBLIC ${PROJECT_LIB_NAME})
set_target_properties(fhn_stats_contribution PROPERTIES
  POSITION_INDEPENDENT_CODE ON EXPORT_NAME StatsKernels)
add_library(Fhenomenon::StatsKernels ALIAS fhn_stats_contribution)
install(TARGETS fhn_stats_contribution EXPORT ${PROJECT_LIB_NAME}Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_library(fhn_matvec_contribution STATIC contributions/matvec/MatvecKernels.cpp)
target_link_libraries(fhn_matvec_contribution PUBLIC ${PROJECT_LIB_NAME})
set_target_properties(fhn_matvec_contribution PROPERTIES
  POSITION_INDEPENDENT_CODE ON EXPORT_NAME MatvecKernels)
add_library(Fhenomenon::MatvecKernels ALIAS fhn_matvec_contribution)
install(TARGETS fhn_matvec_contribution EXPORT ${PROJECT_LIB_NAME}Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_library(fhn_search_contribution STATIC contributions/search/SearchKernels.cpp)
target_link_libraries(fhn_search_contribution PUBLIC fhn_matvec_contribution)
set_target_properties(fhn_search_contribution PROPERTIES
  POSITION_INDEPENDENT_CODE ON EXPORT_NAME SearchKernels)
add_library(Fhenomenon::SearchKernels ALIAS fhn_search_contribution)
install(TARGETS fhn_search_contribution EXPORT ${PROJECT_LIB_NAME}Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_subdirectory(benchmarks)
add_subdirectory(examples/private_search)

# Complex spectral workloads use the same public contribution/runtime boundary.
add_library(fhn_spectral_contribution STATIC contributions/spectral/SpectralKernels.cpp)
target_link_libraries(fhn_spectral_contribution PUBLIC fhn_matvec_contribution)
set_target_properties(fhn_spectral_contribution PROPERTIES
  POSITION_INDEPENDENT_CODE ON EXPORT_NAME SpectralKernels)
add_library(Fhenomenon::SpectralKernels ALIAS fhn_spectral_contribution)
install(TARGETS fhn_spectral_contribution EXPORT ${PROJECT_LIB_NAME}Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
add_subdirectory(examples/private_spectrum)
add_library(fhn_public_linear_contribution STATIC contributions/public_linear/PublicLinearKernels.cpp)
target_link_libraries(fhn_public_linear_contribution PUBLIC ${PROJECT_LIB_NAME})
set_target_properties(fhn_public_linear_contribution PROPERTIES
  POSITION_INDEPENDENT_CODE ON EXPORT_NAME PublicLinearKernels)
add_library(Fhenomenon::PublicLinearKernels ALIAS fhn_public_linear_contribution)
install(TARGETS fhn_public_linear_contribution EXPORT ${PROJECT_LIB_NAME}Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

option(FHN_BUILD_CUDA "Build Cheddar-derived CUDA kernels for the tensor provider" OFF)
add_subdirectory(providers/ckks_tensor)
add_subdirectory(bindings/tensor_runtime)

if(BUILD_TEST)
  enable_testing()
  add_subdirectory(test)
  add_subdirectory(examples)
endif()
