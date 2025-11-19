cmake_minimum_required(VERSION 3.19)
project(
  Fhenomenon
  VERSION 0.0.1
  DESCRIPTION "Example c++ template project"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(PROJECT_LIB_NAME ${PROJECT_NAME})

# Ensure macOS SDK is detected so C++ standard headers (e.g., <cstddef>) are found
if(APPLE)
  # Discover the active SDK path if not already set
  if(NOT CMAKE_OSX_SYSROOT)
    execute_process(
      COMMAND xcrun --show-sdk-path
      OUTPUT_VARIABLE CMAKE_OSX_SYSROOT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
  endif()

  # Help Clang/LLVM find libc++ headers and link correctly when using non-Apple clang
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-isysroot ${CMAKE_OSX_SYSROOT} -stdlib=libc++)
    add_link_options(-isysroot ${CMAKE_OSX_SYSROOT} -stdlib=libc++)
  endif()
endif()

include(BuildType)
include(LTO)
include(CTest)
include(CheckLanguage)
include(CCache)

option(BUILD_DOCUMENTATION "Build doxygen documentation (requires doxygen)" ON)
option(BUILD_TEST "Build tests" ON)

# doxygen
if(BUILD_DOCUMENTATION)
  add_subdirectory(docs)
endif()

# set(SRCS src/Foo.cpp CACHE FILEPATH "Sources" FORCE)

add_subdirectory(external)
add_library(external-libs INTERFACE)
target_link_libraries(external-libs INTERFACE fmt::fmt
                                              nlohmann_json::nlohmann_json)

macro(add_fhenomenon_sources)
  file(GLOB_RECURSE SUBDIR_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
  set(FHENOMENON_SOURCES
      ${FHENOMENON_SOURCES} ${SUBDIR_SOURCES}
      PARENT_SCOPE)
endmacro()

set(FHENOMENON_SOURCES)
add_subdirectory(src)

message(STATUS "SOURCES ${FHENOMENON_SOURCES}")

add_library(${PROJECT_LIB_NAME} STATIC ${FHENOMENON_SOURCES})

target_include_directories(
  ${PROJECT_LIB_NAME}
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(${PROJECT_LIB_NAME}
                      PUBLIC $<BUILD_INTERFACE:external-libs>)

add_lto_if_possible(${PROJECT_LIB_NAME})

include(CompilerWarnings)
# target_set_warnings(${PROJECT_LIB_NAME})
target_set_warnings_as_errors(${PROJECT_LIB_NAME})

add_library(${PROJECT_LIB_NAME}::${PROJECT_LIB_NAME} ALIAS ${PROJECT_LIB_NAME})

# Define exported target
include(GNUInstallDirs)
install(
  TARGETS ${PROJECT_LIB_NAME}
  EXPORT ${PROJECT_LIB_NAME}Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION include
  PUBLIC_HEADER DESTINATION include)

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
        DESTINATION include/${PROJECT_LIB_NAME})

# Install include directory
install(DIRECTORY include/${PROJECT_LIB_NAME} DESTINATION include)

# Make Config.cmake and ConfigVersion.cmake
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${PROJECT_LIB_NAME}ConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_LIB_NAME}Config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_LIB_NAME})

# Install them
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_LIB_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_LIB_NAME}ConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_LIB_NAME})

if(BUILD_TEST)
  enable_testing()
  add_subdirectory(test)
  add_subdirectory(examples)
endif()