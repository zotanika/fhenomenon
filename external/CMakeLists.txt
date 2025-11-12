include(CPM)

# Set C++ standard before including packages that need it
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cpmaddpackage("gh:TheLartians/PackageProject.cmake@1.8.0")

cpmaddpackage(
  NAME
  fmt
  GIT_TAG
  8.1.1
  GITHUB_REPOSITORY
  fmtlib/fmt
  OPTIONS
  "FMT_INSTALL YES")

cpmaddpackage(
  NAME
  nlohmann_json
  GITHUB_REPOSITORY
  nlohmann/json
  VERSION
  3.11.3
  OPTIONS
  EXCLUDE_FROM_ALL
  YES)

# Microsoft SEAL library
# Try using FetchContent directly for better control
include(FetchContent)

FetchContent_Declare(
  seal
  GIT_REPOSITORY https://github.com/microsoft/SEAL.git
  GIT_TAG        v4.1.1
  GIT_SHALLOW    TRUE
)

# Set SEAL options before fetching
set(SEAL_BUILD_DEPS OFF CACHE BOOL "Build SEAL dependencies" FORCE)
set(SEAL_BUILD_EXAMPLES OFF CACHE BOOL "Build SEAL examples" FORCE)
set(SEAL_BUILD_TESTS OFF CACHE BOOL "Build SEAL tests" FORCE)
set(SEAL_USE_MSGSL OFF CACHE BOOL "Use MS-GSL" FORCE)
set(SEAL_USE_ZLIB ON CACHE BOOL "Use ZLIB" FORCE)
set(SEAL_USE_ZSTD OFF CACHE BOOL "Use ZSTD" FORCE)
set(SEAL_USE_ALIGNED_ALLOC OFF CACHE BOOL "Use aligned alloc" FORCE)
set(SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT ON CACHE BOOL "Throw on transparent ciphertext" FORCE)

# Make sure SEAL uses C++17
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard" FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require C++ standard" FORCE)

FetchContent_MakeAvailable(seal)

  