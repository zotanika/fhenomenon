include(CPM)

# Set C++ standard before including packages that need it
set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "C++ standard")
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
