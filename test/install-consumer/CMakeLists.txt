cmake_minimum_required(VERSION 3.19)
project(FhenomenonInstallConsumer LANGUAGES CXX)

find_package(Fhenomenon CONFIG REQUIRED)
add_executable(install-consumer main.cpp)
target_link_libraries(install-consumer PRIVATE Fhenomenon::Fhenomenon)
enable_testing()
add_test(NAME InstalledRuntime COMMAND install-consumer)

# The installed genuine plugin must run through installed public headers only.
find_library(FHN_NATIVE_CKKS NAMES ckks_fhn REQUIRED)
add_executable(native-install-consumer native.cpp rotation_contribution.cpp)
target_link_libraries(native-install-consumer PRIVATE Fhenomenon::Fhenomenon Fhenomenon::StatsKernels Fhenomenon::MatvecKernels ${CMAKE_DL_LIBS})
add_test(NAME InstalledNativeCkks COMMAND native-install-consumer "${FHN_NATIVE_CKKS}")
