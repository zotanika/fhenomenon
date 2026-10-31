fhn_ckks_layer(fhn_ckks_backend LAYER 6 SOURCES Backend.cpp Circuit.cpp PreparedRotation.cpp CoarseKernels.cpp PreparedMatvec.cpp DEPENDS fhn_ckks_keys)
# Only the ABI adapter can see the host API; lower layers cannot include it.
target_include_directories(fhn_ckks_backend PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(fhn_ckks_backend PRIVATE nlohmann_json::nlohmann_json)
add_library(ckks_fhn SHARED Plugin.cpp)
target_link_libraries(ckks_fhn PRIVATE fhn_ckks_backend)
set_target_properties(ckks_fhn PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
install(TARGETS ckks_fhn LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
