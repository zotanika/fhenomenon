add_library(fhn_private_search_example STATIC TrustedClient.cpp EncryptedSearch.cpp ScopedSearch.cpp)
target_link_libraries(fhn_private_search_example PUBLIC fhn_search_contribution nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})
target_include_directories(fhn_private_search_example PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(fhn-private-search main.cpp)
target_link_libraries(fhn-private-search PRIVATE fhn_private_search_example)
add_dependencies(fhn-private-search ckks_fhn)

add_executable(fhn-bench-private-search benchmark.cpp)
target_link_libraries(fhn-bench-private-search PRIVATE fhn_private_search_example)
target_compile_definitions(fhn-bench-private-search PRIVATE FHN_SEARCH_BUILD_TYPE="$<CONFIG>")
add_dependencies(fhn-bench-private-search ckks_fhn)
install(TARGETS fhn-private-search fhn-bench-private-search RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES data/fixture.json DESTINATION ${CMAKE_INSTALL_DATADIR}/fhenomenon/private-search)
