add_library(fhn_private_spectrum_example STATIC Client.cpp Detector.cpp Fixture.cpp Workflow.cpp)
target_link_libraries(fhn_private_spectrum_example PUBLIC fhn_spectral_contribution ${CMAKE_DL_LIBS})
target_include_directories(fhn_private_spectrum_example PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(fhn-private-spectrum main.cpp)
target_link_libraries(fhn-private-spectrum PRIVATE fhn_private_spectrum_example)
add_dependencies(fhn-private-spectrum ckks_fhn)
install(TARGETS fhn-private-spectrum RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
