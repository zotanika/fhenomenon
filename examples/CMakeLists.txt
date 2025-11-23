add_executable(compuon-basic compuon.cpp)
target_link_libraries(compuon-basic PUBLIC Fhenomenon)

add_executable(session-basic session.cpp)
target_link_libraries(session-basic PUBLIC Fhenomenon)

add_executable(session1-basic session1.cpp)
target_link_libraries(session1-basic PUBLIC Fhenomenon)

add_executable(session2-basic session2.cpp)
target_link_libraries(session2-basic PUBLIC Fhenomenon)
