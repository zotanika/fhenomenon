add_fhenomenon_sources()

add_subdirectory(Backend)
add_subdirectory(FHN)
add_subdirectory(Parameter)
add_subdirectory(Scheduler)
add_subdirectory(Session)

# Builds its own per-layer targets and is excluded from FHENOMENON_SOURCES by
# the root CMakeLists — see src/CKKS/CMakeLists.txt.
add_subdirectory(CKKS)
