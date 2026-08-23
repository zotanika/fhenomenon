# Layer targets for the built-in CKKS backend.
#
# This file is invariant I5 from
# docs/superpowers/specs/2026-07-26-builtin-ckks-backend-design.md made
# mechanical: "layer boundaries are link-time facts, not review comments".
#
# Two mechanisms enforce it, and they catch different mistakes:
#
#   1. Each layer owns a PRIVATE include root at <layer-dir>/include. A layer
#      only sees the headers of the layers it links, so a wrong #include is a
#      compile error instead of something a reviewer has to notice. This is
#      why CKKS headers do NOT live in the repository-wide include/ directory:
#      a shared include root is exactly how layering erodes.
#
#   2. Every layer declares its index and may only depend on strictly lower
#      ones. An upward or sideways dependency is a configure-time error, so a
#      cycle cannot be introduced even between two layers that would otherwise
#      link cleanly.
#
# Layer indices follow the spec: 0 Params, 1 Tables, 2 Storage, 3 Arena,
# 4 eval, 5 Keys, 6 Backend. Arena is a leaf despite its index — the index
# orders what may depend on what, not what must.

# Declare one layer of the CKKS backend as its own static library.
#
#   fhn_ckks_layer(<name>
#     LAYER   <index>
#     SOURCES <file>...
#     [DEPENDS <lower-layer-target>...])
function(fhn_ckks_layer name)
  cmake_parse_arguments(ARG "" "LAYER" "SOURCES;DEPENDS" ${ARGN})

  if(NOT DEFINED ARG_LAYER)
    message(FATAL_ERROR "fhn_ckks_layer(${name}): LAYER index is required")
  endif()
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "fhn_ckks_layer(${name}): SOURCES is required")
  endif()
  if(NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include")
    message(
      FATAL_ERROR
        "fhn_ckks_layer(${name}): expected an include root at "
        "${CMAKE_CURRENT_SOURCE_DIR}/include. Each layer owns its headers so "
        "that a wrong #include cannot compile (invariant I5).")
  endif()

  add_library(${name} STATIC ${ARG_SOURCES})
  set_property(TARGET ${name} PROPERTY FHN_CKKS_LAYER ${ARG_LAYER})
  set_property(TARGET ${name} PROPERTY POSITION_INDEPENDENT_CODE ON)

  target_include_directories(${name}
                             PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_features(${name} PUBLIC cxx_std_17)

  foreach(dep IN LISTS ARG_DEPENDS)
    if(NOT TARGET ${dep})
      message(
        FATAL_ERROR
          "fhn_ckks_layer(${name}): dependency '${dep}' is not a target. "
          "Layers must be declared bottom-up.")
    endif()
    get_target_property(dep_layer ${dep} FHN_CKKS_LAYER)
    if(dep_layer MATCHES "NOTFOUND")
      message(
        FATAL_ERROR
          "fhn_ckks_layer(${name}): dependency '${dep}' is not a CKKS layer "
          "target. Layers may only depend on other layers.")
    endif()
    # Note: a plain `if(NOT dep_layer)` would misfire on layer 0, which is
    # falsy in CMake. Compare numerically.
    if(NOT dep_layer LESS ARG_LAYER)
      message(
        FATAL_ERROR
          "fhn_ckks_layer(${name}): layer ${ARG_LAYER} may not depend on "
          "'${dep}' (layer ${dep_layer}). Dependencies point downward only "
          "(invariant I5).")
    endif()
  endforeach()

  if(ARG_DEPENDS)
    target_link_libraries(${name} PUBLIC ${ARG_DEPENDS})
  endif()

  include(CompilerWarnings)
  target_set_warnings_as_errors(${name})
endfunction()
