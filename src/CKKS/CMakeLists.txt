# Built-in RNS-CKKS backend.
#
# Deliberately NOT folded into the monolithic ${PROJECT_LIB_NAME} target: this
# tree is built as one static library per layer, and that separation is the
# enforcement mechanism for invariant I5, not a stylistic preference. See
# cmake/FhnCkksLayer.cmake for what the layer targets guarantee, and
# docs/superpowers/specs/2026-07-26-builtin-ckks-backend-design.md for why.
#
# Layers are declared bottom-up so that fhn_ckks_layer() can check every
# dependency against an already-declared target.
include(FhnCkksLayer)

add_subdirectory(params)
add_subdirectory(arena)
