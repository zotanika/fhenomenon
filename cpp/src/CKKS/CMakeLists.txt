# Built-in RNS-CKKS backend.
#
# Deliberately NOT folded into the monolithic ${PROJECT_LIB_NAME} target: this
# tree is built as one static library per layer, and that separation is the
# enforcement mechanism for invariant I5, not a stylistic preference. See
# cmake/FhnCkksLayer.cmake for what the layer targets guarantee and why, and
# docs/native-ckks.md for the layers themselves.
#
# Layers are declared bottom-up so that fhn_ckks_layer() can check every
# dependency against an already-declared target.
include(FhnCkksLayer)

add_subdirectory(params)
add_subdirectory(tables)
add_subdirectory(storage)
add_subdirectory(arena)
add_subdirectory(eval)
add_subdirectory(keys)
add_subdirectory(backend)
