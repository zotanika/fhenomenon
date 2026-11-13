# The public BuiltinBackend frontend is the in-tree toy reference. Native CKKS
# is built separately and loaded through the public FHN runtime interfaces.
set(BUILTIN_BACKEND "TOY" CACHE STRING "Built-in frontend implementation (TOY)")
set_property(CACHE BUILTIN_BACKEND PROPERTY STRINGS "TOY")
if(NOT BUILTIN_BACKEND STREQUAL "TOY")
  message(FATAL_ERROR
    "Only the built-in TOY frontend is supported by BUILTIN_BACKEND. "
    "Native CKKS is provided by the separate ckks_fhn target.")
endif()
