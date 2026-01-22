include(CheckIPOSupported)
check_ipo_supported(RESULT lto_supported OUTPUT error)

option(DISABLE_LTO "Disable Link Time Optimization" OFF)

if(DISABLE_LTO)
  set(lto_supported FALSE)
  message(STATUS "LTO disabled by user")
elseif(lto_supported)
  message(STATUS "LTO enabled")
else()
  message(STATUS "LTO not supported: <${error}>")
endif()

function(add_lto_if_possible target)
  if(lto_supported)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()
endfunction()
