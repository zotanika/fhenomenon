# Run against the installed package, with no source-tree include paths.
if(NOT DEFINED PROJECT_BUILD_DIR OR NOT DEFINED CONSUMER_SOURCE_DIR)
  message(FATAL_ERROR "PROJECT_BUILD_DIR and CONSUMER_SOURCE_DIR are required")
endif()

set(package_dir "${PROJECT_BUILD_DIR}/install-consumer-test")
file(REMOVE_RECURSE "${package_dir}")
set(prefix "${package_dir}/prefix")
set(consumer_build "${package_dir}/build")

function(run_checked)
  execute_process(COMMAND ${ARGV} RESULT_VARIABLE status
                  OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "Command failed (${status}): ${ARGV}\n${output}\n${error}")
  endif()
endfunction()

run_checked("${CMAKE_COMMAND}" --install "${PROJECT_BUILD_DIR}" --prefix "${prefix}" --config "${CONFIG}")
run_checked("${CMAKE_COMMAND}" -S "${CONSUMER_SOURCE_DIR}" -B "${consumer_build}"
            "-DCMAKE_PREFIX_PATH=${prefix}" "-DCMAKE_CXX_COMPILER=${CONSUMER_CXX_COMPILER}"
            "-DCMAKE_CXX_FLAGS=${CONSUMER_CXX_FLAGS}" "-DCMAKE_EXE_LINKER_FLAGS=${CONSUMER_LINKER_FLAGS}")
run_checked("${CMAKE_COMMAND}" --build "${consumer_build}" --config "${CONFIG}")
run_checked("${CMAKE_CTEST_COMMAND}" --test-dir "${consumer_build}" -C "${CONFIG}" --output-on-failure)
