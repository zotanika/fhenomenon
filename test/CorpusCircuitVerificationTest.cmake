# Exercise native encrypted application results and truthful coverage. This
# uses one verification repetition, not a performance comparison or speed gate.
function(expect_circuit_run expected_status expected_text)
  execute_process(
    COMMAND "${CIRCUIT_BENCH}" --backend "${CKKS_BACKEND}" --prefix ckks_
      --warmup 0 --reps 1 ${ARGN}
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status EQUAL expected_status OR NOT "${output}${error}" MATCHES "${expected_text}")
    message(FATAL_ERROR "Unexpected native circuit result (${status}): ${ARGN}\n${output}\n${error}")
  endif()
  set(circuit_output "${output}${error}" PARENT_SCOPE)
endfunction()

# Run expensive packed applications once, checking individual outcomes as well
# as the summary so a wrong set of ten apps cannot satisfy the coverage gate.
expect_circuit_run(0 "executed=10 unsupported=3 failed=0")
foreach(shape matvec-fused matvec matmul-tile reduce-tree weighted-sum diamond conv1d iter-update stats wide-front)
  if(NOT "${circuit_output}" MATCHES "(^|\n)PASS ${shape}( |\n|$)")
    message(FATAL_ERROR "Required application ${shape} did not pass:\n${circuit_output}")
  endif()
endforeach()
foreach(shape horner15 logreg psi-eq)
  if(NOT "${circuit_output}" MATCHES "(^|\n)UNSUPPORTED ${shape}: [^\n]+")
    message(FATAL_ERROR "Expected explicit unsupported reason for ${shape}:\n${circuit_output}")
  endif()
endforeach()

expect_circuit_run(0 "PASS weighted-sum" --shape weighted-sum --require-execution)
foreach(shape horner15 logreg psi-eq)
  expect_circuit_run(1 "UNSUPPORTED ${shape}: [^\n]+" --shape "${shape}" --require-execution)
endforeach()

expect_circuit_run(2 "[Uu]nknown shape" --shape nonexistent-circuit-shape)

# A C-string config boundary must reject embedded NUL rather than parsing only
# the valid JSON prefix and silently ignoring the rest of the requested config.
expect_circuit_run(2 "usage:" --shape diamond --config "${CMAKE_CURRENT_LIST_DIR}/config-with-null.json")
expect_circuit_run(2 "usage:" --reps 0)
expect_circuit_run(1 "UNSUPPORTED diamond:" --shape diamond --workspace-bytes 1)
