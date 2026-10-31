# Verify real CLI execution and its exit status, including explicit skips.
function(expect_run expected_status expected_text)
  execute_process(COMMAND "${CORPUS}" ${ARGN}
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status EQUAL expected_status OR NOT "${output}${error}" MATCHES "${expected_text}")
    message(FATAL_ERROR "Unexpected corpus result (${status}): ${ARGN}\n${output}\n${error}")
  endif()
endfunction()

expect_run(0 "planned=1 executed=1 skipped=0 failed=0"
  --backend "${HROT_BACKEND}" --prefix alt_ --shape matvec-fused --require-execution)
expect_run(1 "native fused kernel or complete fallback unavailable"
  --backend "${HROT_BACKEND}" --prefix alt_ --shape matvec-fused --require-execution --compare-fallback)
expect_run(0 "planned=1 executed=1 skipped=0 failed=0"
  --backend "${TOY_BACKEND}" --shape matvec-fused --require-execution --compare-fallback --budget-bytes min)
expect_run(0 "native and primitive fallback agree"
  --backend "${TOY_BACKEND}" --shape matvec-fused --require-execution --compare-fallback)
expect_run(0 "verified all 64 slots"
  --backend "${TOY_BACKEND}" --shape matvec --require-execution)
expect_run(0 "planned=1 executed=1 skipped=0 failed=0"
  --backend "${TOY_BACKEND}" --shape weighted-sum --require-execution)
expect_run(0 "planned=1 executed=1 skipped=0 failed=0"
  --backend "${TOY_BACKEND}" --shape wide-front --require-execution)
# These packed outputs exceed ToyFHE's fixed-point range. A verifier must
# report the numerical mismatch, never silently count it as a passed run.
expect_run(1 "planned=1 executed=0 skipped=0 failed=1"
  --backend "${TOY_BACKEND}" --shape stats --require-execution)
expect_run(1 "planned=1 executed=0 skipped=0 failed=1"
  --backend "${TOY_BACKEND}" --shape logreg --require-execution)
expect_run(0 "planned=1 executed=0 skipped=1 failed=0" --shape matvec)
expect_run(1 "required execution was incomplete" --shape matvec --require-execution)
expect_run(1 "exceeds --max-depth"
  --backend "${TOY_BACKEND}" --shape matvec-fused --max-depth 0 --require-execution)
expect_run(1 "backend lacks the required packed/scalar data-plane pair"
  --backend "${SCALAR_BACKEND}" --prefix ptl_ --shape matvec --require-execution)
expect_run(1 "shape contains no fused operation"
  --backend "${TOY_BACKEND}" --shape matvec --compare-fallback --require-execution)
expect_run(2 "usage:" --max-depth invalid)
expect_run(2 "usage:" --budget-bytes -1)
expect_run(2 "usage:" --backend "${TOY_BACKEND}" --shape matvec --budget-bytes " -1")
expect_run(2 "usage:" --backend "${TOY_BACKEND}" --shape matvec --budget-bytes "\t-1")
expect_run(2 "--list cannot require execution" --list --require-execution)
expect_run(2 "usage:" --atol nan)
