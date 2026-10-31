# Circuit execution cross-task contract
Root owns public FHN circuit header, corpus importer/runner, benchmark harness, shared CMake/tests registration and docs. Compiler agent owns src/CKKS/backend/* (including Backend.cpp extraction of internal context) and test/CKKS/CircuitTest.cpp. Crypto agent owns src/CKKS/eval/* and test/CKKS/RegionTest.cpp (+existing EvalTest if needed). Workload agent will own independent corpus import tests/benchmark test support after root defines APIs. Do not edit another owner's files.

## Public C circuit ABI (root publishes include/FHN/fhn_circuit_api.h)
FHN_CIRCUIT_VERSION=1.
FhnCircuitOp: FHN_CIRCUIT_ADD, SUBTRACT, MULTIPLY, SCALE, ADD_CONSTANT, NEGATE, ROTATE, CONJUGATE.
FhnCircuitNode fields: FhnCircuitOp op; uint32_t result_id; uint32_t operands[2]; int64_t rotation; double scalar.
FhnCircuit fields: uint32_t version,num_nodes; const FhnCircuitNode* nodes; uint32_t num_inputs; const uint32_t* input_ids; uint32_t num_outputs; const uint32_t* output_ids.
FhnExecutable opaque.
FhnCompileOptions fields: uint32_t struct_size; uint64_t workspace_limit_bytes; (0 means no explicit limit, counts internal slot + scratch reserve).
FhnExecutableStats fields: uint32_t struct_size; uint32_t source_nodes,regions,linear_regions,product_regions,rotation_regions,storage_slots; uint64_t storage_bytes,scratch_bytes;
Function typedefs plus native declarations in backend/include/CKKS/Circuit.h:
FhnExecutable* ckks_fhn_compile_circuit(FhnBackendCtx*,const FhnCircuit*,const FhnBuffer*const* prototype_inputs,uint32_t num_inputs,const FhnCompileOptions*,char* error,uint32_t error_capacity);
int ckks_fhn_client_prepare_executable(FhnClientCtx*,FhnExecutable*);
int ckks_fhn_run_executable(FhnExecutable*,const FhnBuffer*const* inputs,uint32_t num_inputs,FhnBuffer*const* outputs,uint32_t num_outputs);
int ckks_fhn_executable_stats(const FhnExecutable*,FhnExecutableStats*);
void ckks_fhn_executable_free(FhnExecutable*);
Compile copies graph/signatures, may run before evaluation keys exist. prepare_executable derives only compiled requirements through trusted client, validates/binds public keys once. Run leases weak keys once; no secret/client pointer in executable. Input/output buffers must outlive run, executable before ctx. Caller serializes same instance/context. Compiled node sequence not tied to old opcode table.

## L4 private region execution API
Header src/CKKS/eval/include/CKKS/detail/Region.h, namespace fhenomenon::ckks::region. Internal to native compiler, not stable external ABI. Caller establishes invariants; no repeated coefficient/key scans per region. Source/destination storage disjoint for these kernels. All use caller Arena and never allocate heap. Final public-output transaction is compiler's responsibility. Kernels may still reject malformed cheap metadata/scratch, but must not rescan immutable keys or inputs already certified.
struct LinearTerm { ConstCiphertextView value; int64_t coefficient; };
struct ProductTerm { ConstCiphertextView left; ConstCiphertextView right; int64_t coefficient; };
bool validateInput(ConstCiphertextView,const EvaluationTables&); // full boundary scan
bool validateKey(EvaluationKeyView,SwitchKeyKind,uint64_t galois,const EvaluationTables&); // full boundary scan
size_t scratchBytes(const Params&) noexcept; // conservative max of all region ops; excludes caller output/keys/tables
bool linear(CiphertextView& out,const LinearTerm* terms,size_t count,double constant,CiphertextLayout target,const EvaluationTables&,Arena&);
// terms all same Qprefix/2components/target scale; integer coefficient modular arithmetic; constant encode at target.scale (int64 guard).
bool productSum(CiphertextView& out,const ProductTerm* products,size_t product_count,const LinearTerm* linear_terms,size_t linear_count,double constant,EvaluationKeyView,const EvaluationTables&,Arena&);
// Product operands/linear terms all same basis+scale,2components. Weighted tensor sum + linear lifted by integer common scale -> one relin/rescale. linear_count>0 or constant!=0 requires integral common scale in signed64 range. Pure products support arbitrary positive finite common scale. Output level one lower, scale S*S/q. At least1product, counts public; coefficients signed64. Out disjoint all input/key slabs.
bool multiply(CiphertextView&,ConstCiphertextView,ConstCiphertextView,EvaluationKeyView,const EvaluationTables&,Arena&);
// same basis2polys; scales may differ; real fused tensor/relin/rescale; out level-1,scalea*scaleb/q.
bool rotateAdd(CiphertextView&,ConstCiphertextView,uint64_t galois,EvaluationKeyView,const LinearTerm* addends,size_t count,const EvaluationTables&,Arena&);
// Rotated input plus exact integer linear addends of same basis/scale. count0 pure rotate. Identity g=1 no key. output2polys same level/scale.
bool scaleFractional(CiphertextView&,ConstCiphertextView,double,const EvaluationTables&,Arena&); // same semantics existing multiplyScalar consumes1.
bool levelDown(CiphertextView&,ConstCiphertextView,uint32_t,const EvaluationTables&,Arena&); // copies active Q prefix, same scale.
Required helper for compiler validation/planning: bool encodeConstant(double value,double scale,int64_t &encoded) noexcept; matches region linear quantization and no writes. region::linear count0 uses supplied valid target as constant-only result (c0constant,c1zero).

Crypto agent may add private helpers to Eval.cpp, EvalTables combined CRT accessors, or Region.cpp as needed. Exact Pq fusion changes checked eval::hmult implementation too (keep primitive tensor/relin/rescale for differential oracle). Preserve legacy checked multiplyScalar level consumption; exact divisibility fast path allowed. Compiler uses new integer-preserving region linear for semantic SCALE. Full source-digit NTT preservation permitted; no full QP Galois hoist without proof.

Compiler lowerings: collect bounded affine forms across Add/Sub/Negate/IntegerScale/AddConstant; materialize at nonlinear/rotation boundaries. Fuse single-use rotate into affine consumer when terms allow. Recognize affine sums of same-level/same-scale products plus same-signature linear terms as productSum (wide-front), preventing 24 switches and avoiding wrong-scale join. Don't silently change scale metadata. For stats, normal semantic products plus integer64 preserve scale between variance branches. Reject unsupported scale joins with concrete error. Inputs may be same physical handle; output/input alias allowed via late publication. Distinct output IDs require distinct output destinations. Compile budget bound and diagnostics before exposing executable. Plan stores copied scalar params and graph, metadata signatures, and reusable live interval slots; run performs no allocation.
