# FHN Backend Architecture Design

## Date: 2026-03-31

## Motivation

Fhenomenon's Backend layer needs a standardized interface that:

1. Allows **FHE library developers** (cryptographers, hardware engineers) to integrate their libraries with minimal CS infrastructure knowledge
2. Lets **fhenomenon maintainers** handle all complexity of AST construction, optimization, lowering, and dispatch
3. Keeps **application developers** completely abstracted from FHE internals via `Compuon<T>`

The current Backend interface (`add()`, `multiply()`, `decrypt()` as individual virtual methods) tightly couples the scheduler to backend dispatch. The new design draws from compute acceleration APIs (OpenCL, SYCL) but replaces their program-compilation model with a **flat kernel table** that FHE library developers simply populate.

### Design Target

Cheddar-FHE (GPU-accelerated CKKS library at `refs/cheddar-fhe/`) serves as the first integration target alongside the existing ToyFHE reference backend.

---

## Architecture Overview

```
User code (Compuon<T> operators)
    |
Session -> Scheduler -> AST construction -> ASTPass optimization
    |
LowerToFhnProgram (AST -> FhnProgram)
    |
FhnDefaultExecutor
    |-- kernel table lookup by opcode
    |-- missing? decompose (fused -> primitives)
    |-- found? single function call dispatch
    |
FHE Library Kernel Functions
    (cheddar: Context::Add, Context::HMult, ...)
    (ToyFHE:  Engine::add, Engine::multiply, ...)
```

### Responsibility Matrix

| Role | Responsibility | Interface |
|------|---------------|-----------|
| **Application developer** | Write `a + b`, `a * b` | `Compuon<T>` |
| **Fhenomenon maintainer** | AST->IR lowering, decompose, dispatch, async | `FhnProgram`, `FhnDefaultExecutor` |
| **FHE library developer** | Implement kernel functions, register in table | `fhn_get_info`, `fhn_create`, `fhn_destroy`, `fhn_get_kernels` |

---

## 1. FHN IR Definition

The IR is a flat C ABI representation. No complex graph structures, no pointer chasing. A topologically sorted array of instructions that any backend can iterate linearly.

### 1.1 Opcode Catalog

A wide, shallow catalog of operations. Each opcode represents a **single task** completable in one function call. Fused opcodes (HMULT, HROT) are included; backends that don't support them leave the slot NULL and fhenomenon decomposes automatically.

```c
// fhn_program.h

#ifndef FHN_PROGRAM_H
#define FHN_PROGRAM_H

#include <stdint.h>

typedef enum FhnOpCode {
    FHN_NOP = 0,

    // Data lifecycle
    FHN_ENCODE,              // message -> plaintext
    FHN_ENCRYPT,             // plaintext -> ciphertext
    FHN_DECRYPT,             // ciphertext -> plaintext
    FHN_DECODE,              // plaintext -> message

    // Arithmetic (same-level, same-scale operands)
    FHN_ADD_CC,              // ciphertext + ciphertext
    FHN_ADD_CP,              // ciphertext + plaintext
    FHN_ADD_CS,              // ciphertext + scalar
    FHN_SUB_CC,              // ciphertext - ciphertext
    FHN_SUB_CP,              // ciphertext - plaintext
    FHN_SUB_SC,              // scalar - ciphertext
    FHN_NEGATE,              // -ciphertext
    FHN_MULT_CC,             // ciphertext * ciphertext (tensor only, no relin)
    FHN_MULT_CP,             // ciphertext * plaintext
    FHN_MULT_CS,             // ciphertext * scalar

    // Key-switching operations
    FHN_RELINEARIZE,         // 3-poly -> 2-poly ciphertext
    FHN_RESCALE,             // reduce level by 1
    FHN_ROTATE,              // params[0] = rotation distance
    FHN_CONJUGATE,           // complex conjugation
    FHN_MULT_KEY,            // multiply by evaluation key

    // Level management
    FHN_MOD_DOWN,            // modulus reduction
    FHN_LEVEL_DOWN,          // params[0] = target level

    // Fused composites
    // Backend may support these natively for better performance.
    // If unsupported, fhenomenon decomposes into primitives.
    FHN_HMULT,               // mult + relin + rescale
    FHN_HROT,                // key_switch + permute
    FHN_HROT_ADD,            // rotate(a) + b
    FHN_HCONJ_ADD,           // conj(a) + b
    FHN_MAD,                 // res += a * scalar

    // Boolean / comparison (TFHE-style schemes)
    FHN_AND,
    FHN_OR,
    FHN_XOR,
    FHN_EQ,
    FHN_LT,
    FHN_LE,

    FHN_OPCODE_COUNT         // sentinel
} FhnOpCode;
```

### 1.2 Instruction and Program

```c
typedef struct FhnInstruction {
    FhnOpCode   opcode;
    uint32_t    result_id;       // SSA-style unique ID for this result
    uint32_t    operands[4];     // input result_ids (0 = unused)
    int64_t     params[4];       // opcode-specific integers (rot distance, level, ...)
    double      fparams[2];      // opcode-specific floats (scale, ...)
} FhnInstruction;

typedef struct FhnProgram {
    uint32_t          version;           // IR version for forward compatibility
    uint32_t          num_instructions;
    FhnInstruction   *instructions;      // topologically sorted

    uint32_t          num_inputs;
    uint32_t         *input_ids;         // which result_ids are external inputs
    uint32_t          num_outputs;
    uint32_t         *output_ids;        // which result_ids are final outputs
} FhnProgram;

#endif // FHN_PROGRAM_H
```

**Key design decisions:**
- **Topological order guaranteed**: backends iterate linearly, no graph traversal needed
- **SSA-style result_id**: each instruction produces a unique result, operands reference prior results
- **Fused opcodes in catalog**: backends that support them execute directly; others get automatic decomposition
- **`version` field**: forward compatibility as opcode catalog evolves

---

## 2. Backend Provider Interface

### 2.1 What FHE Library Developers Export

A backend shared library (e.g., `libcheddar_fhn.so`) exports exactly **4 functions**:

```c
// fhn_backend_api.h

#ifndef FHN_BACKEND_API_H
#define FHN_BACKEND_API_H

#include "fhn_program.h"

// Opaque handles
typedef struct FhnBackendCtx  FhnBackendCtx;
typedef struct FhnBuffer      FhnBuffer;

// Device information
typedef struct FhnBackendInfo {
    const char *name;            // "cheddar-ckks", "seal-bfv", ...
    const char *version;
    uint32_t    device_type;     // 0=CPU, 1=GPU, 2=FPGA, 3=ASIC
    uint64_t    device_memory;   // bytes, 0 if N/A
} FhnBackendInfo;

// Kernel function signature
// All kernel functions have this uniform signature.
typedef int (*FhnKernelFn)(FhnBackendCtx *ctx,
                           FhnBuffer *result,
                           const FhnBuffer *const *operands,
                           const int64_t *params,
                           const double *fparams);

// One entry in the kernel table
typedef struct FhnKernelEntry {
    FhnOpCode    opcode;
    FhnKernelFn  fn;             // NULL = not supported
    const char  *name;           // for debugging/logging
} FhnKernelEntry;

// The kernel table
typedef struct FhnKernelTable {
    uint32_t          num_kernels;
    FhnKernelEntry   *kernels;
} FhnKernelTable;

// === Required exports (4 functions) ===

// Backend lifecycle
FhnBackendInfo  *fhn_get_info(void);
FhnBackendCtx   *fhn_create(const char *config_json);
void             fhn_destroy(FhnBackendCtx *ctx);

// Kernel registration
FhnKernelTable  *fhn_get_kernels(FhnBackendCtx *ctx);

#endif // FHN_BACKEND_API_H
```

### 2.2 Optional Advanced Override

Backends that want full control over program execution (e.g., GPU stream pipelining, cross-instruction fusion) can optionally export:

```c
// Optional: override default executor
// If exported, fhenomenon calls this instead of iterating the kernel table.
typedef struct FhnExecHandle FhnExecHandle;

FhnExecHandle  *fhn_submit(FhnBackendCtx *ctx,
                           const FhnProgram *program,
                           FhnBuffer **inputs,
                           uint32_t num_inputs);
int             fhn_poll(FhnExecHandle *handle);    // non-blocking
int             fhn_wait(FhnExecHandle *handle);    // blocking
FhnBuffer     **fhn_get_outputs(FhnExecHandle *handle,
                                uint32_t *num_outputs);
void            fhn_exec_free(FhnExecHandle *handle);
```

Most backends will NOT implement these. The `FhnDefaultExecutor` inside fhenomenon handles everything using the kernel table.

---

## 3. FhnDefaultExecutor (Fhenomenon Internal)

This is the engine that FHE library developers never need to see. It consumes `FhnProgram` and dispatches to kernel functions from the table.

```cpp
class FhnDefaultExecutor {
    std::unordered_map<FhnOpCode, FhnKernelFn> dispatch_;

public:
    explicit FhnDefaultExecutor(FhnKernelTable *table) {
        for (uint32_t i = 0; i < table->num_kernels; i++)
            dispatch_[table->kernels[i].opcode] = table->kernels[i].fn;
    }

    bool supports(FhnOpCode op) const {
        return dispatch_.count(op) > 0;
    }

    int execute(FhnBackendCtx *ctx, const FhnProgram *program,
                FhnBuffer **buffers) {
        for (uint32_t i = 0; i < program->num_instructions; i++) {
            const auto &inst = program->instructions[i];

            auto it = dispatch_.find(inst.opcode);
            if (it == dispatch_.end()) {
                if (!decompose(ctx, inst, program, buffers))
                    return -1;
                continue;
            }

            const FhnBuffer *ops[] = {
                buffers[inst.operands[0]],
                buffers[inst.operands[1]],
                buffers[inst.operands[2]],
                buffers[inst.operands[3]],
            };

            int err = it->second(ctx, buffers[inst.result_id],
                                 ops, inst.params, inst.fparams);
            if (err) return err;
        }
        return 0;
    }

private:
    bool decompose(FhnBackendCtx *ctx, const FhnInstruction &inst,
                   const FhnProgram *prog, FhnBuffer **buffers);
};
```

### 3.1 Decomposition Rules

When a backend does not support a fused opcode, fhenomenon decomposes it into primitives:

| Fused Opcode | Decomposition |
|-------------|--------------|
| `FHN_HMULT` | `FHN_MULT_CC` + `FHN_RELINEARIZE` + `FHN_RESCALE` |
| `FHN_HROT` | `FHN_MULT_KEY` + `FHN_ROTATE` |
| `FHN_HROT_ADD` | `FHN_HROT` + `FHN_ADD_CC` |
| `FHN_HCONJ_ADD` | `FHN_CONJUGATE` (via key) + `FHN_ADD_CC` |
| `FHN_MAD` | `FHN_MULT_CS` + `FHN_ADD_CC` |

Conversely, when lowering the AST, fhenomenon may **fuse** consecutive primitives into a fused opcode if the backend supports it (peephole optimization in `LowerToFhnProgram` or as an `ASTPass`).

---

## 4. Scheduler Integration: AST to FhnProgram Lowering

### 4.1 New Pipeline

```
Current:
  Operations -> AST -> evaluate() -> Backend::add/multiply (direct call)

New:
  Operations -> AST -> ASTPass optimization -> LowerToFhnProgram -> FhnProgram
                                                                       |
                                                            FhnDefaultExecutor
                                                                       |
                                                            kernel table dispatch
```

### 4.2 LowerToFhnProgram Pass

Post-order traversal of the AST produces a topologically sorted instruction array:

```cpp
class LowerToFhnProgram : public ASTPass {
public:
    std::string name() const override { return "LowerToFhnProgram"; }

    FhnProgram lower(const Planner<int> &plan) const {
        FhnProgram program{};
        program.version = 1;
        uint32_t next_id = 1;
        std::unordered_map<ASTNode*, uint32_t> node_to_id;

        for (auto &root : plan.getRoots())
            lowerNode(root.get(), program, next_id, node_to_id);

        return program;
    }

private:
    void lowerNode(ASTNode *node, FhnProgram &prog,
                   uint32_t &next_id,
                   std::unordered_map<ASTNode*, uint32_t> &ids) const {
        if (ids.count(node)) return;

        if (auto *op = dynamic_cast<OperatorNode<int>*>(node)) {
            if (op->left())  lowerNode(op->left().get(),  prog, next_id, ids);
            if (op->right()) lowerNode(op->right().get(), prog, next_id, ids);

            FhnInstruction inst{};
            inst.result_id = next_id++;
            inst.opcode = mapOpType(op->getOperation()->getType());
            if (op->left())  inst.operands[0] = ids[op->left().get()];
            if (op->right()) inst.operands[1] = ids[op->right().get()];

            appendInstruction(prog, inst);
            ids[node] = inst.result_id;
        } else if (auto *leaf = dynamic_cast<OperandNode<int>*>(node)) {
            uint32_t id = next_id++;
            ids[node] = id;
            appendInput(prog, id);
        }
    }

    FhnOpCode mapOpType(OperationType t) const {
        switch (t) {
        case OperationType::Add:      return FHN_ADD_CC;
        case OperationType::Multiply: return FHN_HMULT;
        default:                      return FHN_NOP;
        }
    }
};
```

### 4.3 Scheduler Dispatch

```cpp
void Scheduler::evaluateGraph(Planner<int> &plan) {
    LowerToFhnProgram lowering;
    FhnProgram program = lowering.lower(plan);

    FhnBuffer *inputs[program.num_inputs];
    prepareInputBuffers(plan, program, inputs);

    // If backend provides fhn_submit, use it (advanced path)
    // Otherwise, use FhnDefaultExecutor with kernel table
    executor_.execute(backend_ctx_, &program, buffers);

    collectResults(plan, program, buffers);
}
```

### 4.4 Coexistence with Current Interface

- **Immediate mode** (outside Session): current `Backend::add()` / `Backend::multiply()` path continues to work. Internally these can be thin wrappers that create a single-instruction FhnProgram and execute it.
- **Session mode**: new pipeline (AST -> FhnProgram -> executor)
- Gradual migration: the old interface becomes a convenience layer over the new one.

---

## 5. Cheddar-FHE Integration Example

### 5.1 What the Cheddar Developer Implements

File: `libcheddar_fhn.so` (separate build, links against cheddar)

```c
// cheddar_fhn_backend.cpp

#include "fhn_backend_api.h"
#include "core/Context.h"
#include "UserInterface.h"

typedef struct FhnBackendCtx {
    cheddar::ContextPtr<uint32_t>     context;
    cheddar::UserInterface<uint32_t> *ui;
    cheddar::Parameter<uint32_t>     *param;
} FhnBackendCtx;

// --- Kernel implementations (each is a single function call) ---

static int kernel_add_cc(FhnBackendCtx *ctx, FhnBuffer *res,
                         const FhnBuffer *const *ops,
                         const int64_t *params, const double *fp) {
    auto &r = *(cheddar::Ciphertext<uint32_t> *)res;
    auto &a = *(cheddar::Ciphertext<uint32_t> *)ops[0];
    auto &b = *(cheddar::Ciphertext<uint32_t> *)ops[1];
    ctx->context->Add(r, a, b);
    return 0;
}

static int kernel_hmult(FhnBackendCtx *ctx, FhnBuffer *res,
                        const FhnBuffer *const *ops,
                        const int64_t *params, const double *fp) {
    auto &r = *(cheddar::Ciphertext<uint32_t> *)res;
    auto &a = *(cheddar::Ciphertext<uint32_t> *)ops[0];
    auto &b = *(cheddar::Ciphertext<uint32_t> *)ops[1];
    ctx->context->HMult(r, a, b, ctx->ui->GetMultiplicationKey());
    return 0;
}

static int kernel_rotate(FhnBackendCtx *ctx, FhnBuffer *res,
                         const FhnBuffer *const *ops,
                         const int64_t *params, const double *fp) {
    int dist = (int)params[0];
    auto &r = *(cheddar::Ciphertext<uint32_t> *)res;
    auto &a = *(cheddar::Ciphertext<uint32_t> *)ops[0];
    ctx->context->HRot(r, a, ctx->ui->GetRotationKey(dist), dist);
    return 0;
}

static int kernel_rescale(FhnBackendCtx *ctx, FhnBuffer *res,
                          const FhnBuffer *const *ops,
                          const int64_t *params, const double *fp) {
    auto &r = *(cheddar::Ciphertext<uint32_t> *)res;
    auto &a = *(cheddar::Ciphertext<uint32_t> *)ops[0];
    ctx->context->Rescale(r, a);
    return 0;
}

// ... more kernels for SUB_CC, NEGATE, CONJUGATE, etc.

// --- Kernel table ---

static FhnKernelEntry cheddar_kernels[] = {
    { FHN_ADD_CC,    kernel_add_cc,    "add_cc"    },
    { FHN_SUB_CC,    kernel_sub_cc,    "sub_cc"    },
    { FHN_NEGATE,    kernel_negate,    "negate"    },
    { FHN_MULT_CC,   kernel_mult_cc,   "mult_cc"   },
    { FHN_MULT_CP,   kernel_mult_cp,   "mult_cp"   },
    { FHN_HMULT,     kernel_hmult,     "hmult"     },
    { FHN_ROTATE,    kernel_rotate,    "rotate"    },
    { FHN_CONJUGATE, kernel_conjugate, "conjugate" },
    { FHN_RESCALE,   kernel_rescale,   "rescale"   },
    { FHN_HROT_ADD,  kernel_hrot_add,  "hrot_add"  },
    { FHN_HCONJ_ADD, kernel_hconj_add, "hconj_add" },
    // FHN_AND, FHN_OR, etc. not registered (CKKS does not support boolean)
};

// --- 4 required exports ---

FhnBackendInfo *fhn_get_info(void) {
    static FhnBackendInfo info = {
        .name = "cheddar-ckks",
        .version = "1.0",
        .device_type = 1,  // GPU
        .device_memory = 0 // detected at runtime in fhn_create
    };
    return &info;
}

FhnBackendCtx *fhn_create(const char *config_json) {
    // Parse config, create cheddar Parameter, Context, UserInterface
    // ...
    return ctx;
}

void fhn_destroy(FhnBackendCtx *ctx) {
    delete ctx->ui;
    delete ctx->param;
    delete ctx;
}

FhnKernelTable *fhn_get_kernels(FhnBackendCtx *ctx) {
    static FhnKernelTable table = {
        .num_kernels = sizeof(cheddar_kernels) / sizeof(cheddar_kernels[0]),
        .kernels = cheddar_kernels
    };
    return &table;
}
```

### 5.2 What Fhenomenon Does Automatically

Given the kernel table above, fhenomenon:

1. Knows cheddar supports `FHN_HMULT` natively, so it emits `FHN_HMULT` directly (no decomposition)
2. Knows cheddar does NOT support `FHN_AND` / `FHN_OR`, so it rejects boolean programs or routes to a different backend
3. Calls `kernel_rotate` with `params[0]` set to the rotation distance from the AST
4. Manages buffer allocation, input/output marshalling, async execution

### 5.3 Advanced: Cheddar GPU Stream Override (Optional)

If cheddar wants to pipeline multiple kernel launches on a CUDA stream instead of synchronizing between each kernel call:

```c
// Optional export - overrides FhnDefaultExecutor
FhnExecHandle *fhn_submit(FhnBackendCtx *ctx,
                          const FhnProgram *program,
                          FhnBuffer **inputs, uint32_t num_inputs) {
    // Create CUDA stream
    // Iterate program->instructions, launch kernels on stream
    // Return handle wrapping the stream
}

int fhn_poll(FhnExecHandle *handle) {
    // cudaStreamQuery
}

int fhn_wait(FhnExecHandle *handle) {
    // cudaStreamSynchronize
}
```

---

## 6. BuiltinBackend as Reference Implementation

The existing ToyFHE-based BuiltinBackend is refactored to use the same kernel table pattern. This serves as the template other developers copy.

```cpp
// ToyFHE kernel implementations
static int toyfhe_add_cc(FhnBackendCtx *ctx, FhnBuffer *res,
                         const FhnBuffer *const *ops,
                         const int64_t *params, const double *fp) {
    auto *e = (ToyFheCtx *)ctx;
    auto &a = *(toyfhe::Ciphertext *)ops[0];
    auto &b = *(toyfhe::Ciphertext *)ops[1];
    *(toyfhe::Ciphertext *)res = e->engine.add(a, b);
    return 0;
}

static int toyfhe_hmult(FhnBackendCtx *ctx, FhnBuffer *res,
                        const FhnBuffer *const *ops,
                        const int64_t *params, const double *fp) {
    auto *e = (ToyFheCtx *)ctx;
    auto &a = *(toyfhe::Ciphertext *)ops[0];
    auto &b = *(toyfhe::Ciphertext *)ops[1];
    // ToyFHE does implicit relin in multiply
    *(toyfhe::Ciphertext *)res = e->engine.multiply(a, b);
    return 0;
}

static FhnKernelEntry toyfhe_kernels[] = {
    { FHN_ENCODE,    toyfhe_encode,    "encode"    },
    { FHN_ENCRYPT,   toyfhe_encrypt,   "encrypt"   },
    { FHN_DECRYPT,   toyfhe_decrypt,   "decrypt"   },
    { FHN_DECODE,    toyfhe_decode,    "decode"    },
    { FHN_ADD_CC,    toyfhe_add_cc,    "add_cc"    },
    { FHN_ADD_CS,    toyfhe_add_cs,    "add_cs"    },
    { FHN_SUB_CC,    toyfhe_sub_cc,    "sub_cc"    },
    { FHN_NEGATE,    toyfhe_negate,    "negate"    },
    { FHN_MULT_CC,   toyfhe_mult_cc,   "mult_cc"   },
    { FHN_MULT_CS,   toyfhe_mult_cs,   "mult_cs"   },
    { FHN_HMULT,     toyfhe_hmult,     "hmult"     },
    // No FHN_ROTATE, FHN_CONJUGATE (ToyFHE doesn't support)
    // No FHN_AND/OR/XOR (not a boolean scheme)
};

FhnBackendInfo *fhn_get_info(void) {
    static FhnBackendInfo info = {
        .name = "toyfhe-reference",
        .version = "0.1",
        .device_type = 0,  // CPU
        .device_memory = 0
    };
    return &info;
}
```

**This is the code an FHE library developer reads, copies, and adapts.**

---

## 7. Migration Path

### Phase 1: Infrastructure
- Define `fhn_program.h` and `fhn_backend_api.h` C headers
- Implement `FhnDefaultExecutor`
- Refactor `BuiltinBackend` to use kernel table (reference implementation)
- Keep existing `Backend::add/multiply` as thin wrappers for immediate mode

### Phase 2: Scheduler Integration
- Implement `LowerToFhnProgram` ASTPass
- Wire `Scheduler::evaluateGraph` to use `FhnDefaultExecutor`
- Implement basic decomposition rules

### Phase 3: Cheddar Integration
- Build cheddar-fhe as shared library
- Implement `libcheddar_fhn.so` with kernel table
- Implement `ExternalBackend` class that `dlopen`s backend .so and resolves the 4 required symbols
- Test end-to-end: `Compuon<double>` -> Session -> AST -> FhnProgram -> cheddar GPU execution

### Phase 4: Optimization
- Peephole fusion in lowering (consecutive primitives -> fused opcodes when backend supports them)
- Cheddar optional `fhn_submit` override for GPU stream pipelining
- Async execution integration with Session

---

## 8. Key Design Decisions

1. **C ABI over C++ virtual methods**: enables `dlopen`-based plugin loading, language-agnostic backend implementation
2. **Kernel table over program parsing**: FHE developers implement flat functions, never parse IR
3. **Fused opcodes in catalog**: backends opt-in to fused operations for performance; fhenomenon decomposes automatically if unsupported
4. **`fhn_` prefix**: short, distinctive, avoids collision with `fhe_` (too generic)
5. **Optional `fhn_submit` override**: escape hatch for GPU backends that need full program control, but most backends just fill the kernel table
6. **Topologically sorted instruction array**: simplest possible IR — iterate and execute, no graph algorithms needed
