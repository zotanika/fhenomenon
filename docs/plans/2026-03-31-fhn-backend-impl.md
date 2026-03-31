# FHN Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the FHN compute-device-style backend interface with C ABI kernel table, refactor BuiltinBackend as reference implementation, and integrate Scheduler lowering pipeline.

**Architecture:** C ABI headers define the IR (FhnProgram/FhnInstruction) and backend provider interface (4 exported functions + kernel table). FhnDefaultExecutor dispatches instructions to kernel functions. BuiltinBackend (ToyFHE) serves as the reference implementation that other FHE library developers copy. The Scheduler gains a LowerToFhnProgram pass that converts the AST to a flat instruction array.

**Tech Stack:** C++17, C ABI (`extern "C"`), GoogleTest, CMake 3.19+

**Design Doc:** `docs/plans/2026-03-31-fhn-backend-design.md`

---

## Phase 1: C ABI Headers

### Task 1: Create fhn_program.h (IR Definition)

**Files:**
- Create: `include/FHN/fhn_program.h`
- Test: `test/FhnProgramTest.cpp`

**Step 1: Create the C ABI header**

```c
// include/FHN/fhn_program.h
#ifndef FHN_PROGRAM_H
#define FHN_PROGRAM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum FhnOpCode {
    FHN_NOP = 0,

    // Data lifecycle
    FHN_ENCODE,
    FHN_ENCRYPT,
    FHN_DECRYPT,
    FHN_DECODE,

    // Arithmetic (same-level, same-scale)
    FHN_ADD_CC,       // ciphertext + ciphertext
    FHN_ADD_CP,       // ciphertext + plaintext
    FHN_ADD_CS,       // ciphertext + scalar
    FHN_SUB_CC,
    FHN_SUB_CP,
    FHN_SUB_SC,       // scalar - ciphertext
    FHN_NEGATE,
    FHN_MULT_CC,      // ciphertext * ciphertext (tensor only)
    FHN_MULT_CP,      // ciphertext * plaintext
    FHN_MULT_CS,      // ciphertext * scalar

    // Key-switching operations
    FHN_RELINEARIZE,
    FHN_RESCALE,
    FHN_ROTATE,       // params[0] = rotation distance
    FHN_CONJUGATE,
    FHN_MULT_KEY,

    // Level management
    FHN_MOD_DOWN,
    FHN_LEVEL_DOWN,   // params[0] = target level

    // Fused composites (backend decomposes if unsupported)
    FHN_HMULT,        // mult + relin + rescale
    FHN_HROT,         // key_switch + permute
    FHN_HROT_ADD,     // rotate(a) + b
    FHN_HCONJ_ADD,    // conj(a) + b
    FHN_MAD,          // res += a * scalar

    // Boolean / comparison (TFHE-style)
    FHN_AND,
    FHN_OR,
    FHN_XOR,
    FHN_EQ,
    FHN_LT,
    FHN_LE,

    FHN_OPCODE_COUNT  // sentinel
} FhnOpCode;

typedef struct FhnInstruction {
    FhnOpCode   opcode;
    uint32_t    result_id;
    uint32_t    operands[4];   // 0 = unused
    int64_t     params[4];     // opcode-specific integers
    double      fparams[2];    // opcode-specific floats
} FhnInstruction;

typedef struct FhnProgram {
    uint32_t          version;
    uint32_t          num_instructions;
    FhnInstruction   *instructions;

    uint32_t          num_inputs;
    uint32_t         *input_ids;
    uint32_t          num_outputs;
    uint32_t         *output_ids;
} FhnProgram;

// Utility: allocate/free an FhnProgram (owned by caller)
FhnProgram *fhn_program_alloc(uint32_t num_instructions,
                              uint32_t num_inputs,
                              uint32_t num_outputs);
void        fhn_program_free(FhnProgram *program);

#ifdef __cplusplus
}
#endif

#endif // FHN_PROGRAM_H
```

**Step 2: Write the test**

```cpp
// test/FhnProgramTest.cpp
#include "FHN/fhn_program.h"
#include <gtest/gtest.h>

TEST(FhnProgram, AllocFree) {
    auto *prog = fhn_program_alloc(3, 2, 1);
    ASSERT_NE(prog, nullptr);
    EXPECT_EQ(prog->version, 1u);
    EXPECT_EQ(prog->num_instructions, 3u);
    EXPECT_EQ(prog->num_inputs, 2u);
    EXPECT_EQ(prog->num_outputs, 1u);
    EXPECT_NE(prog->instructions, nullptr);
    EXPECT_NE(prog->input_ids, nullptr);
    EXPECT_NE(prog->output_ids, nullptr);
    fhn_program_free(prog);
}

TEST(FhnProgram, InstructionFields) {
    auto *prog = fhn_program_alloc(1, 0, 0);
    auto &inst = prog->instructions[0];

    inst.opcode = FHN_ADD_CC;
    inst.result_id = 3;
    inst.operands[0] = 1;
    inst.operands[1] = 2;
    inst.params[0] = 42;
    inst.fparams[0] = 3.14;

    EXPECT_EQ(inst.opcode, FHN_ADD_CC);
    EXPECT_EQ(inst.result_id, 3u);
    EXPECT_EQ(inst.operands[0], 1u);
    EXPECT_EQ(inst.operands[1], 2u);
    EXPECT_EQ(inst.params[0], 42);
    EXPECT_DOUBLE_EQ(inst.fparams[0], 3.14);

    fhn_program_free(prog);
}

TEST(FhnProgram, BuildSimpleAddProgram) {
    // Program: result = input0 + input1
    auto *prog = fhn_program_alloc(1, 2, 1);
    prog->input_ids[0] = 1;
    prog->input_ids[1] = 2;
    prog->output_ids[0] = 3;

    prog->instructions[0] = FhnInstruction{
        .opcode = FHN_ADD_CC,
        .result_id = 3,
        .operands = {1, 2, 0, 0},
        .params = {0, 0, 0, 0},
        .fparams = {0.0, 0.0}
    };

    EXPECT_EQ(prog->instructions[0].opcode, FHN_ADD_CC);
    EXPECT_EQ(prog->instructions[0].result_id, 3u);

    fhn_program_free(prog);
}

TEST(FhnProgram, OpcodeCount) {
    // Sentinel value should be last
    EXPECT_GT(FHN_OPCODE_COUNT, 0);
    EXPECT_EQ(static_cast<int>(FHN_NOP), 0);
}
```

**Step 3: Create the implementation**

```cpp
// src/FHN/fhn_program.cpp
#include "FHN/fhn_program.h"
#include <cstdlib>
#include <cstring>

extern "C" {

FhnProgram *fhn_program_alloc(uint32_t num_instructions,
                              uint32_t num_inputs,
                              uint32_t num_outputs) {
    auto *prog = static_cast<FhnProgram *>(calloc(1, sizeof(FhnProgram)));
    if (!prog) return nullptr;

    prog->version = 1;
    prog->num_instructions = num_instructions;
    prog->num_inputs = num_inputs;
    prog->num_outputs = num_outputs;

    if (num_instructions > 0) {
        prog->instructions = static_cast<FhnInstruction *>(
            calloc(num_instructions, sizeof(FhnInstruction)));
    }
    if (num_inputs > 0) {
        prog->input_ids = static_cast<uint32_t *>(
            calloc(num_inputs, sizeof(uint32_t)));
    }
    if (num_outputs > 0) {
        prog->output_ids = static_cast<uint32_t *>(
            calloc(num_outputs, sizeof(uint32_t)));
    }
    return prog;
}

void fhn_program_free(FhnProgram *program) {
    if (!program) return;
    free(program->instructions);
    free(program->input_ids);
    free(program->output_ids);
    free(program);
}

} // extern "C"
```

**Step 4: Add to build system**

Add to `src/CMakeLists.txt`:
```cmake
add_subdirectory(FHN)
```

Create `src/FHN/CMakeLists.txt`:
```cmake
add_fhenomenon_sources()
```

Add to `test/CMakeLists.txt`:
```cmake
add_executable(FhnProgramTest FhnProgramTest.cpp)
target_link_libraries(FhnProgramTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(FhnProgramTest)
```

**Step 5: Build and run test**

Run: `cmake --build build -j8 && ctest --test-dir build -R FhnProgram -V`
Expected: 4 tests PASS

**Step 6: Commit**

```bash
git add include/FHN/fhn_program.h src/FHN/ test/FhnProgramTest.cpp \
        src/CMakeLists.txt test/CMakeLists.txt
git commit -m "feat: add FHN IR definition (fhn_program.h)"
```

---

### Task 2: Create fhn_backend_api.h (Backend Provider Interface)

**Files:**
- Create: `include/FHN/fhn_backend_api.h`
- Test: `test/FhnBackendApiTest.cpp`

**Step 1: Create the C ABI header**

```c
// include/FHN/fhn_backend_api.h
#ifndef FHN_BACKEND_API_H
#define FHN_BACKEND_API_H

#include "FHN/fhn_program.h"

#ifdef __cplusplus
extern "C" {
#endif

// ── Opaque handles ──────────────────────────────────
typedef struct FhnBackendCtx  FhnBackendCtx;
typedef struct FhnBuffer      FhnBuffer;
typedef struct FhnExecHandle  FhnExecHandle;

// ── Device information ──────────────────────────────
typedef enum FhnDeviceType {
    FHN_DEVICE_CPU  = 0,
    FHN_DEVICE_GPU  = 1,
    FHN_DEVICE_FPGA = 2,
    FHN_DEVICE_ASIC = 3,
} FhnDeviceType;

typedef struct FhnBackendInfo {
    const char   *name;
    const char   *version;
    FhnDeviceType device_type;
    uint64_t      device_memory;   // bytes, 0 if N/A
} FhnBackendInfo;

// ── Kernel function signature ───────────────────────
// Uniform signature for all kernel functions.
// Returns 0 on success, non-zero on error.
typedef int (*FhnKernelFn)(FhnBackendCtx *ctx,
                           FhnBuffer *result,
                           const FhnBuffer *const *operands,
                           const int64_t *params,
                           const double *fparams);

// ── Kernel table entry ──────────────────────────────
typedef struct FhnKernelEntry {
    FhnOpCode    opcode;
    FhnKernelFn  fn;      // NULL = not supported
    const char  *name;    // for debugging/logging
} FhnKernelEntry;

typedef struct FhnKernelTable {
    uint32_t          num_kernels;
    FhnKernelEntry   *kernels;
} FhnKernelTable;

// ── Required backend exports (4 functions) ──────────
// Every backend shared library MUST export these symbols.

// Return static backend information.
// FhnBackendInfo *fhn_get_info(void);

// Create a backend context with JSON configuration.
// FhnBackendCtx *fhn_create(const char *config_json);

// Destroy a backend context.
// void fhn_destroy(FhnBackendCtx *ctx);

// Return the kernel table for this backend.
// FhnKernelTable *fhn_get_kernels(FhnBackendCtx *ctx);

// ── Function pointer typedefs for dlopen ────────────
typedef FhnBackendInfo  *(*FhnGetInfoFn)(void);
typedef FhnBackendCtx   *(*FhnCreateFn)(const char *config_json);
typedef void             (*FhnDestroyFn)(FhnBackendCtx *ctx);
typedef FhnKernelTable  *(*FhnGetKernelsFn)(FhnBackendCtx *ctx);

// ── Optional advanced exports ───────────────────────
// Backends that want full program control (GPU stream pipelining, etc.)
// can optionally export these. If absent, FhnDefaultExecutor is used.
typedef FhnExecHandle  *(*FhnSubmitFn)(FhnBackendCtx *ctx,
                                       const FhnProgram *program,
                                       FhnBuffer **inputs,
                                       uint32_t num_inputs);
typedef int             (*FhnPollFn)(FhnExecHandle *handle);
typedef int             (*FhnWaitFn)(FhnExecHandle *handle);
typedef FhnBuffer     **(*FhnGetOutputsFn)(FhnExecHandle *handle,
                                           uint32_t *num_outputs);
typedef void            (*FhnExecFreeFn)(FhnExecHandle *handle);

// ── Backend handle (used by ExternalBackend loader) ─
typedef struct FhnBackendVTable {
    FhnGetInfoFn     get_info;
    FhnCreateFn      create;
    FhnDestroyFn     destroy;
    FhnGetKernelsFn  get_kernels;

    // Optional (NULL if not provided)
    FhnSubmitFn      submit;
    FhnPollFn        poll;
    FhnWaitFn        wait;
    FhnGetOutputsFn  get_outputs;
    FhnExecFreeFn    exec_free;
} FhnBackendVTable;

#ifdef __cplusplus
}
#endif

#endif // FHN_BACKEND_API_H
```

**Step 2: Write test verifying struct layout and typedefs**

```cpp
// test/FhnBackendApiTest.cpp
#include "FHN/fhn_backend_api.h"
#include <gtest/gtest.h>
#include <cstring>

TEST(FhnBackendApi, BackendInfoLayout) {
    FhnBackendInfo info{};
    info.name = "test-backend";
    info.version = "0.1";
    info.device_type = FHN_DEVICE_CPU;
    info.device_memory = 1024 * 1024;

    EXPECT_STREQ(info.name, "test-backend");
    EXPECT_STREQ(info.version, "0.1");
    EXPECT_EQ(info.device_type, FHN_DEVICE_CPU);
    EXPECT_EQ(info.device_memory, 1024u * 1024u);
}

TEST(FhnBackendApi, KernelTableLayout) {
    // Simulate a minimal kernel table with one entry
    auto dummy_kernel = [](FhnBackendCtx *, FhnBuffer *,
                           const FhnBuffer *const *, const int64_t *,
                           const double *) -> int { return 0; };

    FhnKernelEntry entries[] = {
        {FHN_ADD_CC, dummy_kernel, "add_cc"},
    };

    FhnKernelTable table{};
    table.num_kernels = 1;
    table.kernels = entries;

    EXPECT_EQ(table.num_kernels, 1u);
    EXPECT_EQ(table.kernels[0].opcode, FHN_ADD_CC);
    EXPECT_NE(table.kernels[0].fn, nullptr);
    EXPECT_STREQ(table.kernels[0].name, "add_cc");
}

TEST(FhnBackendApi, VTableNullOptional) {
    FhnBackendVTable vtable{};
    vtable.get_info = nullptr;
    vtable.create = nullptr;
    vtable.destroy = nullptr;
    vtable.get_kernels = nullptr;
    vtable.submit = nullptr;  // optional

    // All optional fields should be safely nullable
    EXPECT_EQ(vtable.submit, nullptr);
    EXPECT_EQ(vtable.poll, nullptr);
    EXPECT_EQ(vtable.wait, nullptr);
}

TEST(FhnBackendApi, DeviceTypeEnum) {
    EXPECT_EQ(static_cast<int>(FHN_DEVICE_CPU), 0);
    EXPECT_EQ(static_cast<int>(FHN_DEVICE_GPU), 1);
    EXPECT_EQ(static_cast<int>(FHN_DEVICE_FPGA), 2);
    EXPECT_EQ(static_cast<int>(FHN_DEVICE_ASIC), 3);
}
```

**Step 3: Add to build**

Add to `test/CMakeLists.txt`:
```cmake
add_executable(FhnBackendApiTest FhnBackendApiTest.cpp)
target_link_libraries(FhnBackendApiTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(FhnBackendApiTest)
```

**Step 4: Build and run**

Run: `cmake --build build -j8 && ctest --test-dir build -R FhnBackendApi -V`
Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add include/FHN/fhn_backend_api.h test/FhnBackendApiTest.cpp test/CMakeLists.txt
git commit -m "feat: add FHN backend provider interface (fhn_backend_api.h)"
```

---

## Phase 2: FhnDefaultExecutor

### Task 3: Implement FhnDefaultExecutor core dispatch loop

**Files:**
- Create: `include/FHN/FhnDefaultExecutor.h`
- Create: `src/FHN/FhnDefaultExecutor.cpp`
- Test: `test/FhnExecutorTest.cpp`

**Step 1: Write the test**

```cpp
// test/FhnExecutorTest.cpp
#include "FHN/FhnDefaultExecutor.h"
#include "FHN/fhn_program.h"
#include <gtest/gtest.h>
#include <vector>

namespace {

// Fake backend context holding int buffers for testing
struct TestCtx {
    std::vector<int64_t> slots; // slot[id] = value
};

// Fake buffer = pointer to a slot index
struct TestBuffer {
    int64_t value;
};

int test_add_cc(FhnBackendCtx *ctx, FhnBuffer *result,
                const FhnBuffer *const *operands,
                const int64_t *, const double *) {
    auto *r = reinterpret_cast<TestBuffer *>(result);
    auto *a = reinterpret_cast<const TestBuffer *>(operands[0]);
    auto *b = reinterpret_cast<const TestBuffer *>(operands[1]);
    r->value = a->value + b->value;
    return 0;
}

int test_mult_cc(FhnBackendCtx *ctx, FhnBuffer *result,
                 const FhnBuffer *const *operands,
                 const int64_t *, const double *) {
    auto *r = reinterpret_cast<TestBuffer *>(result);
    auto *a = reinterpret_cast<const TestBuffer *>(operands[0]);
    auto *b = reinterpret_cast<const TestBuffer *>(operands[1]);
    r->value = a->value * b->value;
    return 0;
}

int test_negate(FhnBackendCtx *ctx, FhnBuffer *result,
                const FhnBuffer *const *operands,
                const int64_t *, const double *) {
    auto *r = reinterpret_cast<TestBuffer *>(result);
    auto *a = reinterpret_cast<const TestBuffer *>(operands[0]);
    r->value = -a->value;
    return 0;
}

FhnKernelEntry test_kernels[] = {
    {FHN_ADD_CC,  test_add_cc,  "add_cc"},
    {FHN_MULT_CC, test_mult_cc, "mult_cc"},
    {FHN_NEGATE,  test_negate,  "negate"},
};

FhnKernelTable test_table = {
    .num_kernels = 3,
    .kernels = test_kernels,
};

} // namespace

using fhenomenon::FhnDefaultExecutor;

TEST(FhnExecutor, SupportsRegisteredOpcodes) {
    FhnDefaultExecutor exec(&test_table);
    EXPECT_TRUE(exec.supports(FHN_ADD_CC));
    EXPECT_TRUE(exec.supports(FHN_MULT_CC));
    EXPECT_TRUE(exec.supports(FHN_NEGATE));
    EXPECT_FALSE(exec.supports(FHN_ROTATE));
    EXPECT_FALSE(exec.supports(FHN_HMULT));
}

TEST(FhnExecutor, ExecuteSimpleAdd) {
    FhnDefaultExecutor exec(&test_table);

    // Program: r3 = r1 + r2
    auto *prog = fhn_program_alloc(1, 2, 1);
    prog->input_ids[0] = 1;
    prog->input_ids[1] = 2;
    prog->output_ids[0] = 3;
    prog->instructions[0] = {FHN_ADD_CC, 3, {1, 2, 0, 0}, {}, {}};

    // Buffers indexed by result_id
    TestBuffer bufs[4] = {{0}, {10}, {20}, {0}};
    FhnBuffer *buf_ptrs[4] = {
        reinterpret_cast<FhnBuffer *>(&bufs[0]),
        reinterpret_cast<FhnBuffer *>(&bufs[1]),
        reinterpret_cast<FhnBuffer *>(&bufs[2]),
        reinterpret_cast<FhnBuffer *>(&bufs[3]),
    };

    int err = exec.execute(nullptr, prog, buf_ptrs);
    EXPECT_EQ(err, 0);
    EXPECT_EQ(bufs[3].value, 30);  // 10 + 20

    fhn_program_free(prog);
}

TEST(FhnExecutor, ExecuteChain) {
    FhnDefaultExecutor exec(&test_table);

    // Program: t3 = r1 + r2; t4 = t3 * r1
    auto *prog = fhn_program_alloc(2, 2, 1);
    prog->input_ids[0] = 1;
    prog->input_ids[1] = 2;
    prog->output_ids[0] = 4;
    prog->instructions[0] = {FHN_ADD_CC,  3, {1, 2, 0, 0}, {}, {}};
    prog->instructions[1] = {FHN_MULT_CC, 4, {3, 1, 0, 0}, {}, {}};

    TestBuffer bufs[5] = {{0}, {5}, {3}, {0}, {0}};
    FhnBuffer *buf_ptrs[5];
    for (int i = 0; i < 5; i++)
        buf_ptrs[i] = reinterpret_cast<FhnBuffer *>(&bufs[i]);

    int err = exec.execute(nullptr, prog, buf_ptrs);
    EXPECT_EQ(err, 0);
    EXPECT_EQ(bufs[3].value, 8);   // 5 + 3
    EXPECT_EQ(bufs[4].value, 40);  // 8 * 5

    fhn_program_free(prog);
}

TEST(FhnExecutor, UnsupportedOpcodeReturnsError) {
    FhnDefaultExecutor exec(&test_table);

    auto *prog = fhn_program_alloc(1, 1, 1);
    prog->input_ids[0] = 1;
    prog->output_ids[0] = 2;
    prog->instructions[0] = {FHN_ROTATE, 2, {1, 0, 0, 0}, {1, 0, 0, 0}, {}};

    TestBuffer bufs[3] = {{0}, {10}, {0}};
    FhnBuffer *buf_ptrs[3];
    for (int i = 0; i < 3; i++)
        buf_ptrs[i] = reinterpret_cast<FhnBuffer *>(&bufs[i]);

    int err = exec.execute(nullptr, prog, buf_ptrs);
    EXPECT_NE(err, 0);  // should fail for unsupported opcode

    fhn_program_free(prog);
}
```

**Step 2: Write the executor**

```cpp
// include/FHN/FhnDefaultExecutor.h
#pragma once

#include "FHN/fhn_backend_api.h"
#include <unordered_map>

namespace fhenomenon {

class FhnDefaultExecutor {
public:
    explicit FhnDefaultExecutor(FhnKernelTable *table);

    bool supports(FhnOpCode opcode) const;

    // Execute a program by dispatching each instruction to kernel functions.
    // buffers is indexed by result_id.
    // Returns 0 on success, non-zero on error.
    int execute(FhnBackendCtx *ctx, const FhnProgram *program,
                FhnBuffer **buffers);

private:
    std::unordered_map<FhnOpCode, FhnKernelFn> dispatch_;

    // Attempt to decompose a fused opcode into primitives.
    // Returns true if decomposition succeeded and was executed.
    bool decompose(FhnBackendCtx *ctx, const FhnInstruction &inst,
                   FhnBuffer **buffers);
};

} // namespace fhenomenon
```

```cpp
// src/FHN/FhnDefaultExecutor.cpp
#include "FHN/FhnDefaultExecutor.h"

namespace fhenomenon {

FhnDefaultExecutor::FhnDefaultExecutor(FhnKernelTable *table) {
    if (!table) return;
    for (uint32_t i = 0; i < table->num_kernels; i++) {
        if (table->kernels[i].fn) {
            dispatch_[table->kernels[i].opcode] = table->kernels[i].fn;
        }
    }
}

bool FhnDefaultExecutor::supports(FhnOpCode opcode) const {
    return dispatch_.count(opcode) > 0;
}

int FhnDefaultExecutor::execute(FhnBackendCtx *ctx,
                                const FhnProgram *program,
                                FhnBuffer **buffers) {
    if (!program || !buffers) return -1;

    for (uint32_t i = 0; i < program->num_instructions; i++) {
        const auto &inst = program->instructions[i];

        auto it = dispatch_.find(inst.opcode);
        if (it == dispatch_.end()) {
            if (!decompose(ctx, inst, buffers))
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

bool FhnDefaultExecutor::decompose(FhnBackendCtx *,
                                   const FhnInstruction &,
                                   FhnBuffer **) {
    // TODO: Phase 2 Task 4 will implement decomposition rules
    return false;
}

} // namespace fhenomenon
```

**Step 3: Add to build**

Add to `test/CMakeLists.txt`:
```cmake
add_executable(FhnExecutorTest FhnExecutorTest.cpp)
target_link_libraries(FhnExecutorTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(FhnExecutorTest)
```

**Step 4: Build and run**

Run: `cmake --build build -j8 && ctest --test-dir build -R FhnExecutor -V`
Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add include/FHN/FhnDefaultExecutor.h src/FHN/FhnDefaultExecutor.cpp \
        test/FhnExecutorTest.cpp test/CMakeLists.txt
git commit -m "feat: add FhnDefaultExecutor with kernel table dispatch"
```

---

### Task 4: Implement decomposition rules

**Files:**
- Modify: `src/FHN/FhnDefaultExecutor.cpp`
- Add tests to: `test/FhnExecutorTest.cpp`

**Step 1: Add decomposition test**

Append to `test/FhnExecutorTest.cpp`:

```cpp
TEST(FhnExecutor, DecomposeHMultToMultRelinRescale) {
    // Register MULT_CC, RELINEARIZE, RESCALE but NOT HMULT
    FhnKernelEntry kernels[] = {
        {FHN_MULT_CC,      test_mult_cc,  "mult_cc"},
        {FHN_RELINEARIZE,  test_noop,     "relin"},  // noop for test
        {FHN_RESCALE,      test_noop,     "rescale"},
    };
    FhnKernelTable table = {3, kernels};
    FhnDefaultExecutor exec(&table);

    EXPECT_FALSE(exec.supports(FHN_HMULT));  // not directly

    // Program: r3 = HMULT(r1, r2)
    auto *prog = fhn_program_alloc(1, 2, 1);
    prog->input_ids[0] = 1;
    prog->input_ids[1] = 2;
    prog->output_ids[0] = 3;
    prog->instructions[0] = {FHN_HMULT, 3, {1, 2, 0, 0}, {}, {}};

    TestBuffer bufs[6] = {{0}, {5}, {3}, {0}, {0}, {0}};
    FhnBuffer *buf_ptrs[6];
    for (int i = 0; i < 6; i++)
        buf_ptrs[i] = reinterpret_cast<FhnBuffer *>(&bufs[i]);

    int err = exec.execute(nullptr, prog, buf_ptrs);
    EXPECT_EQ(err, 0);
    // MULT_CC should have been called: 5 * 3 = 15
    EXPECT_EQ(bufs[3].value, 15);

    fhn_program_free(prog);
}
```

Add the `test_noop` kernel to the anonymous namespace:

```cpp
int test_noop(FhnBackendCtx *, FhnBuffer *,
              const FhnBuffer *const *, const int64_t *,
              const double *) {
    return 0;  // no-op for testing
}
```

**Step 2: Implement decompose()**

Replace the `decompose` method in `src/FHN/FhnDefaultExecutor.cpp`:

```cpp
bool FhnDefaultExecutor::decompose(FhnBackendCtx *ctx,
                                   const FhnInstruction &inst,
                                   FhnBuffer **buffers) {
    switch (inst.opcode) {
    case FHN_HMULT: {
        // HMULT -> MULT_CC + RELINEARIZE + RESCALE
        if (!supports(FHN_MULT_CC)) return false;

        // Step 1: MULT_CC (result into inst.result_id)
        const FhnBuffer *mult_ops[] = {
            buffers[inst.operands[0]], buffers[inst.operands[1]],
            nullptr, nullptr
        };
        int err = dispatch_[FHN_MULT_CC](
            ctx, buffers[inst.result_id], mult_ops, inst.params, inst.fparams);
        if (err) return false;

        // Step 2: RELINEARIZE (in-place if supported)
        if (supports(FHN_RELINEARIZE)) {
            const FhnBuffer *relin_ops[] = {
                buffers[inst.result_id], nullptr, nullptr, nullptr
            };
            dispatch_[FHN_RELINEARIZE](
                ctx, buffers[inst.result_id], relin_ops, inst.params, inst.fparams);
        }

        // Step 3: RESCALE (in-place if supported)
        if (supports(FHN_RESCALE)) {
            const FhnBuffer *rescale_ops[] = {
                buffers[inst.result_id], nullptr, nullptr, nullptr
            };
            dispatch_[FHN_RESCALE](
                ctx, buffers[inst.result_id], rescale_ops, inst.params, inst.fparams);
        }
        return true;
    }
    case FHN_HROT: {
        // HROT -> MULT_KEY + ROTATE
        if (!supports(FHN_MULT_KEY) || !supports(FHN_ROTATE)) return false;
        const FhnBuffer *ops[] = {
            buffers[inst.operands[0]], nullptr, nullptr, nullptr
        };
        int err = dispatch_[FHN_MULT_KEY](
            ctx, buffers[inst.result_id], ops, inst.params, inst.fparams);
        if (err) return false;
        const FhnBuffer *rot_ops[] = {
            buffers[inst.result_id], nullptr, nullptr, nullptr
        };
        return dispatch_[FHN_ROTATE](
            ctx, buffers[inst.result_id], rot_ops, inst.params, inst.fparams) == 0;
    }
    case FHN_HROT_ADD: {
        // HROT_ADD -> HROT(a) + ADD_CC
        // Recursively: HROT decomposes further if needed
        FhnInstruction hrot_inst = inst;
        hrot_inst.opcode = FHN_HROT;
        // Try direct dispatch or decompose
        auto hrot_it = dispatch_.find(FHN_HROT);
        bool ok;
        if (hrot_it != dispatch_.end()) {
            const FhnBuffer *ops[] = {
                buffers[inst.operands[0]], nullptr, nullptr, nullptr
            };
            ok = hrot_it->second(ctx, buffers[inst.result_id],
                                 ops, inst.params, inst.fparams) == 0;
        } else {
            ok = decompose(ctx, hrot_inst, buffers);
        }
        if (!ok || !supports(FHN_ADD_CC)) return false;
        const FhnBuffer *add_ops[] = {
            buffers[inst.result_id], buffers[inst.operands[1]],
            nullptr, nullptr
        };
        return dispatch_[FHN_ADD_CC](
            ctx, buffers[inst.result_id], add_ops, inst.params, inst.fparams) == 0;
    }
    default:
        return false;
    }
}
```

**Step 3: Build and run**

Run: `cmake --build build -j8 && ctest --test-dir build -R FhnExecutor -V`
Expected: 5 tests PASS (including new decompose test)

**Step 4: Commit**

```bash
git add src/FHN/FhnDefaultExecutor.cpp test/FhnExecutorTest.cpp
git commit -m "feat: add decomposition rules (HMULT, HROT, HROT_ADD)"
```

---

## Phase 3: BuiltinBackend Refactor (Reference Implementation)

### Task 5: Implement ToyFHE kernel table

**Files:**
- Create: `src/FHN/ToyFheKernels.cpp`
- Create: `include/FHN/ToyFheKernels.h`
- Test: `test/FhnToyFheTest.cpp`

**Step 1: Write the test**

```cpp
// test/FhnToyFheTest.cpp
#include "FHN/FhnDefaultExecutor.h"
#include "FHN/ToyFheKernels.h"
#include "FHN/fhn_program.h"
#include <gtest/gtest.h>

using namespace fhenomenon;

class FhnToyFheTest : public ::testing::Test {
protected:
    void SetUp() override {
        info_ = toyfhe_fhn_get_info();
        ctx_ = toyfhe_fhn_create(nullptr);
        table_ = toyfhe_fhn_get_kernels(ctx_);
        executor_ = std::make_unique<FhnDefaultExecutor>(table_);
    }

    void TearDown() override {
        toyfhe_fhn_destroy(ctx_);
    }

    FhnBackendInfo *info_ = nullptr;
    FhnBackendCtx *ctx_ = nullptr;
    FhnKernelTable *table_ = nullptr;
    std::unique_ptr<FhnDefaultExecutor> executor_;
};

TEST_F(FhnToyFheTest, BackendInfo) {
    EXPECT_STREQ(info_->name, "toyfhe-reference");
    EXPECT_EQ(info_->device_type, FHN_DEVICE_CPU);
}

TEST_F(FhnToyFheTest, SupportsBasicOps) {
    EXPECT_TRUE(executor_->supports(FHN_ENCRYPT));
    EXPECT_TRUE(executor_->supports(FHN_DECRYPT));
    EXPECT_TRUE(executor_->supports(FHN_ADD_CC));
    EXPECT_TRUE(executor_->supports(FHN_HMULT));
    EXPECT_FALSE(executor_->supports(FHN_ROTATE));  // ToyFHE doesn't support
    EXPECT_FALSE(executor_->supports(FHN_AND));      // Not a boolean scheme
}

TEST_F(FhnToyFheTest, EncryptAddDecrypt) {
    // Program:
    //   buf[1] = ENCRYPT(value=10)
    //   buf[2] = ENCRYPT(value=20)
    //   buf[3] = ADD_CC(buf[1], buf[2])
    //   buf[4] = DECRYPT(buf[3])

    auto *prog = fhn_program_alloc(4, 0, 1);
    prog->output_ids[0] = 4;

    prog->instructions[0] = {FHN_ENCRYPT, 1, {0,0,0,0}, {10,0,0,0}, {}};
    prog->instructions[1] = {FHN_ENCRYPT, 2, {0,0,0,0}, {20,0,0,0}, {}};
    prog->instructions[2] = {FHN_ADD_CC,  3, {1,2,0,0}, {},          {}};
    prog->instructions[3] = {FHN_DECRYPT, 4, {3,0,0,0}, {},          {}};

    // Allocate buffers via backend
    FhnBuffer *bufs[5];
    for (int i = 0; i < 5; i++)
        bufs[i] = toyfhe_fhn_buffer_alloc(ctx_);

    int err = executor_->execute(ctx_, prog, bufs);
    EXPECT_EQ(err, 0);

    // Retrieve decrypted value
    int64_t result = toyfhe_fhn_buffer_read_int(ctx_, bufs[4]);
    EXPECT_EQ(result, 30);

    for (int i = 0; i < 5; i++)
        toyfhe_fhn_buffer_free(ctx_, bufs[i]);
    fhn_program_free(prog);
}
```

NOTE: The exact implementation of `ToyFheKernels.h/cpp` will depend on how ToyFHE's `Engine` class maps to the FhnBuffer opaque type. The implementation should:
- Define `FhnBackendCtx` as a struct wrapping `toyfhe::Engine`
- Define `FhnBuffer` as a struct wrapping `toyfhe::Ciphertext` (or a variant for plaintext/int values)
- Implement kernel functions that cast `FhnBuffer*` to the appropriate ToyFHE type
- Export `toyfhe_fhn_get_info`, `toyfhe_fhn_create`, `toyfhe_fhn_destroy`, `toyfhe_fhn_get_kernels`

**Step 2: Implement ToyFHE kernels**

See design doc Section 6 for reference. Key implementation detail: `FhnBuffer` wraps a tagged union of `{toyfhe::Ciphertext, int64_t, double}` to handle encrypt/decrypt data flow.

**Step 3: Add to build and run**

```cmake
add_executable(FhnToyFheTest FhnToyFheTest.cpp)
target_link_libraries(FhnToyFheTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(FhnToyFheTest)
```

Run: `cmake --build build -j8 && ctest --test-dir build -R FhnToyFhe -V`
Expected: 3 tests PASS

**Step 4: Commit**

```bash
git add include/FHN/ToyFheKernels.h src/FHN/ToyFheKernels.cpp \
        test/FhnToyFheTest.cpp test/CMakeLists.txt
git commit -m "feat: add ToyFHE kernel table (reference implementation)"
```

---

### Task 6: Wire BuiltinBackend to use FhnDefaultExecutor

**Files:**
- Modify: `include/Backend/Builtin.h`
- Modify: `src/Backend/Builtin.cpp`

**Goal:** Add FhnDefaultExecutor as a member of BuiltinBackend. Keep existing `add()`/`multiply()` methods working (they now delegate to the executor internally). This is a gradual migration — old interface wraps new infrastructure.

**Step 1: Add executor member to BuiltinBackend**

In `include/Backend/Builtin.h`, add:
```cpp
#include "FHN/FhnDefaultExecutor.h"
#include "FHN/ToyFheKernels.h"

// Add to private members:
std::unique_ptr<FhnDefaultExecutor> executor_;
FhnBackendCtx *fhn_ctx_ = nullptr;
FhnKernelTable *fhn_table_ = nullptr;
```

**Step 2: Initialize executor in constructor**

In `BuiltinBackend::initialize()`:
```cpp
fhn_ctx_ = toyfhe_fhn_create(nullptr);
fhn_table_ = toyfhe_fhn_get_kernels(fhn_ctx_);
executor_ = std::make_unique<FhnDefaultExecutor>(fhn_table_);
```

**Step 3: Verify existing tests still pass**

Run: `cmake --build build -j8 && ctest --test-dir build -V`
Expected: ALL existing tests PASS (no regression)

**Step 4: Commit**

```bash
git add include/Backend/Builtin.h src/Backend/Builtin.cpp
git commit -m "refactor: wire FhnDefaultExecutor into BuiltinBackend"
```

---

## Phase 4: Scheduler Integration

### Task 7: Implement LowerToFhnProgram pass

**Files:**
- Create: `include/Scheduler/LowerToFhnProgram.h`
- Create: `src/Scheduler/LowerToFhnProgram.cpp`
- Test: `test/LowerToFhnProgramTest.cpp`

**Step 1: Write the test**

```cpp
// test/LowerToFhnProgramTest.cpp
#include "Scheduler/LowerToFhnProgram.h"
#include "Scheduler/ASTNode.h"
#include "Scheduler/Operation.h"
#include "Scheduler/Planner.h"
#include "FHN/fhn_program.h"
#include "Compuon.h"
#include <gtest/gtest.h>

using namespace fhenomenon;
using namespace fhenomenon::scheduler;

TEST(LowerToFhnProgram, SingleAdd) {
    // Build AST: result = a + b
    auto a = std::make_shared<Compuon<int>>(10);
    auto b = std::make_shared<Compuon<int>>(20);
    auto result = std::make_shared<Compuon<int>>(0);

    auto op = std::make_shared<Operation<int>>(OperationType::Add, a, b, result);
    auto left = std::make_shared<OperandNode<int>>(a);
    auto right = std::make_shared<OperandNode<int>>(b);
    auto root = std::make_shared<OperatorNode<int>>(
        op, OperationType::Add, left, right, result);

    Planner<int> plan;
    plan.addRoot(root);

    LowerToFhnProgram lowering;
    auto *prog = lowering.lower(plan);

    ASSERT_NE(prog, nullptr);
    // 2 inputs (a, b) + 1 operation (add)
    EXPECT_EQ(prog->num_inputs, 2u);
    EXPECT_EQ(prog->num_instructions, 1u);
    EXPECT_EQ(prog->instructions[0].opcode, FHN_ADD_CC);
    EXPECT_EQ(prog->num_outputs, 1u);

    fhn_program_free(prog);
}

TEST(LowerToFhnProgram, ChainedAddMult) {
    // AST: t = a + b; result = t * c
    auto a = std::make_shared<Compuon<int>>(1);
    auto b = std::make_shared<Compuon<int>>(2);
    auto c = std::make_shared<Compuon<int>>(3);
    auto t = std::make_shared<Compuon<int>>(0);
    auto result = std::make_shared<Compuon<int>>(0);

    auto add_op = std::make_shared<Operation<int>>(OperationType::Add, a, b, t);
    auto mul_op = std::make_shared<Operation<int>>(OperationType::Multiply, t, c, result);

    auto leaf_a = std::make_shared<OperandNode<int>>(a);
    auto leaf_b = std::make_shared<OperandNode<int>>(b);
    auto leaf_c = std::make_shared<OperandNode<int>>(c);
    auto add_node = std::make_shared<OperatorNode<int>>(
        add_op, OperationType::Add, leaf_a, leaf_b, t);
    auto mul_node = std::make_shared<OperatorNode<int>>(
        mul_op, OperationType::Multiply, add_node, leaf_c, result);

    Planner<int> plan;
    plan.addRoot(mul_node);

    LowerToFhnProgram lowering;
    auto *prog = lowering.lower(plan);

    ASSERT_NE(prog, nullptr);
    EXPECT_EQ(prog->num_inputs, 3u);     // a, b, c
    EXPECT_EQ(prog->num_instructions, 2u);
    EXPECT_EQ(prog->instructions[0].opcode, FHN_ADD_CC);  // topological: add first
    EXPECT_EQ(prog->instructions[1].opcode, FHN_HMULT);   // then mult
    // The mult's first operand should reference the add's result
    EXPECT_EQ(prog->instructions[1].operands[0],
              prog->instructions[0].result_id);

    fhn_program_free(prog);
}
```

**Step 2: Implement the lowering pass**

```cpp
// include/Scheduler/LowerToFhnProgram.h
#pragma once

#include "FHN/fhn_program.h"
#include "Scheduler/ASTNode.h"
#include "Scheduler/Planner.h"
#include <unordered_map>
#include <vector>

namespace fhenomenon {
namespace scheduler {

class LowerToFhnProgram {
public:
    template <typename T>
    FhnProgram *lower(const Planner<T> &plan) const;

private:
    template <typename T>
    void lowerNode(ASTNode *node, std::vector<FhnInstruction> &insts,
                   std::vector<uint32_t> &inputs,
                   std::vector<uint32_t> &outputs,
                   uint32_t &next_id,
                   std::unordered_map<ASTNode*, uint32_t> &ids) const;

    static FhnOpCode mapOpType(OperationType t);
};

} // namespace scheduler
} // namespace fhenomenon
```

Implementation in `src/Scheduler/LowerToFhnProgram.cpp`:
- Post-order traversal of AST roots
- OperandNode → register as input, assign result_id
- OperatorNode → emit FhnInstruction with children's result_ids as operands
- Root results → register as outputs
- Allocate FhnProgram via `fhn_program_alloc`, copy data in

**Step 3: Add to build and run**

```cmake
add_executable(LowerToFhnProgramTest LowerToFhnProgramTest.cpp)
target_link_libraries(LowerToFhnProgramTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(LowerToFhnProgramTest)
```

Run: `cmake --build build -j8 && ctest --test-dir build -R LowerToFhn -V`
Expected: 2 tests PASS

**Step 4: Commit**

```bash
git add include/Scheduler/LowerToFhnProgram.h src/Scheduler/LowerToFhnProgram.cpp \
        test/LowerToFhnProgramTest.cpp test/CMakeLists.txt
git commit -m "feat: add LowerToFhnProgram pass (AST -> FhnProgram)"
```

---

### Task 8: Wire Scheduler::evaluateGraph to new pipeline

**Files:**
- Modify: `include/Scheduler/Scheduler.h`

**Goal:** Add alternative `evaluateGraphFhn()` method that uses the new pipeline: lower → execute. The existing `evaluateGraph()` stays for backward compatibility.

**Step 1: Add evaluateGraphFhn to Scheduler**

```cpp
template <typename T>
void evaluateGraphFhn(Planner<T> &plan, FhnDefaultExecutor &executor,
                      FhnBackendCtx *ctx) {
    LowerToFhnProgram lowering;
    auto *program = lowering.lower(plan);

    // Allocate buffers for all result_ids
    // ... (buffer management depends on backend)

    executor.execute(ctx, program, buffers);

    // Collect results back into Compuon entities
    // ...

    fhn_program_free(program);
}
```

**Step 2: Test end-to-end with Session (integration test)**

This test creates a Session, runs operations, and verifies the new pipeline produces correct results matching the old pipeline.

**Step 3: Run all tests**

Run: `cmake --build build -j8 && ctest --test-dir build -V`
Expected: ALL tests PASS

**Step 4: Commit**

```bash
git add include/Scheduler/Scheduler.h
git commit -m "feat: wire evaluateGraphFhn into Scheduler"
```

---

## Phase 5: ExternalBackend with dlopen

### Task 9: Implement ExternalBackend loader

**Files:**
- Modify: `include/Backend/External.h`
- Create: `src/Backend/External.cpp`
- Test: `test/FhnExternalBackendTest.cpp`

**Goal:** ExternalBackend uses `dlopen` to load a `.so`, resolves the 4 required `fhn_*` symbols into an `FhnBackendVTable`, and dispatches through `FhnDefaultExecutor`.

**Step 1: Implement ExternalBackend**

```cpp
// Key members:
void *dl_handle_;          // dlopen handle
FhnBackendVTable vtable_;  // resolved function pointers
FhnBackendCtx *fhn_ctx_;
std::unique_ptr<FhnDefaultExecutor> executor_;
```

Constructor:
1. `dlopen(libraryPath, RTLD_LAZY)`
2. `dlsym` for `fhn_get_info`, `fhn_create`, `fhn_destroy`, `fhn_get_kernels`
3. Optional: `dlsym` for `fhn_submit`, `fhn_poll`, `fhn_wait` (NULL if absent)
4. Call `fhn_create(config_json)` to get context
5. Call `fhn_get_kernels(ctx)` to get kernel table
6. Create `FhnDefaultExecutor` from kernel table

**Step 2: Test with a mock .so**

Build ToyFHE kernels as a shared library for testing, load via ExternalBackend.

**Step 3: Commit**

```bash
git add include/Backend/External.h src/Backend/External.cpp \
        test/FhnExternalBackendTest.cpp
git commit -m "feat: implement ExternalBackend with dlopen loader"
```

---

## Phase 6: Cheddar Integration (separate repository work)

### Task 10: Create cheddar FHN backend skeleton

**Files (in cheddar-fhe repo or as a separate build target):**
- Create: `src/fhn/cheddar_fhn_backend.cpp`
- Create: `src/fhn/cheddar_fhn_backend.h`

**Goal:** Implement the 4 required exports for cheddar:
- `fhn_get_info()` → returns GPU device info
- `fhn_create(config_json)` → parse JSON params, create `cheddar::Context<uint32_t>` + `UserInterface<uint32_t>`
- `fhn_destroy()` → cleanup
- `fhn_get_kernels()` → return kernel table with cheddar operations

**Kernel implementations (each maps to one `cheddar::Context` call):**
- `FHN_ADD_CC` → `Context::Add(res, a, b)`
- `FHN_SUB_CC` → `Context::Sub(res, a, b)`
- `FHN_NEGATE` → `Context::Neg(res, a)`
- `FHN_MULT_CC` → `Context::Mult(res, a, b)`
- `FHN_MULT_CP` → `Context::Mult(res, a, b)` (plaintext overload)
- `FHN_HMULT` → `Context::HMult(res, a, b, mult_key)`
- `FHN_ROTATE` → `Context::HRot(res, a, rot_key, dist)`
- `FHN_CONJUGATE` → `Context::HConj(res, a, conj_key)`
- `FHN_RESCALE` → `Context::Rescale(res, a)`
- `FHN_RELINEARIZE` → `Context::Relinearize(res, a, key)`
- `FHN_HROT_ADD` → `Context::HRotAdd(res, a, b, key, dist)`
- `FHN_HCONJ_ADD` → `Context::HConjAdd(res, a, b, key)`
- `FHN_ENCRYPT` → `UserInterface::Encrypt(ct, pt)`
- `FHN_DECRYPT` → `UserInterface::Decrypt(pt, ct)`
- `FHN_ENCODE` → `Encoder::Encode(pt, level, scale, msg)`
- `FHN_DECODE` → `Encoder::Decode(msg, pt)`

**Build:** Produces `libcheddar_fhn.so` linked against `libcheddar.so` + CUDA.

**Test:** Load via fhenomenon's ExternalBackend, run encrypt-add-decrypt cycle.

---

## Task Dependency Graph

```
Task 1 (fhn_program.h)
  └─> Task 2 (fhn_backend_api.h)
        └─> Task 3 (FhnDefaultExecutor)
              ├─> Task 4 (decomposition rules)
              └─> Task 5 (ToyFHE kernel table)
                    └─> Task 6 (wire BuiltinBackend)
                          └─> Task 8 (wire Scheduler)
        └─> Task 7 (LowerToFhnProgram)
              └─> Task 8 (wire Scheduler)
        └─> Task 9 (ExternalBackend dlopen)
              └─> Task 10 (cheddar integration)
```

## Build & Test Commands

```bash
# Configure (from project root)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build build -j8

# Run all tests
ctest --test-dir build -V

# Run specific test
ctest --test-dir build -R FhnProgram -V
ctest --test-dir build -R FhnExecutor -V
ctest --test-dir build -R FhnToyFhe -V
ctest --test-dir build -R LowerToFhn -V
```
