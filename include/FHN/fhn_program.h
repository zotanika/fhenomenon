#ifndef FHN_PROGRAM_H
#define FHN_PROGRAM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum FhnOpCode {
    FHN_NOP = 0,

    /* Data lifecycle */
    FHN_ENCODE,
    FHN_ENCRYPT,
    FHN_DECRYPT,
    FHN_DECODE,

    /* Arithmetic (same-level, same-scale) */
    FHN_ADD_CC,       /* ciphertext + ciphertext */
    FHN_ADD_CP,       /* ciphertext + plaintext  */
    FHN_ADD_CS,       /* ciphertext + scalar     */
    FHN_SUB_CC,
    FHN_SUB_CP,
    FHN_SUB_SC,       /* scalar - ciphertext     */
    FHN_NEGATE,
    FHN_MULT_CC,      /* ciphertext * ciphertext (tensor only) */
    FHN_MULT_CP,      /* ciphertext * plaintext  */
    FHN_MULT_CS,      /* ciphertext * scalar     */

    /* Key-switching operations */
    FHN_RELINEARIZE,
    FHN_RESCALE,
    FHN_ROTATE,       /* params[0] = rotation distance */
    FHN_CONJUGATE,
    FHN_MULT_KEY,

    /* Level management */
    FHN_MOD_DOWN,
    FHN_LEVEL_DOWN,   /* params[0] = target level */

    /* Fused composites (backend decomposes if unsupported) */
    FHN_HMULT,        /* mult + relin + rescale   */
    FHN_HROT,         /* key_switch + permute     */
    FHN_HROT_ADD,     /* rotate(a) + b            */
    FHN_HCONJ_ADD,    /* conj(a) + b              */
    FHN_MAD,          /* res += a * scalar         */

    /* Boolean / comparison (TFHE-style) */
    FHN_AND,
    FHN_OR,
    FHN_XOR,
    FHN_EQ,
    FHN_LT,
    FHN_LE,

    FHN_OPCODE_COUNT  /* sentinel */
} FhnOpCode;

typedef struct FhnInstruction {
    FhnOpCode   opcode;
    uint32_t    result_id;
    uint32_t    operands[4];   /* 0 = unused */
    int64_t     params[4];     /* opcode-specific integers */
    double      fparams[2];    /* opcode-specific floats   */
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

FhnProgram *fhn_program_alloc(uint32_t num_instructions,
                              uint32_t num_inputs,
                              uint32_t num_outputs);
void        fhn_program_free(FhnProgram *program);

#ifdef __cplusplus
}
#endif

#endif /* FHN_PROGRAM_H */
