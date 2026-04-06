#include "FHN/fhn_program.h"

#include <cstdlib>

extern "C" {

FhnProgram *fhn_program_alloc(uint32_t num_instructions, uint32_t num_inputs, uint32_t num_outputs) {
  auto *program = static_cast<FhnProgram *>(std::calloc(1, sizeof(FhnProgram)));
  if (program == nullptr) {
    return nullptr;
  }

  program->version = 1;
  program->num_instructions = num_instructions;
  program->num_inputs = num_inputs;
  program->num_outputs = num_outputs;

  if (num_instructions > 0) {
    program->instructions = static_cast<FhnInstruction *>(std::calloc(num_instructions, sizeof(FhnInstruction)));
    if (program->instructions == nullptr) {
      std::free(program);
      return nullptr;
    }
  }

  if (num_inputs > 0) {
    program->input_ids = static_cast<uint32_t *>(std::calloc(num_inputs, sizeof(uint32_t)));
    if (program->input_ids == nullptr) {
      std::free(program->instructions);
      std::free(program);
      return nullptr;
    }
  }

  if (num_outputs > 0) {
    program->output_ids = static_cast<uint32_t *>(std::calloc(num_outputs, sizeof(uint32_t)));
    if (program->output_ids == nullptr) {
      std::free(program->input_ids);
      std::free(program->instructions);
      std::free(program);
      return nullptr;
    }
  }

  return program;
}

void fhn_program_free(FhnProgram *program) {
  if (program == nullptr) {
    return;
  }
  std::free(program->instructions);
  std::free(program->input_ids);
  std::free(program->output_ids);
  std::free(program);
}

} /* extern "C" */
