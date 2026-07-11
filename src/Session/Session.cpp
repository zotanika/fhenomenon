#include "Session/Session.h"
#include "FHN/FhnDefaultExecutor.h"
#include "FHN/fhn_program.h"
#include "Scheduler/MatMulRecognitionPass.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace fhenomenon;

thread_local std::shared_ptr<Session> Session::session_ptr_ = nullptr;

namespace {

// Execute the recorded graph through the backend's FHN runtime: lower the
// AST to an FhnProgram, provision the buffer table, dispatch through the
// executor, and write results back into the bound entities.
// Returns false when lowering produced nothing executable, in which case the
// caller falls back to the legacy per-operation path.
bool executeThroughFhnRuntime(scheduler::Scheduler &scheduler, scheduler::Planner<int> &planner, const Backend &backend,
                              const FhnRuntime &runtime) {
  scheduler::LowerToFhnProgram::EntityBindings<int> bindings;
  std::unique_ptr<FhnProgram, decltype(&fhn_program_free)> program(scheduler.lowerGraph<int>(planner, &bindings),
                                                                   &fhn_program_free);
  if (!program || bindings.empty()) {
    return false;
  }

  // Buffer table sized by the largest id in play (id 0 means "unused").
  uint32_t max_id = 0;
  for (uint32_t i = 0; i < program->num_instructions; ++i) {
    max_id = std::max(max_id, program->instructions[i].result_id);
  }
  for (uint32_t i = 0; i < program->num_inputs; ++i) {
    max_id = std::max(max_id, program->input_ids[i]);
  }
  for (const auto &binding : bindings) {
    max_id = std::max(max_id, binding.first);
  }

  std::vector<std::shared_ptr<FhnBuffer>> owned(max_id + 1);

  // First-wins fill: an id's first binding is its defining site. When that
  // entity already holds an FHN ciphertext owned by this backend, use it in
  // place (zero copy). A ciphertext owned by a different backend is an
  // error, not a skip: its bytes are meaningless to this backend's kernels.
  for (const auto &[id, entity] : bindings) {
    if (owned[id] || !entity || !entity->isEncrypted_) {
      continue;
    }
    if (const auto *ct = std::any_cast<FhnCiphertext>(&entity->ciphertext_)) {
      if (ct->owner != &backend) {
        throw std::runtime_error("Session: operand was encrypted by a different backend");
      }
      owned[id] = ct->buffer;
    }
  }

  // Every program input must now be provisioned: an input id without a
  // buffer means an operand entity that was never encrypted (no belong()),
  // and executing would silently compute on an empty buffer.
  for (uint32_t i = 0; i < program->num_inputs; ++i) {
    if (!owned[program->input_ids[i]]) {
      throw std::runtime_error("Session: operand is not encrypted — call belong() before using it in a session");
    }
  }

  for (uint32_t id = 1; id <= max_id; ++id) {
    if (owned[id]) {
      continue;
    }
    FhnBuffer *raw = runtime.buffer_alloc(runtime.ctx);
    if (!raw) {
      throw std::runtime_error("Session: FHN buffer allocation failed");
    }
    // The deleter shares the runtime keepalive so intermediate buffers can
    // safely outlive the backend that allocated them.
    auto *ctx = runtime.ctx;
    auto free_fn = runtime.buffer_free;
    auto keepalive = runtime.keepalive;
    owned[id] = std::shared_ptr<FhnBuffer>(raw, [ctx, free_fn, keepalive](FhnBuffer *buffer) { free_fn(ctx, buffer); });
  }

  std::vector<FhnBuffer *> buffers(max_id + 1, nullptr);
  for (uint32_t id = 1; id <= max_id; ++id) {
    buffers[id] = owned[id].get();
  }

  const int rc = runtime.executor->execute(runtime.ctx, program.get(), buffers.data());
  if (rc != 0) {
    throw std::runtime_error("Session: FHN executor failed with rc=" + std::to_string(rc));
  }

  // Write-back: walking the bindings forward, the last binding per entity
  // wins — that is the id holding the value the entity must observe after
  // the run (assignment overwrite semantics).
  std::unordered_map<Fhenon<int> *, std::pair<std::shared_ptr<Fhenon<int>>, uint32_t>> latest;
  for (const auto &[id, entity] : bindings) {
    if (entity) {
      latest[entity.get()] = {entity, id};
    }
  }
  for (auto &[raw_entity, bound] : latest) {
    (void)raw_entity;
    bound.first->ciphertext_ = FhnCiphertext{owned[bound.second], &backend};
    bound.first->isEncrypted_ = true;
  }

  return true;
}

} // namespace

void Session::optimize() {
  LOG_MESSAGE("Session: optimize");

  // A recorded entity died before evaluation: executing the graph would read
  // dead objects, so refuse deterministically instead.
  if (poisoned_) {
    throw std::runtime_error(poison_reason_);
  }

  if (!scheduler_) {
    throw std::runtime_error("Session: Scheduler not initialized");
  }

  // An empty recording is a valid no-op run.
  if (operations_.empty()) {
    return;
  }

  // Register passes once; the scheduler persists across run() calls.
  if (!passes_registered_) {
    // Register pre-AST passes
    scheduler_->addPreASTPass(std::make_shared<scheduler::MatMulRecognitionPass>());

    // Scheduler uses backend as delegate - it's decoupled from actual computation
    scheduler_->addASTPass(std::make_shared<scheduler::PrintASTPass>());
    passes_registered_ = true;
  }
  // scheduler_->addASTPass(std::make_shared<scheduler::FuseOperationsASTPass>());
  // scheduler_->addASTPass(std::make_shared<scheduler::PrintASTPass>());
  // scheduler_->addASTPass(std::make_shared<scheduler::AdditionToMultiplicationASTPass>());
  // scheduler_->addASTPass(std::make_shared<scheduler::PrintASTPass>());

  scheduler::Planner<int> planner;
  planner = scheduler_->buildGraph<int>(operations_, planner);
  scheduler_->optimizeGraph(planner);

  // Backends that expose an FHN runtime execute the whole plan as one
  // FhnProgram through the executor; the rest use the legacy per-operation
  // evaluation. Lowering failures also fall back.
  const FhnRuntime *runtime = backend_.fhnRuntime();
  if (runtime && executeThroughFhnRuntime(*scheduler_, planner, backend_, *runtime)) {
    return;
  }
  scheduler_->evaluateGraph(planner);
}
