#include "Session/Session.h"
#include "FHN/FhnDefaultExecutor.h"
#include "FHN/FhnMovementPlan.h"
#include "FHN/fhn_program.h"
#include "Scheduler/MatMulRecognitionPass.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

  // Walking the bindings forward, the last binding per entity wins — that
  // id holds the value the entity must observe after the run.
  std::unordered_map<Fhenon<int> *, std::pair<std::shared_ptr<Fhenon<int>>, uint32_t>> latest;
  for (const auto &[id, entity] : bindings) {
    if (entity) {
      latest[entity.get()] = {entity, id};
    }
  }

  // Pin what must survive execution: program inputs (entity-owned buffers
  // the plan must not free) and every write-back target. Superseded
  // intermediate results are deliberately NOT pinned — the plan frees them
  // at their last use.
  std::vector<uint32_t> pinned(program->input_ids, program->input_ids + program->num_inputs);
  // Aliased entities (e.g. an input that is also a write-back target) can
  // push the same id into `pinned` more than once; analyze() dedupes
  // pinned ids internally, so no dedup is needed here.
  for (const auto &[raw_entity, bound] : latest) {
    (void)raw_entity;
    pinned.push_back(bound.second);
  }

  const auto plan = FhnMovementPlan::analyze(*program, pinned, /*device_budget=*/0);
  if (!plan) {
    throw std::runtime_error("Session: FHN program failed movement analysis (operand used without a definition?)");
  }

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

  // Input provisioning only: every other id is allocated by the plan. An
  // encrypted entity bound to an input id contributes its buffer in place
  // (zero copy); a foreign owner is an error, not a skip.
  const std::unordered_set<uint32_t> input_ids(program->input_ids, program->input_ids + program->num_inputs);
  std::vector<std::shared_ptr<FhnBuffer>> input_hold(max_id + 1);
  for (const auto &[id, entity] : bindings) {
    if (!input_ids.count(id) || input_hold[id] || !entity || !entity->isEncrypted_) {
      continue;
    }
    if (const auto *ct = std::any_cast<FhnCiphertext>(&entity->ciphertext_)) {
      if (ct->owner != &backend) {
        throw std::runtime_error("Session: operand was encrypted by a different backend");
      }
      input_hold[id] = ct->buffer;
    }
  }
  for (uint32_t i = 0; i < program->num_inputs; ++i) {
    if (!input_hold[program->input_ids[i]]) {
      throw std::runtime_error("Session: operand is not encrypted — call belong() before using it in a session");
    }
  }

  std::vector<FhnBuffer *> buffers(max_id + 1, nullptr);
  for (uint32_t id = 1; id <= max_id; ++id) {
    if (input_hold[id]) {
      buffers[id] = input_hold[id].get();
    }
  }

  const FhnMovementHooks hooks{runtime.ctx, runtime.buffer_alloc, runtime.buffer_free, runtime.prefetch, runtime.evict};
  const int rc = runtime.executor->execute(hooks, program.get(), buffers.data(), *plan);
  if (rc != 0) {
    throw std::runtime_error("Session: FHN executor failed with rc=" + std::to_string(rc));
  }

  // Adopt surviving pinned buffers and write back. Adoption is deduped by
  // id: two entities bound to the same value id must share one shared_ptr,
  // not wrap the same raw pointer twice (double free). Input ids reuse the
  // entity-owned shared_ptr; plan-allocated ids get a deleter sharing the
  // runtime keepalive, exactly like the old preallocation path.
  auto *ctx = runtime.ctx;
  auto free_fn = runtime.buffer_free;
  auto keepalive = runtime.keepalive;
  std::unordered_map<uint32_t, std::shared_ptr<FhnBuffer>> adopted;
  auto adopt = [&](uint32_t id) -> std::shared_ptr<FhnBuffer> {
    if (input_hold[id]) {
      return input_hold[id];
    }
    auto &slot = adopted[id];
    if (!slot) {
      slot =
        std::shared_ptr<FhnBuffer>(buffers[id], [ctx, free_fn, keepalive](FhnBuffer *buffer) { free_fn(ctx, buffer); });
    }
    return slot;
  };
  for (auto &[raw_entity, bound] : latest) {
    (void)raw_entity;
    bound.first->ciphertext_ = FhnCiphertext{adopt(bound.second), &backend};
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
