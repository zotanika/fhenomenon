#include "FHN/Scope.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <unordered_map>
#include <utility>

namespace fhenomenon::runtime {
namespace {
constexpr auto absent = std::numeric_limits<std::size_t>::max();
std::uint64_t nextScopeIdentity() {
  static std::atomic<std::uint64_t> next{1};
  auto current = next.load(std::memory_order_relaxed);
  for (;;) {
    if (current == std::numeric_limits<std::uint64_t>::max())
      throw Error(ErrorCode::ExecutionFailed, "scope identities exhausted");
    if (next.compare_exchange_weak(current, current + 1, std::memory_order_relaxed))
      return current;
  }
}
void checkGrowth(std::size_t existing, std::size_t additional, std::size_t cap, const char *message) {
  if (existing > cap || additional > cap - existing)
    throw Error(ErrorCode::BudgetExceeded, message);
}
} // namespace

struct Scope::Impl {
  struct Slot {
    Value concrete;
    std::size_t definition = absent;
    std::size_t last_use = absent;
    bool exported = false;
  };
  Runtime &runtime;
  ScopeOptions options;
  std::uint64_t identity = nextScopeIdentity();
  bool capturing = true;
  std::size_t input_edges = 0;
  std::vector<ScopeCommand> commands;
  std::vector<Slot> slots;
  std::unordered_map<std::uint64_t, std::size_t> imports;
  Impl(Runtime &owner, ScopeOptions limits) : runtime(owner), options(limits) {}
};

Scope::Scope(Runtime &runtime, ScopeOptions options) : impl_(std::make_unique<Impl>(runtime, options)) {
  if (!options.max_commands || !options.max_values || !options.max_input_edges || !options.max_exports ||
      !options.max_lookahead || !options.max_match_attempts)
    throw Error(ErrorCode::InvalidRequest, "scope numeric caps must be positive");
}
Scope::~Scope() = default;
bool Scope::belongsTo(const Runtime &runtime) const noexcept { return &impl_->runtime == &runtime; }
void Scope::validate(ScopeValue symbol) const {
  if (symbol.scope_ != impl_->identity || symbol.id_ >= impl_->slots.size())
    throw Error(ErrorCode::InvalidRequest, "invalid symbol or symbol belongs to another scope");
}
ScopeValue Scope::input(const Value &value) {
  if (!impl_->capturing)
    throw Error(ErrorCode::InvalidRequest, "scope recording has finished");
  if (!impl_->runtime.owns(value))
    throw Error(ErrorCode::InvalidRequest, "scope input belongs to another runtime or is empty");
  const auto existing = impl_->imports.find(value.identity());
  if (existing != impl_->imports.end())
    return ScopeValue(impl_->identity, existing->second);
  checkGrowth(impl_->slots.size(), 1, impl_->options.max_values, "scope value cap exceeded");
  const auto index = impl_->slots.size();
  impl_->slots.push_back({value, absent, absent, false});
  impl_->imports.emplace(value.identity(), index);
  return ScopeValue(impl_->identity, index);
}
ScopeValues Scope::submit(std::string operation, ScopeValues inputs, std::any attributes, std::size_t output_count) {
  if (!impl_->capturing || operation.empty())
    throw Error(ErrorCode::InvalidRequest, "scope recording requires an operation and an active capture");
  for (auto symbol : inputs)
    validate(symbol);
  checkGrowth(impl_->commands.size(), 1, impl_->options.max_commands, "scope command cap exceeded");
  checkGrowth(impl_->slots.size(), output_count, impl_->options.max_values, "scope value cap exceeded");
  checkGrowth(impl_->input_edges, inputs.size(), impl_->options.max_input_edges, "scope input edge cap exceeded");
  ScopeValues outputs;
  outputs.reserve(output_count);
  for (std::size_t i = 0; i < output_count; ++i)
    outputs.push_back(ScopeValue(impl_->identity, impl_->slots.size() + i));
  const auto cursor = impl_->commands.size();
  impl_->commands.reserve(cursor + 1);
  impl_->slots.reserve(impl_->slots.size() + output_count);
  impl_->commands.push_back({std::move(operation), std::move(inputs), std::move(attributes), outputs});
  for (std::size_t i = 0; i < output_count; ++i)
    impl_->slots.push_back({{}, cursor, absent, false});
  for (auto symbol : impl_->commands.back().inputs)
    impl_->slots[symbol.id_].last_use = cursor;
  impl_->input_edges += impl_->commands.back().inputs.size();
  return outputs;
}

ScopeView::ScopeView(const Scope &scope, std::size_t start, std::size_t length)
  : scope_(&scope), start_(start), length_(length) {}
std::size_t ScopeView::size() const noexcept { return length_; }
std::size_t ScopeView::start() const noexcept { return start_; }
const ScopeCommand &ScopeView::command(std::size_t localOffset) const {
  if (localOffset >= length_)
    throw Error(ErrorCode::InvalidRequest, "scope command offset exceeds view");
  return scope_->impl_->commands[start_ + localOffset];
}
const Value *ScopeView::find(ScopeValue symbol) const {
  scope_->validate(symbol);
  const auto &concrete = scope_->impl_->slots[symbol.id_].concrete;
  return concrete ? &concrete : nullptr;
}
const Value &ScopeView::value(ScopeValue symbol) const {
  const auto *concrete = find(symbol);
  if (!concrete)
    throw Error(ErrorCode::InvalidRequest, "scope symbol is not concrete at this cursor");
  return *concrete;
}
ScopeValues ScopeView::liveIns(std::size_t length) const {
  if (length > length_)
    throw Error(ErrorCode::InvalidRequest, "scope span exceeds view");
  ScopeValues result;
  std::vector<bool> seen(scope_->impl_->slots.size(), false);
  for (std::size_t i = 0; i < length; ++i) {
    for (auto input : command(i).inputs) {
      const auto definition = scope_->impl_->slots[input.id_].definition;
      if ((definition == absent || definition < start_) && !seen[input.id_]) {
        seen[input.id_] = true;
        result.push_back(input);
      }
    }
  }
  return result;
}
ScopeValues ScopeView::liveOuts(std::size_t length) const {
  if (length > length_)
    throw Error(ErrorCode::InvalidRequest, "scope span exceeds view");
  ScopeValues result;
  const auto end = start_ + length;
  for (std::size_t i = 0; i < length; ++i) {
    for (auto output : command(i).outputs) {
      const auto &slot = scope_->impl_->slots[output.id_];
      if (slot.exported || (slot.last_use != absent && slot.last_use >= end))
        result.push_back(output);
    }
  }
  return result; // Recording assigns monotonically increasing output ids.
}

Values Scope::execute(const ScopeValues &exports, const std::vector<std::shared_ptr<const ScopeKernel>> &kernels,
                      const std::function<void(TraceKind, const std::string &, const std::string &)> &record) {
  impl_->capturing = false;
  checkGrowth(0, exports.size(), impl_->options.max_exports, "scope export cap exceeded");
  for (auto symbol : exports)
    validate(symbol);
  for (auto symbol : exports)
    impl_->slots[symbol.id_].exported = true;
  std::size_t attempts = 0;
  for (std::size_t cursor = 0; cursor < impl_->commands.size();) {
    const ScopeKernel *selected = nullptr;
    std::size_t count = 1;
    std::string name = "runtime.submit";
    std::string detail = "start=" + std::to_string(cursor) + " count=0 liveins=0 liveouts=0";
    try {
      if (impl_->options.use_kernels) {
        for (const auto &kernel : kernels) {
          name = kernel->name;
          if (attempts == impl_->options.max_match_attempts)
            throw Error(ErrorCode::BudgetExceeded, "scope match attempt cap exceeded");
          ++attempts;
          const auto window =
            std::min({kernel->max_commands, impl_->options.max_lookahead, impl_->commands.size() - cursor});
          ScopeView candidate(*this, cursor, window);
          detail = "start=" + std::to_string(cursor) + " count=" + std::to_string(window) +
                   " liveins=" + std::to_string(candidate.liveIns(window).size()) +
                   " liveouts=" + std::to_string(candidate.liveOuts(window).size());
          std::string reason;
          std::size_t matched;
          impl_->runtime.setScopeMatching(true);
          try {
            matched = kernel->match(candidate, reason);
          } catch (...) {
            impl_->runtime.setScopeMatching(false);
            throw;
          }
          impl_->runtime.setScopeMatching(false);
          if (matched > window)
            throw Error(ErrorCode::InvalidRequest, "scope matcher returned a span outside its view");
          if (!matched) {
            record(TraceKind::Declined, name, detail + " " + reason);
            continue;
          }
          selected = kernel.get();
          count = matched;
          break;
        }
      }
      if (!selected)
        name = "runtime.submit";
      ScopeView span(*this, cursor, count);
      const auto liveins = span.liveIns(count);
      const auto liveouts = span.liveOuts(count);
      detail = "start=" + std::to_string(cursor) + " count=" + std::to_string(count) +
               " liveins=" + std::to_string(liveins.size()) + " liveouts=" + std::to_string(liveouts.size());
      record(TraceKind::Selected, name, detail);
      Values outputs;
      ScopeValues symbols;
      if (selected) {
        symbols = liveouts;
        outputs = selected->execute(impl_->runtime, span);
      } else {
        const auto &command = span.command(0);
        Request request{command.operation, {}, command.attributes};
        request.inputs.reserve(command.inputs.size());
        for (auto input : command.inputs)
          request.inputs.push_back(span.value(input));
        symbols = command.outputs;
        outputs = impl_->runtime.submit(request);
      }
      if (outputs.size() != symbols.size())
        throw Error(ErrorCode::InvalidRequest, "scope executor returned an incorrect output count");
      for (const auto &output : outputs)
        if (!impl_->runtime.owns(output))
          throw Error(ErrorCode::InvalidRequest, "scope output belongs to another runtime or is empty");
      for (std::size_t i = 0; i < outputs.size(); ++i)
        impl_->slots[symbols[i].id_].concrete = std::move(outputs[i]);
      cursor += count;
      for (auto &slot : impl_->slots)
        if (!slot.exported && (slot.last_use == absent || slot.last_use < cursor))
          slot.concrete = {};
      record(TraceKind::Completed, name, detail);
    } catch (const Error &error) {
      record(TraceKind::Failed, name, detail + " " + error.what());
      throw;
    } catch (const std::exception &error) {
      record(TraceKind::Failed, name, detail + " " + error.what());
      throw Error(ErrorCode::ExecutionFailed, error.what());
    } catch (...) {
      record(TraceKind::Failed, name, detail + " non-standard scope callback exception");
      throw Error(ErrorCode::ExecutionFailed, "non-standard scope callback exception");
    }
  }
  Values result;
  result.reserve(exports.size());
  for (auto symbol : exports) {
    const auto &value = impl_->slots[symbol.id_].concrete;
    if (!value)
      throw Error(ErrorCode::ExecutionFailed, "scope export was not produced");
    result.push_back(value);
  }
  return result;
}
} // namespace fhenomenon::runtime
