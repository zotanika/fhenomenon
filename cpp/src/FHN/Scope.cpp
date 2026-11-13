#include "FHN/Scope.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <unordered_map>
#include <unordered_set>
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
template <class T> void reserveGrowing(std::vector<T> &values, std::size_t required, std::size_t cap) {
  if (required <= values.capacity())
    return;
  const auto capacity = values.capacity();
  const auto increment = capacity / 2 + 1;
  const auto grown = increment > cap - capacity ? cap : capacity + increment;
  values.reserve(std::max(required, grown));
}
void validateDescriptor(const ScopeDescriptor &descriptor) {
  if (descriptor.format.empty() ||
      (descriptor.kind != ResourceKind::Value && descriptor.kind != ResourceKind::Artifact))
    throw Error(ErrorCode::InvalidRequest, "scope descriptor requires a format and a valid resource kind");
}
void validateBase(const Value &value, const ScopeDescriptor &descriptor) {
  validateDescriptor(descriptor);
  if (value.format() != descriptor.format || value.kind() != descriptor.kind || value.bytes() != descriptor.bytes)
    throw Error(ErrorCode::InvalidRequest, "scope resource differs from its declared format, kind or bytes");
}
} // namespace

struct Scope::Impl {
  struct Slot {
    Value concrete;
    std::size_t definition = absent;
    std::size_t last_use = absent;
    bool exported = false;
    ScopeDescriptor descriptor;
  };
  Runtime &runtime;
  ScopeOptions options;
  std::uint64_t identity = nextScopeIdentity();
  bool capturing = true;
  bool in_contract_callback = false;
  const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
  std::size_t input_edges = 0;
  std::vector<ScopeCommand> commands;
  std::vector<Slot> slots;
  std::unordered_map<std::uint64_t, std::size_t> imports;
  Impl(Runtime &owner, ScopeOptions limits) : runtime(owner), options(std::move(limits)) {
    if (options.contract)
      options.contract = std::make_shared<const ScopeContract>(*options.contract);
  }
  struct MetadataGuard {
    Impl &owner;
    bool previous;
    explicit MetadataGuard(Impl &impl) : owner(impl) {
      if (owner.in_contract_callback)
        throw Error(ErrorCode::InvalidRequest, "scope metadata callbacks cannot reenter recording");
      owner.in_contract_callback = true;
      previous = owner.runtime.setScopeMetadata(true);
    }
    ~MetadataGuard() {
      owner.runtime.setScopeMetadata(previous);
      owner.in_contract_callback = false;
    }
  };
  void checkDeadline() const {
    if (options.max_duration_ms) {
      const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
      if (elapsed >= 0 && static_cast<std::uintmax_t>(elapsed) >= options.max_duration_ms)
        throw Error(ErrorCode::BudgetExceeded, "scope duration budget exceeded");
    }
  }
};

Scope::Scope(Runtime &runtime, ScopeOptions options) : impl_(std::make_unique<Impl>(runtime, options)) {
  if (!options.max_commands || !options.max_values || !options.max_input_edges || !options.max_exports ||
      !options.max_lookahead || !options.max_match_attempts)
    throw Error(ErrorCode::InvalidRequest, "scope numeric caps must be positive");
  if (impl_->options.contract &&
      (!impl_->options.contract->describe || !impl_->options.contract->infer || !impl_->options.contract->validate))
    throw Error(ErrorCode::InvalidRequest, "scope contract requires describe, infer and validate callbacks");
}
Scope::~Scope() = default;
bool Scope::belongsTo(const Runtime &runtime) const noexcept { return &impl_->runtime == &runtime; }
void Scope::validate(ScopeValue symbol) const {
  if (symbol.scope_ != impl_->identity || symbol.id_ >= impl_->slots.size())
    throw Error(ErrorCode::InvalidRequest, "invalid symbol or symbol belongs to another scope");
}
ScopeValue Scope::input(const Value &value) {
  if (!impl_->capturing || impl_->in_contract_callback)
    throw Error(ErrorCode::InvalidRequest, "scope recording has finished");
  if (!impl_->runtime.owns(value))
    throw Error(ErrorCode::InvalidRequest, "scope input belongs to another runtime or is empty");
  const auto existing = impl_->imports.find(value.identity());
  if (existing != impl_->imports.end())
    return ScopeValue(impl_->identity, existing->second);
  checkGrowth(impl_->slots.size(), 1, impl_->options.max_values, "scope value cap exceeded");
  ScopeDescriptor descriptor;
  if (impl_->options.contract) {
    Impl::MetadataGuard guard(*impl_);
    descriptor = impl_->options.contract->describe(value);
    validateBase(value, descriptor);
  }
  const auto index = impl_->slots.size();
  reserveGrowing(impl_->slots, index + 1, impl_->options.max_values);
  impl_->imports.emplace(value.identity(), index);
  impl_->slots.push_back({value, absent, absent, false, std::move(descriptor)});
  auto &progress = impl_->runtime.scopeProgress();
  ++progress.values;
  ++progress.imports;
  return ScopeValue(impl_->identity, index);
}
ScopeValues Scope::submit(std::string operation, ScopeValues inputs, std::any attributes, std::size_t output_count) {
  if (!impl_->capturing || impl_->in_contract_callback || operation.empty())
    throw Error(ErrorCode::InvalidRequest, "scope recording requires an operation and an active capture");
  for (auto symbol : inputs)
    validate(symbol);
  checkGrowth(impl_->commands.size(), 1, impl_->options.max_commands, "scope command cap exceeded");
  checkGrowth(impl_->slots.size(), output_count, impl_->options.max_values, "scope value cap exceeded");
  checkGrowth(impl_->input_edges, inputs.size(), impl_->options.max_input_edges, "scope input edge cap exceeded");
  std::vector<ScopeDescriptor> descriptors;
  if (impl_->options.contract) {
    std::vector<ScopeDescriptor> arguments;
    arguments.reserve(inputs.size());
    for (auto symbol : inputs)
      arguments.push_back(impl_->slots[symbol.id_].descriptor);
    Impl::MetadataGuard guard(*impl_);
    descriptors = impl_->options.contract->infer(operation, arguments, attributes);
    if (descriptors.size() != output_count)
      throw Error(ErrorCode::InvalidRequest, "scope contract inferred an incorrect output count");
    for (const auto &descriptor : descriptors)
      validateDescriptor(descriptor);
  }
  ScopeValues outputs;
  outputs.reserve(output_count);
  for (std::size_t i = 0; i < output_count; ++i)
    outputs.push_back(ScopeValue(impl_->identity, impl_->slots.size() + i));
  const auto cursor = impl_->commands.size();
  reserveGrowing(impl_->commands, cursor + 1, impl_->options.max_commands);
  reserveGrowing(impl_->slots, impl_->slots.size() + output_count, impl_->options.max_values);
  impl_->commands.push_back({std::move(operation), std::move(inputs), std::move(attributes), outputs});
  for (std::size_t i = 0; i < output_count; ++i)
    impl_->slots.push_back(
      {{}, cursor, absent, false, impl_->options.contract ? std::move(descriptors[i]) : ScopeDescriptor{}});
  for (auto symbol : impl_->commands.back().inputs)
    impl_->slots[symbol.id_].last_use = cursor;
  impl_->input_edges += impl_->commands.back().inputs.size();
  auto &progress = impl_->runtime.scopeProgress();
  ++progress.commands;
  progress.values += output_count;
  progress.input_edges = impl_->input_edges;
  return outputs;
}
const ScopeDescriptor &Scope::descriptor(ScopeValue symbol) const {
  validate(symbol);
  if (!impl_->options.contract)
    throw Error(ErrorCode::InvalidRequest, "scope descriptors require a contract");
  return impl_->slots[symbol.id_].descriptor;
}

ScopeView::ScopeView(const Scope &scope, std::size_t start, std::size_t length)
  : scope_(&scope), start_(start), length_(length) {}
std::size_t ScopeView::size() const noexcept { return length_; }
std::size_t ScopeView::start() const noexcept { return start_; }
ScopeView ScopeView::subview(std::size_t localOffset, std::size_t length) const {
  if (localOffset > length_ || length > length_ - localOffset)
    throw Error(ErrorCode::InvalidRequest, "scope subview exceeds view");
  return ScopeView(*scope_, start_ + localOffset, length);
}
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
const ScopeDescriptor &ScopeView::descriptor(ScopeValue symbol) const { return scope_->descriptor(symbol); }
ScopeValues ScopeView::liveIns(std::size_t length) const {
  if (length > length_)
    throw Error(ErrorCode::InvalidRequest, "scope span exceeds view");
  ScopeValues result;
  std::unordered_set<std::size_t> seen;
  for (std::size_t i = 0; i < length; ++i) {
    for (auto input : command(i).inputs) {
      const auto definition = scope_->impl_->slots[input.id_].definition;
      if ((definition == absent || definition < start_) && seen.insert(input.id_).second) {
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

void Scope::finishCapture(const ScopeValues &exports) {
  impl_->capturing = false;
  checkGrowth(0, exports.size(), impl_->options.max_exports, "scope export cap exceeded");
  for (auto symbol : exports)
    validate(symbol);
  for (auto symbol : exports)
    impl_->slots[symbol.id_].exported = true;
  impl_->runtime.scopeProgress().exports = exports.size();
  impl_->checkDeadline();
}

void Scope::inspect(const ScopeValues &exports, const std::function<void(const ScopeView &)> &callback) {
  finishCapture(exports);
  callback(ScopeView(*this, 0, impl_->commands.size()));
  impl_->checkDeadline();
}

Values Scope::execute(const ScopeValues &exports, const std::vector<std::shared_ptr<const ScopeKernel>> &kernels,
                      const std::function<void(TraceKind, const std::string &, const std::string &)> &record) {
  finishCapture(exports);
  // Every symbol is scheduled once, at its last use (or definition if dead).
  // This keeps reclamation linear in recorded structure rather than scanning
  // all values after every command. Span execution drains all covered buckets.
  std::vector<std::vector<std::size_t>> releases(impl_->commands.size());
  for (std::size_t i = 0; i < impl_->slots.size(); ++i) {
    const auto &slot = impl_->slots[i];
    if (!slot.exported && !releases.empty()) {
      const auto last = slot.last_use != absent ? slot.last_use : slot.definition != absent ? slot.definition : 0;
      releases[last].push_back(i);
    }
  }
  std::size_t attempts = 0;
  for (std::size_t cursor = 0; cursor < impl_->commands.size();) {
    const ScopeKernel *selected = nullptr;
    std::size_t count = 1;
    std::string name = "runtime.submit";
    std::string detail = "start=" + std::to_string(cursor) + " count=0 liveins=0 liveouts=0";
    try {
      impl_->checkDeadline();
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
      impl_->checkDeadline();
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
      if (impl_->options.contract) {
        Impl::MetadataGuard guard(*impl_);
        for (std::size_t i = 0; i < outputs.size(); ++i) {
          const auto &descriptor = impl_->slots[symbols[i].id_].descriptor;
          validateBase(outputs[i], descriptor);
          impl_->options.contract->validate(outputs[i], descriptor);
        }
      }
      for (std::size_t i = 0; i < outputs.size(); ++i)
        impl_->slots[symbols[i].id_].concrete = std::move(outputs[i]);
      const auto end = cursor + count;
      for (; cursor < end; ++cursor)
        for (const auto id : releases[cursor])
          impl_->slots[id].concrete = {};
      auto &progress = impl_->runtime.scopeProgress();
      progress.executed_commands += count;
      if (selected)
        ++progress.kernel_spans;
      else
        ++progress.recipe_spans;
      record(TraceKind::Completed, name, detail);
      impl_->checkDeadline();
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
