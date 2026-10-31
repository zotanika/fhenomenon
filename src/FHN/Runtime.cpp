#include "FHN/Runtime.h"

#include <algorithm>
#include <atomic>
#include <deque>
#include <limits>
#include <utility>

namespace fhenomenon::runtime {
namespace {
struct Accounting {
  RuntimeOptions options;
  RuntimeStats stats;
};

// A charge reserves space before invoking provider code and releases it after
// the resource's payload and dependencies have been destroyed.
struct Charge {
  std::shared_ptr<Accounting> owner;
  std::size_t bytes;

  Charge(std::shared_ptr<Accounting> accounting, std::size_t amount) : owner(std::move(accounting)), bytes(amount) {
    auto &stats = owner->stats;
    const auto limit = owner->options.max_live_bytes;
    if (bytes > std::numeric_limits<std::size_t>::max() - stats.live_bytes ||
        (limit && (stats.live_bytes > limit || bytes > limit - stats.live_bytes))) {
      throw Error(ErrorCode::BudgetExceeded, "retained resource budget exceeded");
    }
    ++stats.live_resources;
    stats.live_bytes += bytes;
    stats.peak_live_bytes = std::max(stats.peak_live_bytes, stats.live_bytes);
  }
  ~Charge() {
    owner->stats.live_bytes -= bytes;
    --owner->stats.live_resources;
  }
  Charge(const Charge &) = delete;
  Charge &operator=(const Charge &) = delete;
};

std::uint64_t nextIdentity() {
  // Identities remain distinct even if a runtime dies while handles survive.
  static std::atomic<std::uint64_t> next{1};
  auto candidate = next.load(std::memory_order_relaxed);
  for (;;) {
    if (candidate == std::numeric_limits<std::uint64_t>::max()) {
      throw Error(ErrorCode::ExecutionFailed, "resource identities exhausted");
    }
    if (next.compare_exchange_weak(candidate, candidate + 1, std::memory_order_relaxed))
      return candidate;
  }
}
} // namespace

struct Value::Resource {
  std::shared_ptr<Charge> charge;
  std::uint64_t identity;
  std::string format;
  ResourceKind kind;
  Values dependencies;
  // Provider cleanup may still use dependency storage. Destroy payload first.
  std::any payload;

  Resource(std::shared_ptr<Charge> reserved, std::string resource_format, ResourceKind resource_kind,
           std::any resource_payload, Values retained)
    : charge(std::move(reserved)), identity(nextIdentity()), format(std::move(resource_format)), kind(resource_kind),
      dependencies(std::move(retained)), payload(std::move(resource_payload)) {}
};

struct Runtime::Impl {
  std::shared_ptr<Accounting> accounting;
  std::vector<std::shared_ptr<const Recipe>> recipes;
  std::deque<TraceEvent> events;

  explicit Impl(RuntimeOptions options) : accounting(std::make_shared<Accounting>(Accounting{options, {}})) {}

  void record(TraceKind kind, const std::string &operation, const std::string &recipe, const std::string &detail = {}) {
    const auto limit = accounting->options.max_trace_events;
    if (!limit)
      return;
    if (events.size() == limit)
      events.pop_front();
    events.push_back({kind, operation, recipe, detail});
  }
};

Error::Error(ErrorCode code, const std::string &message) : std::runtime_error(message), code_(code) {}
ErrorCode Error::code() const noexcept { return code_; }

Value::Value(std::shared_ptr<const Resource> resource) : resource_(std::move(resource)) {}
Value::operator bool() const noexcept { return bool(resource_); }
const Value::Resource &Value::get() const {
  if (!resource_)
    throw Error(ErrorCode::InvalidRequest, "empty resource handle");
  return *resource_;
}
std::uint64_t Value::identity() const { return get().identity; }
const std::string &Value::format() const { return get().format; }
ResourceKind Value::kind() const { return get().kind; }
std::size_t Value::bytes() const { return get().charge->bytes; }
const std::any &Value::payload() const { return get().payload; }
const Values &Value::dependencies() const { return get().dependencies; }

Runtime::Runtime(RuntimeOptions options) : impl_(std::make_unique<Impl>(options)) {}
Runtime::~Runtime() = default;

void Runtime::addRecipe(Recipe recipe) {
  if (recipe.operation.empty() || recipe.name.empty() || !recipe.accepts || !recipe.execute) {
    throw Error(ErrorCode::InvalidRequest, "recipe requires operation, name and callbacks");
  }
  auto registered = std::make_shared<const Recipe>(std::move(recipe));
  const auto position = std::find_if(impl_->recipes.begin(), impl_->recipes.end(),
                                     [&](const auto &existing) { return existing->priority < registered->priority; });
  impl_->recipes.insert(position, std::move(registered));
}

Values Runtime::submit(const Request &request) {
  if (request.operation.empty()) {
    throw Error(ErrorCode::InvalidRequest, "request requires an operation");
  }
  const auto validate = [&](const Values &values) {
    for (const auto &value : values) {
      if (!value || value.resource_->charge->owner != impl_->accounting) {
        throw Error(ErrorCode::InvalidRequest, "resource handle belongs to another runtime or is empty");
      }
    }
  };
  validate(request.inputs);
  // Snapshot registrations so a callback can register future recipes without
  // invalidating this dispatch or changing its fallback order.
  const auto candidates = impl_->recipes;
  for (const auto &recipe : candidates) {
    if (recipe->operation != request.operation)
      continue;
    try {
      std::string reason;
      if (!recipe->accepts(request, reason)) {
        impl_->record(TraceKind::Declined, request.operation, recipe->name, reason);
        continue;
      }
      impl_->record(TraceKind::Selected, request.operation, recipe->name);
      auto outputs = recipe->execute(*this, request);
      validate(outputs);
      impl_->record(TraceKind::Completed, request.operation, recipe->name);
      return outputs;
    } catch (const Error &error) {
      impl_->record(TraceKind::Failed, request.operation, recipe->name, error.what());
      throw;
    } catch (const std::exception &error) {
      impl_->record(TraceKind::Failed, request.operation, recipe->name, error.what());
      throw Error(ErrorCode::ExecutionFailed, error.what());
    } catch (...) {
      impl_->record(TraceKind::Failed, request.operation, recipe->name, "non-standard callback exception");
      throw Error(ErrorCode::ExecutionFailed, "non-standard callback exception");
    }
  }
  impl_->record(TraceKind::Failed, request.operation, {}, "no applicable recipe");
  throw Error(ErrorCode::Unsupported, "no applicable recipe for " + request.operation);
}

Value Runtime::makeValue(std::string format, ResourceKind kind, std::size_t bytes,
                         const std::function<std::any()> &factory, Values dependencies) {
  if (!factory)
    throw Error(ErrorCode::InvalidRequest, "resource requires a factory");
  for (const auto &dependency : dependencies) {
    if (!dependency || dependency.resource_->charge->owner != impl_->accounting) {
      throw Error(ErrorCode::InvalidRequest, "dependency belongs to another runtime or is empty");
    }
  }
  auto charge = std::make_shared<Charge>(impl_->accounting, bytes);
  try {
    auto payload = factory();
    return Value(std::make_shared<const Value::Resource>(std::move(charge), std::move(format), kind, std::move(payload),
                                                         std::move(dependencies)));
  } catch (const Error &) {
    throw;
  } catch (const std::exception &error) {
    throw Error(ErrorCode::ExecutionFailed, error.what());
  } catch (...) {
    throw Error(ErrorCode::ExecutionFailed, "non-standard resource factory exception");
  }
}

RuntimeStats Runtime::stats() const noexcept { return impl_->accounting->stats; }
std::vector<TraceEvent> Runtime::trace() const { return {impl_->events.begin(), impl_->events.end()}; }
} // namespace fhenomenon::runtime
