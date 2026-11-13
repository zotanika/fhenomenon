#pragma once

#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fhenomenon::runtime {

enum class ResourceKind { Value, Artifact };
enum class ErrorCode { Unsupported, InvalidRequest, BudgetExceeded, ExecutionFailed };

class Error : public std::runtime_error {
  public:
  Error(ErrorCode code, const std::string &message);
  ErrorCode code() const noexcept;

  private:
  ErrorCode code_;
};

class Value;
using Values = std::vector<Value>;

// Copies share one immutable resource and one accounting charge. Payloads may
// contain shared provider objects; providers must not mutate escaped inputs.
class Value {
  public:
  Value() noexcept = default;
  explicit operator bool() const noexcept;
  std::uint64_t identity() const;
  const std::string &format() const;
  ResourceKind kind() const;
  std::size_t bytes() const;
  const std::any &payload() const;
  const Values &dependencies() const;

  private:
  struct Resource;
  std::shared_ptr<const Resource> resource_;
  explicit Value(std::shared_ptr<const Resource> resource);
  const Resource &get() const;
  friend class Runtime;
};

struct Request {
  std::string operation;
  Values inputs;
  std::any attributes;
};

class Runtime;
class Scope;
class ScopeValue;
class ScopeView;
struct ScopeOptions;
struct ScopeKernel;
struct Recipe {
  std::string operation;
  std::string name;
  int priority = 0; // Higher priority first; registration order breaks ties.
  std::function<bool(const Request &, std::string &reason)> accepts;
  std::function<Values(Runtime &, const Request &)> execute;
};

struct RuntimeOptions {
  std::size_t max_live_bytes = 0;     // Zero means unlimited retained resources.
  std::size_t max_trace_events = 256; // Zero disables trace retention.
};

struct RuntimeStats {
  std::size_t live_bytes = 0;
  std::size_t peak_live_bytes = 0;
  std::size_t live_resources = 0;
};

// The most recent admitted runScope/inspectScope call, including progress before
// failure. Capture counts include only successfully recorded structure. Execution
// counts include only spans whose outputs passed validation. Inspection executes
// no spans. An invalid nested call does not reset the enclosing scope's report.
struct ScopeStats {
  std::size_t commands = 0;
  std::size_t values = 0;
  std::size_t imports = 0;
  std::size_t input_edges = 0;
  std::size_t exports = 0;
  std::size_t executed_commands = 0;
  std::size_t recipe_spans = 0;
  std::size_t kernel_spans = 0;
};

enum class TraceKind { Declined, Selected, Completed, Failed };
struct TraceEvent {
  TraceKind kind;
  std::string operation;
  std::string recipe;
  std::string detail;
};

// Synchronous and single-threaded. Retained resource handles may outlive this
// object. Budgets exclude callback temporaries, keys, scratch and allocator cost.
class Runtime {
  public:
  explicit Runtime(RuntimeOptions options = {});
  ~Runtime();
  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;
  Runtime(Runtime &&) = delete;
  Runtime &operator=(Runtime &&) = delete;

  void addRecipe(Recipe recipe);
  bool owns(const Value &value) const noexcept;
  void addScopeKernel(ScopeKernel kernel);
  Values runScope(const std::function<std::vector<ScopeValue>(Scope &)> &capture);
  Values runScope(const std::function<std::vector<ScopeValue>(Scope &)> &capture, ScopeOptions options);
  // Capture exactly once, validate exports, then inspect the complete recorded
  // graph without matching or executing recipes/kernels. During both callbacks
  // runtime mutation and nested scopes are rejected; capture may record symbols.
  // The immutable view and its references are borrowed only until inspect returns.
  void inspectScope(const std::function<std::vector<ScopeValue>(Scope &)> &capture,
                    const std::function<void(const ScopeView &)> &inspect);
  void inspectScope(const std::function<std::vector<ScopeValue>(Scope &)> &capture,
                    const std::function<void(const ScopeView &)> &inspect, ScopeOptions options);
  Values submit(const Request &request);
  Value makeValue(std::string format, ResourceKind kind, std::size_t bytes, const std::function<std::any()> &factory,
                  Values dependencies = {});
  RuntimeStats stats() const noexcept;
  ScopeStats scopeStats() const noexcept;
  std::vector<TraceEvent> trace() const;

  private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  void setScopeMatching(bool matching) noexcept;
  bool setScopeMetadata(bool active) noexcept;
  ScopeStats &scopeProgress() noexcept;
  friend class Scope;
};

} // namespace fhenomenon::runtime
