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
  Values submit(const Request &request);
  Value makeValue(std::string format, ResourceKind kind, std::size_t bytes, const std::function<std::any()> &factory,
                  Values dependencies = {});
  RuntimeStats stats() const noexcept;
  std::vector<TraceEvent> trace() const;

  private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace fhenomenon::runtime
