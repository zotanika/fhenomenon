#pragma once

#include "FHN/Runtime.h"

#include <limits>

namespace fhenomenon::runtime {

// An opaque, scope-local symbol. It cannot reveal or convert to a concrete Value.
class ScopeValue {
  public:
  ScopeValue() noexcept = default;
  std::size_t id() const noexcept { return id_; }
  friend bool operator==(ScopeValue a, ScopeValue b) noexcept { return a.scope_ == b.scope_ && a.id_ == b.id_; }
  friend bool operator!=(ScopeValue a, ScopeValue b) noexcept { return !(a == b); }

  private:
  std::uint64_t scope_ = 0;
  std::size_t id_ = std::numeric_limits<std::size_t>::max();
  ScopeValue(std::uint64_t scope, std::size_t id) : scope_(scope), id_(id) {}
  friend class Scope;
  friend class ScopeView;
};
using ScopeValues = std::vector<ScopeValue>;

// Limits apply independently to each runScope()/native Session::run() call.
// All numeric fields must be positive, even when use_kernels is false.
// These bound recorded structure and matcher calls, not memory bytes or CPU time;
// use RuntimeOptions::max_live_bytes for charged ciphertext/artifact storage.
struct ScopeOptions {
  // Recorded submit() calls, including commands declaring zero outputs.
  std::size_t max_commands = 256;
  // Distinct imported identities + all declared output symbols, including dead ones.
  // Repeated input() calls for the same concrete Value count only once.
  std::size_t max_values = 2048;
  // Sum of command input counts; repeated uses of one symbol each count as an edge.
  std::size_t max_input_edges = 4096;
  // Entries returned by the capture callback; duplicate exports each count.
  std::size_t max_exports = 2048;
  // Maximum commands visible to one matcher, also limited by kernel.max_commands
  // and the remaining sequence. Shortens the view; it is not an overflow error.
  std::size_t max_lookahead = 32;
  // Total ScopeKernel::match calls over the whole run, including declines.
  // Needing another call after this limit raises BudgetExceeded, without fallback.
  std::size_t max_match_attempts = 4096;
  // False skips span matching but keeps capture, liveness and ordinary recipe
  // selection. Prepared/fused recipes and explicitly supplied state still work.
  bool use_kernels = true;
};

struct ScopeCommand {
  std::string operation;
  ScopeValues inputs;
  // Supply an owning immutable snapshot; borrowed mutable storage is unsupported.
  std::any attributes;
  ScopeValues outputs;
};

class Scope;
// Views are borrowed only for the duration of a matcher or executor callback.
class ScopeView {
  public:
  std::size_t size() const noexcept;
  std::size_t start() const noexcept;
  const ScopeCommand &command(std::size_t localOffset) const;
  const Value *find(ScopeValue symbol) const;
  const Value &value(ScopeValue symbol) const;
  ScopeValues liveIns(std::size_t length) const;
  ScopeValues liveOuts(std::size_t length) const;

  private:
  const Scope *scope_;
  std::size_t start_;
  std::size_t length_;
  ScopeView(const Scope &scope, std::size_t start, std::size_t length);
  friend class Scope;
};

struct ScopeKernel {
  std::string name;
  int priority = 0;
  std::size_t max_commands = 1;
  std::function<std::size_t(const ScopeView &, std::string &reason)> match;
  // Return exactly view.liveOuts(view.size()), in that symbolic order.
  std::function<Values(Runtime &, const ScopeView &)> execute;
};

class Scope {
  public:
  ~Scope();
  bool belongsTo(const Runtime &runtime) const noexcept;
  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;
  ScopeValue input(const Value &value);
  ScopeValues submit(std::string operation, ScopeValues inputs, std::any attributes = {}, std::size_t output_count = 1);

  private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Scope(Runtime &runtime, ScopeOptions options);
  Values execute(const ScopeValues &exports, const std::vector<std::shared_ptr<const ScopeKernel>> &kernels,
                 const std::function<void(TraceKind, const std::string &, const std::string &)> &record);
  void validate(ScopeValue symbol) const;
  friend class Runtime;
  friend class ScopeView;
};

} // namespace fhenomenon::runtime
